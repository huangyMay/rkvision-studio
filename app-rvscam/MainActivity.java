package com.example.camera2;

import android.Manifest;
import android.annotation.SuppressLint;
import android.content.pm.PackageManager;
import android.media.MediaCodec;
import android.media.MediaCodecInfo;
import android.media.MediaFormat;
import android.os.Bundle;
import android.util.Log;
import android.text.InputType;
import android.view.Surface;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import java.io.IOException;

/**
 * RVS 通用相机推流：Camera2 选源（默认 HDMI-in）→ NV21 → MediaCodec 硬编 H264
 * → RTSP 推流（ANNOUNCE/SETUP/PLAY，TCP interleaved）。
 *
 * 目标地址：默认 127.0.0.1:8554（USB 模式配合 `adb reverse tcp:8554 tcp:8554`）；
 * 走网线时改为 PC 网卡 IP:8554，直连 mediamtx。
 */
public class MainActivity extends AppCompatActivity {
    static { System.loadLibrary("camera2"); }

    private static final String TAG = "RvsCam";
    private static final int REQ_PERM = 1;

    private SurfaceView mPreview;
    private TextView mStatus;
    private EditText mTarget;
    private android.widget.Spinner mCameraSel;
    private Button mBtnStart, mBtnStop;
    private HdmiCameraHelper mCameraHelper;
    private RtspPushClient mRtsp;

    private volatile boolean mRunning = false;    // 相机采集运行
    private volatile boolean mStreaming = false;  // 推流运行
    private Thread mEncoderThread;

    // 最新帧获胜：编码慢于采集时丢弃旧帧
    private final java.util.concurrent.atomic.AtomicReference<byte[]> mPendingFrame =
            new java.util.concurrent.atomic.AtomicReference<>(null);
    private volatile int mFrameW = 0, mFrameH = 0;

    private void setStatus(final String s) {
        runOnUiThread(() -> mStatus.setText(s));
    }

    // ---------------- 相机回调 ----------------

    private void onFrame(byte[] nv21, int w, int h) {
        if (!mRunning) return;
        mFrameW = w; mFrameH = h;
        if (mStreaming) mPendingFrame.set(nv21);
    }

    // ---------------- 生命周期 ----------------

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        buildUi();
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA}, REQ_PERM);
        } else {
            startCamera();
        }
    }

    private void buildUi() {
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setPadding(32, 32, 32, 32);
        root.setBackgroundColor(0xFF0D1117);

        mStatus = new TextView(this);
        mStatus.setText("等待 HDMI 输入…\n目标默认 127.0.0.1:8554（USB 模式需 adb reverse）");
        mStatus.setTextColor(0xFF00E5FF);
        mStatus.setTextSize(14);
        root.addView(mStatus);

        mTarget = new EditText(this);
        mTarget.setHint("推流目标 IP:PORT（如 192.168.1.10:8554）");
        mTarget.setText(getSharedPreferences("rvs", MODE_PRIVATE)
                .getString("target", "127.0.0.1:8554"));
        mTarget.setInputType(InputType.TYPE_TEXT_VARIATION_URI);
        mTarget.setFocusableInTouchMode(true);       // 防止启动即弹软键盘
        mTarget.setImeOptions(mTarget.getImeOptions() | android.view.inputmethod.EditorInfo.IME_FLAG_NO_FULLSCREEN);
        root.addView(mTarget);

        mCameraSel = new Spinner(this);
        root.addView(mCameraSel);
        populateCameras();

        LinearLayout row = new LinearLayout(this);
        mBtnStart = new Button(this); mBtnStart.setText("▶ 开始推流");
        mBtnStop = new Button(this); mBtnStop.setText("■ 停止");
        mBtnStop.setEnabled(false);
        row.addView(mBtnStart, new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1));
        row.addView(mBtnStop, new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1));
        root.addView(row);

        mPreview = new SurfaceView(this);
        root.addView(mPreview, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, 0, 1));

        setContentView(root);

        mBtnStart.setOnClickListener(v -> onStartClick());
        mBtnStop.setOnClickListener(v -> onStopClick());
    }

    @SuppressLint("MissingPermission")
    private void startCamera() {
        setStatus("枚举相机…");
        mRunning = true;
        // 等 SurfaceView 真正创建再开相机——App 刚启动时 surface 未就绪，
        // 直接把 getSurface() 传给采集层会变成 analysis-only（预览黑屏）
        SurfaceHolder holder = mPreview.getHolder();
        if (holder.getSurface() != null && holder.getSurface().isValid()) {
            openCameraNow();
        } else {
            holder.addCallback(new SurfaceHolder.Callback() {
                @Override public void surfaceCreated(@NonNull SurfaceHolder h) { openCameraNow(); }
                @Override public void surfaceChanged(@NonNull SurfaceHolder h, int f, int w, int hh) {}
                @Override public void surfaceDestroyed(@NonNull SurfaceHolder h) {}
            });
        }
    }

    private void openCameraNow() {
        mCameraHelper = new HdmiCameraHelper(this, mPreview.getHolder().getSurface(), this::onFrame);
        mCameraHelper.setStatusListener((connected, w, h, id) -> {
            if (!mRunning) return;
            setStatus(connected
                    ? String.format("HDMI 已连接 %dx%d · camera %s · 就绪", w, h, id)
                    : "等待 HDMI 输入…");
            // 相机就绪即自动开始推流（appliance 场景免手点）
            if (connected && !mStreaming) onStartClick();
        });
        mCameraHelper.start();
    }

    @Override
    public void onRequestPermissionsResult(int code, @NonNull String[] perms, @NonNull int[] results) {
        super.onRequestPermissionsResult(code, perms, results);
        if (code == REQ_PERM && results.length > 0 && results[0] == PackageManager.PERMISSION_GRANTED) {
            startCamera();
        } else {
            Toast.makeText(this, "需要相机权限（HDMI-in 经相机 HAL 暴露）", Toast.LENGTH_LONG).show();
        }
    }

    // ---------------- 推流 ----------------


    private void populateCameras() {
        android.hardware.camera2.CameraManager cm =
                (android.hardware.camera2.CameraManager) getSystemService(Context.CAMERA_SERVICE);
        java.util.ArrayList<String> items = new java.util.ArrayList<>();
        items.add("自动选 HDMI（EXTERNAL/最大 id）");
        try {
            for (String id : cm.getCameraIdList()) {
                android.hardware.camera2.CameraCharacteristics cc = cm.getCameraCharacteristics(id);
                Integer facing = cc.get(android.hardware.camera2.CameraCharacteristics.LENS_FACING);
                String facingName = facing == null ? "?" : (facing == 2 ? "EXTERNAL"
                        : facing == 0 ? "BACK" : facing == 1 ? "FRONT" : String.valueOf(facing));
                items.add("camera " + id + " (" + facingName + ")");
            }
        } catch (Exception e) {
            android.util.Log.e(TAG, "populateCameras", e);
        }
        android.widget.ArrayAdapter<String> ad = new android.widget.ArrayAdapter<>(this,
                android.R.layout.simple_spinner_dropdown_item, items);
        mCameraSel.setAdapter(ad);
        mCameraSel.setSelection(0);
    }

    private void onStartClick() {
        if (mStreaming) return;
        final String target = mTarget.getText().toString().trim();
        if (target.isEmpty()) { setStatus("目标不能为空"); return; }
        getSharedPreferences("rvs", MODE_PRIVATE).edit()
                .putString("target", target).apply();
        final String camId = (String) mCameraSel.getSelectedItem();
        if (mCameraHelper != null && camId != null && !camId.startsWith("自动")
                && !camId.isEmpty()) {
            mCameraHelper.setPreferredCameraId(camId.split(" ")[1] == null
                    ? camId : camId.split(" ")[1]);
        }

        final String[] hp = target.split(":");
        final String host = hp[0];
        final int port = hp.length > 1 ? Integer.parseInt(hp[1]) : 8554;

        mStreaming = true;
        mBtnStart.setEnabled(false);
        mBtnStop.setEnabled(true);
        setStatus("推流启动中 → " + target + " …");

        mEncoderThread = new Thread(() -> encodeLoop(host, port), "encoder");
        mEncoderThread.start();
    }

    private void onStopClick() {
        mStreaming = false;
        mBtnStop.setEnabled(false);
        if (mRtsp != null) mRtsp.stop();
        setStatus("已停止");
        mBtnStart.setEnabled(true);
    }

    /** 编码线程：首帧到达时初始化编码器；csd 就绪后发起 RTSP 握手 */
    private void encodeLoop(String host, int port) {
        android.os.Process.setThreadPriority(android.os.Process.THREAD_PRIORITY_DISPLAY);
        MediaCodec enc = null;
        RtspPushClient rtsp = null;
        long ptsUs = 0;
        final long STEP = 40000;   // 25fps

        try {
            int dbg = 0, dbgIn = 0, dbgOut = 0, dbgFmt = 0, dbgNoFrame = 0;
            while (mStreaming) {
                byte[] nv21 = mPendingFrame.getAndSet(null);
                if (nv21 == null) {
                    if (++dbgNoFrame % 150 == 0)
                        Log.i(TAG, "enc: no frame (in=" + dbgIn + " out=" + dbgOut
                                + " fmt=" + dbgFmt + " empty=" + dbgNoFrame + ")");
                    Thread.sleep(3); continue;
                }
                final int w = mFrameW, h = mFrameH;

                if (enc == null) {
                    enc = createEncoder(w, h);
                    setStatus(String.format("编码器就绪 %dx%d，等待 csd…", w, h));
                }

                // NV21 → NV12（UV 互换）
                byte[] nv12 = new byte[nv21.length];
                System.arraycopy(nv21, 0, nv12, 0, w * h);
                final int uvLen = w * h / 2;
                for (int i = 0; i < uvLen; i += 2) {
                    nv12[w * h + i] = nv21[w * h + i + 1];
                    nv12[w * h + i + 1] = nv21[w * h + i];
                }

                int inIdx = enc.dequeueInputBuffer(10000);
                if (inIdx >= 0) {
                    java.nio.ByteBuffer ib = enc.getInputBuffer(inIdx);
                    ib.clear();
                    ib.put(nv12);                    // direct buffer：必须 put 不可 array()
                    ptsUs += STEP;
                    enc.queueInputBuffer(inIdx, 0, nv12.length, ptsUs, 0);
                    dbgIn++;
                }

                MediaCodec.BufferInfo info = new MediaCodec.BufferInfo();
                int outIdx = enc.dequeueOutputBuffer(info, 10000);
                if (outIdx == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED) {
                    dbgFmt++;
                    MediaFormat of = enc.getOutputFormat();
                    byte[] sps = of.getByteBuffer("csd-0") != null
                            ? bytes(of.getByteBuffer("csd-0")) : new byte[0];
                    byte[] pps = of.getByteBuffer("csd-1") != null
                            ? bytes(of.getByteBuffer("csd-1")) : new byte[0];
                    rtsp = new RtspPushClient(host, port, "/hdmiin", sps, pps,
                            s -> setStatus("RTSP: " + s));
                    rtsp.start();
                } else if (outIdx >= 0) {
                    dbgOut++;
                    java.nio.ByteBuffer ob = enc.getOutputBuffer(outIdx);
                    if (ob != null && info.size > 0 && rtsp != null) {
                        // 绝对索引读取，不动 position/limit（capacity 与 offset/size 不一致时 clamp）
                        int off = Math.min(info.offset, ob.capacity());
                        int len = Math.min(info.size, ob.capacity() - off);
                        byte[] out = new byte[len];
                        for (int i = 0; i < len; i++) out[i] = ob.get(off + i);
                        rtsp.pushFrame(out, info.presentationTimeUs,
                                (info.flags & MediaCodec.BUFFER_FLAG_KEY_FRAME) != 0);
                    }
                    enc.releaseOutputBuffer(outIdx, false);
                }

                if (++dbg % 150 == 0)
                    Log.i(TAG, "enc: frames=" + dbg + " in=" + dbgIn + " out=" + dbgOut
                            + " fmt=" + dbgFmt);
            }
        } catch (Throwable e) {
            java.io.StringWriter sw = new java.io.StringWriter();
            e.printStackTrace(new java.io.PrintWriter(sw));
            setStatus("推流线程异常: " + e);
            android.util.Log.e(TAG, "encodeLoop", e);
        } finally {
            if (rtsp != null) rtsp.stop();
            if (enc != null) { try { enc.stop(); enc.release(); } catch (Exception ignored) {} }
            runOnUiThread(() -> { if (!mStreaming) setStatus("已停止"); });
        }
    }

    private static byte[] bytes(java.nio.ByteBuffer bb) {
        bb.clear();
        byte[] out = new byte[bb.remaining()];
        bb.get(out);
        return out;
    }

    private static byte[] bytesAt(java.nio.ByteBuffer bb, int off, int len) {
        bb.position(off);
        bb.limit(off + len);
        byte[] out = new byte[len];
        bb.get(out);
        bb.clear();
        return out;
    }

    private MediaCodec createEncoder(int w, int h) throws IOException {
        MediaCodec enc = MediaCodec.createEncoderByType("video/avc");
        MediaFormat f = MediaFormat.createVideoFormat("video/avc", w, h);
        f.setInteger(MediaFormat.KEY_COLOR_FORMAT,
                MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420SemiPlanar);
        f.setInteger(MediaFormat.KEY_BIT_RATE, 4_000_000);
        f.setInteger(MediaFormat.KEY_FRAME_RATE, 25);
        f.setInteger(MediaFormat.KEY_I_FRAME_INTERVAL, 2);
        enc.configure(f, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE);
        enc.start();
        return enc;
    }

    @Override
    protected void onDestroy() {
        mStreaming = false;
        mRunning = false;
        if (mCameraHelper != null) mCameraHelper.release();
        if (mEncoderThread != null) mEncoderThread.interrupt();
        super.onDestroy();
    }
}
