package com.example.camera2;

import android.Manifest;
import android.content.Context;
import android.content.pm.PackageManager;
import android.graphics.ImageFormat;
import android.graphics.SurfaceTexture;
import android.hardware.camera2.CameraCaptureSession;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraDevice;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.CameraManager.AvailabilityCallback;
import android.hardware.camera2.CaptureRequest;
import android.hardware.camera2.params.StreamConfigurationMap;
import android.media.Image;
import android.media.ImageReader;
import android.os.Handler;
import android.os.HandlerThread;
import android.util.Log;
import android.util.Size;
import android.view.Surface;
import android.view.Surface;

import androidx.annotation.NonNull;
import androidx.core.content.ContextCompat;

import java.util.ArrayList;
import java.util.List;

/**
 * HDMI 输入采集（RK3576）。
 *
 * 板子的 HDMI-in 由相机 HAL 暴露为 Camera2 外置相机（LENS_FACING_EXTERNAL），
 * 因此全程只走标准 Camera2 API——不依赖 rockchip.hardware.hdmi HIDL、
 * 不直接打开 /dev/v4l-subdev*（那是系统应用才有的 SELinux 权限）。
 *
 * 热插拔：HDMI 插入/拔出时 hdmirx 相机在 CameraService 里出现/消失，
 * 通过 AvailabilityCallback 感知；分辨率切换会让 capture session 报
 * disconnect，延时后自动重开（对齐 rkCamera2 demo 的 2s 策略）。
 */
public class HdmiCameraHelper {
    private static final String TAG = "HdmiCam";
    private static final int PREVIEW_W = 1920, PREVIEW_H = 1080;
    private static final int ANALYSIS_W = 1280, ANALYSIS_H = 720;
    private static final long ANALYSIS_INTERVAL_MS = 30; // 推理线程自节奏（~50ms），此处仅防突发
    private static final long REOPEN_DELAY_MS = 2000;

    public interface FrameCallback {
        /** nv21 为 ANALYSIS_W×ANALYSIS_H 的 NV21 数据，在相机线程回调 */
        void onFrame(byte[] nv21, int w, int h);
    }

    public interface StatusListener {
        void onStatus(boolean connected, int w, int h, String cameraId);
    }

    private final Context mContext;
    private final Surface mPreview;
    private final FrameCallback mFrameCb;
    private StatusListener mStatusCb;

    private final CameraManager mCameraManager;
    private String mCameraId;
    private CameraDevice mDevice;
    private CameraCaptureSession mSession;
    private ImageReader mReader;
    private HandlerThread mThread;
    private Handler mHandler;
    private volatile boolean mReleased = false;
    private boolean mWantOpen = false;
    private boolean mOpening = false;          // open 流程进行中，防重入
    private long mLastAttemptMs = 0;

    private long mLastAnalysisMs = 0;
    private String mPreferredId = null;        // 用户强制指定的相机 id（空=自动选 HDMI）
    private static native boolean buildNv21(Object yBuf, int yRow, int yPix,
                                            Object uBuf, int uRow, int uPix,
                                            Object vBuf, int vRow, int vPix,
                                            byte[] out, int w, int h);
    private final byte[][] mNv21Pool = new byte[4][0]; // 环形缓冲，推理期间不被覆写
    private int mPoolIdx = 0;

    /** 插拔感知：外部相机出现/消失时重新评估 */
    private final AvailabilityCallback mAvailCb = new AvailabilityCallback() {
        @Override public void onCameraAvailable(@NonNull String id) { reevaluate(); }
        @Override public void onCameraUnavailable(@NonNull String id) { reevaluate(); }
    };

    public HdmiCameraHelper(Context ctx, Surface preview, FrameCallback frameCb) {
        mContext = ctx;
        mPreview = preview;
        mFrameCb = frameCb;
        mCameraManager = (CameraManager) ctx.getSystemService(Context.CAMERA_SERVICE);
        mThread = new HandlerThread("HdmiCam");
        mThread.start();
        mHandler = new Handler(mThread.getLooper());
    }

    public void setStatusListener(StatusListener l) { mStatusCb = l; }

    /** 指定相机 id（置空恢复自动选择）——"选择 camera 输入"接口 */
    public void setPreferredCameraId(String id) {
        mPreferredId = (id == null || id.isEmpty()) ? null : id;
        // 若已打开的是别的相机，关闭后由 reevaluate 按新目标重开
        if (mDevice != null && !idEquals(mCameraId, mPreferredId)) closeCamera();
        mHandler.post(this::reevaluate);
    }

    private static boolean idEquals(String a, String b) {
        return a == null ? b == null : a.equals(b);
    }

    public void start() {
        mWantOpen = true;
        mCameraManager.registerAvailabilityCallback(mAvailCb, mHandler);
        mHandler.post(this::reevaluate);
    }

    public void release() {
        mReleased = true;
        mWantOpen = false;
        try { mCameraManager.unregisterAvailabilityCallback(mAvailCb); } catch (Exception ignored) {}
        closeCamera();
        // 保留 mThread/mHandler 引用（quit 后 post 静默失效），避免与相机线程竞态
        if (mThread != null) {
            mThread.quitSafely();
        }
    }

    // ---------------- camera selection / (re)open ----------------

    private void reevaluate() {
        if (mReleased || !mWantOpen) return;
        if (mDevice != null || mOpening) return; // 已在运行或已在打开流程中
        long now = android.os.SystemClock.uptimeMillis();
        if (now - mLastAttemptMs < 1000) return; // 回调风暴节流，watchdog 兜底
        mLastAttemptMs = now;
        String id = pickHdmiCamera();
        if (id == null) {
            notifyStatus(false, 0, 0, null);
            mHandler.postDelayed(this::reevaluate, REOPEN_DELAY_MS);
            return;
        }
        openCamera(id);
    }

    /**
     * HDMI-in 优先级：LENS_FACING_EXTERNAL > 数值最大的 camera id。
     * 板上 hdmirx 由独立 provider 注册（插入后动态出现，id 通常靠后）；
     * 本板 internal provider 无设备，USB 相机会报告为 BACK。
     */
    private String pickHdmiCamera() {
        try {
            String[] ids = mCameraManager.getCameraIdList();
            for (String id : ids) {
                Log.d(TAG, "camera " + id + " facing=" + mCameraManager
                        .getCameraCharacteristics(id).get(CameraCharacteristics.LENS_FACING));
            }
            if (mPreferredId != null) {
                for (String id : ids) {
                    if (id.equals(mPreferredId)) return id;
                }
                Log.w(TAG, "preferred camera " + mPreferredId + " not found, fallback auto");
            }
            String fallback = null;
            for (String id : ids) {
                CameraCharacteristics cc = mCameraManager.getCameraCharacteristics(id);
                Integer facing = cc.get(CameraCharacteristics.LENS_FACING);
                if (facing != null && facing == CameraCharacteristics.LENS_FACING_EXTERNAL) return id;
                if (fallback == null
                        || Integer.parseInt(id) > Integer.parseInt(fallback)) {
                    fallback = id;
                }
            }
            return fallback;
        } catch (Exception e) {
            Log.e(TAG, "pickHdmiCamera", e);
            return null;
        }
    }

    private void openCamera(String id) {
        if (ContextCompat.checkSelfPermission(mContext, Manifest.permission.CAMERA)
                != PackageManager.PERMISSION_GRANTED) return;
        mCameraId = id;
        mOpening = true;
        Log.i(TAG, "opening camera " + id);
        try {
            mCameraManager.openCamera(id, new CameraDevice.StateCallback() {
                @Override public void onOpened(@NonNull CameraDevice device) {
                    mOpening = false;
                    if (mReleased) { device.close(); return; }
                    mDevice = device;
                    setupSession();
                }
                @Override public void onDisconnected(@NonNull CameraDevice device) {
                    Log.w(TAG, "camera disconnected (HDMI unplug or format change)");
                    mOpening = false;
                    device.close();
                    mDevice = null;
                    if (!mReleased) {
                        closeSessionOnly();
                        notifyStatus(false, 0, 0, null);
                        mHandler.postDelayed(HdmiCameraHelper.this::reevaluate, REOPEN_DELAY_MS);
                    }
                }
                @Override public void onError(@NonNull CameraDevice device, int error) {
                    Log.e(TAG, "camera error " + error);
                    mOpening = false;
                    device.close();
                    mDevice = null;
                    if (!mReleased) {
                        notifyStatus(false, 0, 0, null);
                        mHandler.postDelayed(HdmiCameraHelper.this::reevaluate, REOPEN_DELAY_MS);
                    }
                }
            }, mHandler);
        } catch (Exception e) {
            Log.e(TAG, "openCamera fail", e);
            mOpening = false;
            if (!mReleased) mHandler.postDelayed(this::reevaluate, REOPEN_DELAY_MS);
        }
    }

    private Size pickAnalysisSize(CameraCharacteristics cc) {
        StreamConfigurationMap map =
                cc.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP);
        Size best = new Size(ANALYSIS_W, ANALYSIS_H);
        if (map == null) return best;
        Size[] ys = map.getOutputSizes(ImageFormat.YUV_420_888);
        if (ys == null || ys.length == 0) return best;
        long target = (long) ANALYSIS_W * ANALYSIS_H;
        long bestDiff = Long.MAX_VALUE;
        for (Size s : ys) {
            long diff = Math.abs((long) s.getWidth() * s.getHeight() - target);
            if (diff < bestDiff) { bestDiff = diff; best = s; }
        }
        return best;
    }

    private void setupSession() {
        try {
            CameraCharacteristics cc = mCameraManager.getCameraCharacteristics(mCameraId);
            Size analysisSize = pickAnalysisSize(cc);
            Log.i(TAG, "analysis size " + analysisSize);

            final int w = analysisSize.getWidth(), h = analysisSize.getHeight();
            mReader = ImageReader.newInstance(w, h, ImageFormat.YUV_420_888, 2);
            mReader.setOnImageAvailableListener(this::onImage, mHandler);

            List<Surface> outputs = new ArrayList<>();
            if (mPreview != null && mPreview.isValid()) {
                outputs.add(mPreview);
            } else {
                Log.w(TAG, "preview not ready, analysis-only session");
            }
            outputs.add(mReader.getSurface());

            mDevice.createCaptureSession(outputs, new CameraCaptureSession.StateCallback() {
                @Override public void onConfigured(@NonNull CameraCaptureSession session) {
                    if (mReleased || mDevice == null) return;
                    mSession = session;
                    try {
                        CaptureRequest.Builder req = mDevice
                                .createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW);
                        for (Surface s : outputs) req.addTarget(s);
                        session.setRepeatingRequest(req.build(), null, mHandler);
                        Log.i(TAG, "capture session started");
                        notifyStatus(true, w, h, mCameraId);
                    } catch (Exception e) {
                        Log.e(TAG, "setRepeating fail", e);
                    }
                }
                @Override public void onConfigureFailed(@NonNull CameraCaptureSession s) {
                    Log.e(TAG, "session config failed");
                }
            }, mHandler);
        } catch (Exception e) {
            Log.e(TAG, "setupSession fail", e);
            // 设备可能已失效（如 HDMI 分辨率切换瞬间），关掉后延时重开
            if (!mReleased) {
                closeCamera();
                mHandler.postDelayed(this::reevaluate, REOPEN_DELAY_MS);
            }
        }
    }

    // ---------------- frame → NV21 ----------------

    private void onImage(ImageReader reader) {
        Image img = null;
        try {
            img = reader.acquireLatestImage();
            if (img == null) return;

            long now = System.currentTimeMillis();
            if (now - mLastAnalysisMs < ANALYSIS_INTERVAL_MS) return; // img 已在 finally close
            mLastAnalysisMs = now;

            int w = img.getWidth(), h = img.getHeight();
            byte[] dst = mNv21Pool[mPoolIdx];
            if (dst.length != w * h * 3 / 2) dst = new byte[w * h * 3 / 2];
            Image.Plane[] pl = img.getPlanes();
            // C++ 转换（~2ms），替代 Java 逐像素循环（~20ms）
            boolean ok = buildNv21(
                    pl[0].getBuffer(), pl[0].getRowStride(), pl[0].getPixelStride(),
                    pl[1].getBuffer(), pl[1].getRowStride(), pl[1].getPixelStride(),
                    pl[2].getBuffer(), pl[2].getRowStride(), pl[2].getPixelStride(),
                    dst, w, h);
            if (ok) {
                mNv21Pool[mPoolIdx] = dst;
                mPoolIdx = (mPoolIdx + 1) % mNv21Pool.length;
                if (mFrameCb != null) mFrameCb.onFrame(dst, w, h);
            }
        } catch (Exception e) {
            Log.e(TAG, "onImage", e);
        } finally {
            if (img != null) img.close();
        }
    }

    // ---------------- misc ----------------

    private void notifyStatus(boolean connected, int w, int h, String id) {
        StatusListener l = mStatusCb;
        if (l != null) {
            // 监听器会触碰 UI，必须回主线程
            new android.os.Handler(android.os.Looper.getMainLooper())
                    .post(() -> l.onStatus(connected, w, h, id));
        }
    }

    private void closeSessionOnly() {
        if (mSession != null) { try { mSession.close(); } catch (Exception ignored) {} mSession = null; }
        if (mReader != null) { try { mReader.close(); } catch (Exception ignored) {} mReader = null; }
    }

    private void closeCamera() {
        closeSessionOnly();
        if (mDevice != null) { try { mDevice.close(); } catch (Exception ignored) {} mDevice = null; }
    }
}
