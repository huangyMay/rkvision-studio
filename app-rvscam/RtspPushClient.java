package com.example.camera2;

import android.util.Base64;
import android.util.Log;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.net.Socket;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;

/**
 * 极简 RTSP 推流客户端：ANNOUNCE/SETUP/PLAY 推到 mediamtx 等 RTSP 服务。
 *
 * 传输：TCP interleaved（媒体包 $ch len 与 RTSP 信令同 socket）。
 * 打包：RFC 6184——小 NAL 单包，大 NAL FU-A 分片；帧末包置 marker。
 * 队列：编码线程经有界队列投递（容量 3），满即丢非关键帧——直播只保最新。
 *
 * 时序：setSpsPps(sps,pps)（来自 MediaCodec csd）后 start()，客户端自行完成
 * 信令握手（sprop-parameter-sets 用 sps/pps Base64），握手成功即开始发包。
 */
public class RtspPushClient {
    private static final String TAG = "RtspPush";
    private static final int MAX_PAYLOAD = 1400;   // TCP interleaved 无 MTU 限制，取保守值
    private static final int CLOCK = 90000;

    public interface Listener { void onState(String s); }

    private final String mHost;
    private final int mPort;
    private final String mPath;
    private final Listener mListener;
    private final byte[] mSps, mPps;

    private Socket mSock;
    private OutputStream mOut;
    private BufferedReader mIn;
    private int mCSeq = 1;
    private String mSession = null;
    private int mSeq = 0;
    private final int mSsrc = 0x52565331;   // 'RVS1'
    private volatile boolean mStarted = false;
    private volatile boolean mReady = false;
    private Thread mThread;

    /** 待发包：Annex-B（00 00 00 01 NAL...）+ 90k 时钟时间戳 */
    private static class Frame {
        final byte[] data; final long ptsUs; final boolean key;
        Frame(byte[] d, long pts, boolean key) { data = d; ptsUs = pts; this.key = key; }
    }
    private final BlockingQueue<Frame> mQueue = new ArrayBlockingQueue<>(3);

    public RtspPushClient(String host, int port, String path,
                          byte[] sps, byte[] pps, Listener l) {
        mHost = host; mPort = port; mPath = path;
        mSps = sps; mPps = pps; mListener = l;
    }

    public void start() {
        mStarted = true;
        mThread = new Thread(this::runLoop, "rtsp-push");
        mThread.start();
    }

    public void stop() {
        mStarted = false;
        try { if (mSock != null) mSock.close(); } catch (Exception ignored) {}
        mReady = false;
    }

    /** 编码线程调用：满队/未就绪时丢帧（直播语义） */
    public void pushFrame(byte[] annexB, long ptsUs, boolean key) {
        if (!mReady) return;
        Frame f = new Frame(annexB, ptsUs, key);
        if (!mQueue.offer(f)) {
            mQueue.poll();               // 挤掉最旧
            mQueue.offer(f);
        }
    }

    // ---------------- 网络线程 ----------------

    private void state(String s) {
        Log.i(TAG, s);
        if (mListener != null) {
            android.os.Handler h = new android.os.Handler(android.os.Looper.getMainLooper());
            h.post(() -> { if (mListener != null) mListener.onState(s); });
        }
    }

    private void runLoop() {
        while (mStarted) {
            try {
                handshake();
                if (!mStarted) break;
                pumpFrames();          // 阻塞服务发送队列直到断连
            } catch (Exception e) {
                state("推流断开: " + e.getMessage());
            }
            mReady = false;
            try { if (mSock != null) mSock.close(); } catch (Exception ignored) {}
            if (mStarted) {
                state("3 秒后重连…");
                try { Thread.sleep(3000); } catch (InterruptedException ignored) {}
            }
        }
    }

    private void handshake() throws IOException {
        state("连接 " + mHost + ":" + mPort);
        mSock = new Socket();
        mSock.connect(new InetSocketAddress(mHost, mPort), 5000);
        mOut = mSock.getOutputStream();
        mIn = new BufferedReader(new InputStreamReader(mSock.getInputStream()));
        mCSeq = 1; mSession = null;

        final String url = "rtsp://" + mHost + ":" + mPort + mPath;
        String resp = request("OPTIONS " + url + " RTSP/1.0", null);
        state("OPTIONS ok");

        final String b64Sps = Base64.encodeToString(mSps, Base64.NO_WRAP);
        final String b64Pps = Base64.encodeToString(mPps, Base64.NO_WRAP);
        String profile = "42C028";
        if (mSps.length >= 4) profile = String.format("%02X%02X%02X",
                mSps[1], mSps[2], mSps[3]);

        final String sdp =
            "v=0\r\n" +
            "o=- 0 0 IN IP4 127.0.0.1\r\n" +
            "s=RVS Camera Streamer\r\n" +
            "t=0 0\r\n" +
            "m=video 0 RTP/AVP 96\r\n" +
            "a=rtpmap:96 H264/90000\r\n" +
            "a=fmtp:96 packetization-mode=1;profile-level-id=" + profile +
                ";sprop-parameter-sets=" + b64Sps + "," + b64Pps + "\r\n" +
            "a=control:trackID=0\r\n";
        resp = request("ANNOUNCE " + url + " RTSP/1.0",
                       "Content-Type: application/sdp\r\nContent-Length: " + sdp.length() + "\r\n\r\n" + sdp);
        state("ANNOUNCE ok");

        resp = request("SETUP " + url + "/trackID=0 RTSP/1.0",
                       "Transport: RTP/AVP/TCP;interleaved=0-1;mode=record\r\n");
        mSession = headerValue(resp, "Session");
        state("SETUP ok, session=" + mSession);

        request("RECORD " + url + " RTSP/1.0", "Range: npt=0.000-\r\n");
        mReady = true;
        state("推流中 → rtsp://" + mHost + ":" + mPort + mPath);
    }

    private void pumpFrames() throws IOException {
        long lastKeepalive = System.currentTimeMillis();
        while (mStarted && mReady) {
            Frame f;
            try { f = mQueue.poll(1, java.util.concurrent.TimeUnit.SECONDS); }
            catch (InterruptedException e) { continue; }
            if (f == null) {
                if (System.currentTimeMillis() - lastKeepalive > 25000) {
                    request("OPTIONS * RTSP/1.0", null);
                    lastKeepalive = System.currentTimeMillis();
                }
                continue;
            }
            sendFrameRtp(f.data, f.ptsUs, f.key);
            lastKeepalive = System.currentTimeMillis();
        }
    }

    // ---------------- 媒体打包（RFC 6184）----------------

    private void sendFrameRtp(byte[] annexB, long ptsUs, boolean key) throws IOException {
        final long ts = ptsUs * 9 / 1000;   // us → 90kHz
        int off = 0;
        boolean first = true;
        int lastSc = -1, lastScLen = 0;
        // 先定位最后一个 NAL（marker 只置帧末包）
        while (off < annexB.length) {
            int scLen = 0;
            if (off + 3 < annexB.length && annexB[off] == 0 && annexB[off + 1] == 0
                    && annexB[off + 2] == 1) scLen = 3;
            else if (off + 4 < annexB.length && annexB[off] == 0 && annexB[off + 1] == 0
                    && annexB[off + 2] == 0 && annexB[off + 3] == 1) scLen = 4;
            else break;
            lastSc = off + scLen; lastScLen = 0;
            int nalEnd = nalEndOf(annexB, off + scLen);
            lastScLen = nalEnd - (off + scLen);
            off = nalEnd;
        }
        off = 0;
        while (off < annexB.length) {
            int scLen = 0;
            if (off + 3 < annexB.length && annexB[off] == 0 && annexB[off + 1] == 0
                    && annexB[off + 2] == 1) scLen = 3;
            else if (off + 4 < annexB.length && annexB[off] == 0 && annexB[off + 1] == 0
                    && annexB[off + 2] == 0 && annexB[off + 3] == 1) scLen = 4;
            else break;
            int nalStart = off + scLen;
            int nalEnd = nalEndOf(annexB, nalStart);
            boolean lastNalOfFrame = (nalStart == lastSc);
            emitNal(annexB, nalStart, nalEnd - nalStart, ts, key, first, lastNalOfFrame);
            first = false;
            off = nalEnd;
        }
    }

    private static int nalEndOf(byte[] b, int start) {
        int i = start;
        while (i + 3 <= b.length) {
            if (b[i] == 0 && b[i + 1] == 0
                    && (b[i + 2] == 1 || (b[i + 2] == 0 && i + 3 < b.length && b[i + 3] == 1)))
                return i;
            i++;
        }
        return b.length;
    }

    private void emitNal(byte[] data, int off, int len, long ts, boolean key,
                         boolean firstOfFrame, boolean lastNalOfFrame) throws IOException {
        final int nalType = data[off] & 0x1F;
        if (len <= MAX_PAYLOAD) {                      // 按长度判断——IDR 切片几十 KB 必须走 FU-A
            sendRtp(data, off, len, ts, lastNalOfFrame);
        } else {                                        // FU-A 分片
            final int payloadMax = MAX_PAYLOAD - 2;
            final int num = (len + payloadMax - 1) / payloadMax;
            int pos = 1;                                // 跳过 NAL 头 1 字节
            for (int i = 0; i < num; i++) {
                int size = Math.min(payloadMax, len - pos);
                size = Math.min(size, data.length - (off + pos));   // 越界钳制
                if (size <= 0) break;
                byte[] pkt = new byte[2 + size];
                pkt[0] = (byte)((data[off] & 0x60) | 28);           // F=0, NRI 继承, type=28
                pkt[1] = (byte)((i == 0 ? 0x80 : 0) | (i == num - 1 ? 0x40 : 0)
                        | (nalType & 0x1F));                            // S/E + 原 type
                System.arraycopy(data, off + pos, pkt, 2, size);    // pos=1 已含 NAL 头
                sendRtp(pkt, 0, pkt.length, ts, lastNalOfFrame && i == num - 1);
                pos += size;
            }
        }
    }

    private void sendRtp(byte[] payload, int off, int len, long ts, boolean marker) throws IOException {
        if (len > MAX_PAYLOAD) len = MAX_PAYLOAD;
        mSeq = (mSeq + 1) & 0xFFFF;
        byte[] pkt = new byte[28 + len];          // 4B interleaved 头 + 12B RTP 头 + 载荷
        pkt[12] = '$'; pkt[13] = 0;                       // interleaved channel 0
        pkt[14] = (byte)((12 + len) >> 8); pkt[15] = (byte)((12 + len) & 0xFF);
        pkt[16] = (byte)0x80;                              // RTP v2
        pkt[17] = (byte)(96 | (marker ? 0x80 : 0));        // PT=96, marker
        pkt[18] = (byte)(mSeq >> 8); pkt[19] = (byte)mSeq;
        pkt[20] = (byte)(ts >> 24); pkt[21] = (byte)(ts >> 16);
        pkt[22] = (byte)(ts >> 8);  pkt[23] = (byte)ts;
        pkt[24] = (byte)(mSsrc >> 24); pkt[25] = (byte)(mSsrc >> 16);
        pkt[26] = (byte)(mSsrc >> 8);  pkt[27] = (byte)mSsrc;
        System.arraycopy(payload, off, pkt, 28, len);
        mOut.write(pkt);
        mOut.flush();
    }

    // ---------------- RTSP 信令 ----------------

    private String request(String requestLine, String extra) throws IOException {
        final int seq = ++mCSeq;
        StringBuilder sb = new StringBuilder();
        sb.append(requestLine).append("\r\n");
        sb.append("CSeq: ").append(seq).append("\r\n");
        sb.append("User-Agent: RVS-Cam/1.0\r\n");
        if (mSession != null && !requestLine.startsWith("OPTIONS *"))
            sb.append("Session: ").append(mSession).append("\r\n");
        if (extra != null) sb.append(extra);
        sb.append("\r\n");
        mOut.write(sb.toString().getBytes());
        mOut.flush();

        // 读响应头（RTSP 响应无 body——除 ANNOUNCE 场景，这里服务端不回 body）
        String statusLine = mIn.readLine();
        if (statusLine == null) throw new IOException("server closed");
        if (!statusLine.contains("200")) {
            String err = statusLine;
            // 吞掉本响应剩余头
            String l;
            while ((l = mIn.readLine()) != null && !l.isEmpty()) err += " | " + l;
            throw new IOException(err);
        }
        String all = statusLine;
        String l;
        while ((l = mIn.readLine()) != null && !l.isEmpty()) {
            all += "\n" + l;
            if (l.startsWith("Session:")) mSession = l.substring(8).split(";")[0].trim();
        }
        return all;
    }

    private static String headerValue(String resp, String header) {
        for (String line : resp.split("\n")) {
            if (line.startsWith(header)) {
                return line.substring(header.length() + 1).split(";")[0].trim();
            }
        }
        return null;
    }
}
