#pragma once
// MJPEG 拉流客户端：解析 multipart/x-mixed-replace，按 JPEG SOI/EOI 切帧。
// 断线 2 秒自动重连。
#include <QByteArray>
#include <QObject>
#include <QImage>
#include <QTcpSocket>

class MjpegClient : public QObject {
    Q_OBJECT
public:
    explicit MjpegClient(QObject *parent = nullptr);
    void start(const QString &host, int port, const QString &path = QStringLiteral("/stream"));
    void stop();

signals:
    void frame(const QImage &img);
    void frameJpeg(const QByteArray &jpg);       // 原始 JPEG 帧（PC 实时推理用）
    void stateChanged(const QString &status);

private slots:
    void onConnected();
    void onReadyRead();
    void onError(QAbstractSocket::SocketError err);

private:
    void extractFrames();
    QTcpSocket *m_sock = nullptr;
    QString m_host;
    int m_port = 80;
    QString m_path;
    QByteArray m_buf;
    bool m_stop = false;
};
