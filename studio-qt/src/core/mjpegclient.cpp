#include "mjpegclient.h"
#include <QTimer>

MjpegClient::MjpegClient(QObject *parent) : QObject(parent) {
    m_sock = new QTcpSocket(this);
    connect(m_sock, &QTcpSocket::connected, this, &MjpegClient::onConnected);
    connect(m_sock, &QTcpSocket::readyRead, this, &MjpegClient::onReadyRead);
    connect(m_sock, &QTcpSocket::errorOccurred, this, &MjpegClient::onError);
}

void MjpegClient::start(const QString &host, int port, const QString &path) {
    m_host = host; m_port = port; m_path = path; m_stop = false;
    m_buf.clear();
    emit stateChanged(QStringLiteral("连接中…"));
    m_sock->connectToHost(host, quint16(port));
}

void MjpegClient::stop() {
    m_stop = true;
    m_sock->abort();
}

void MjpegClient::onConnected() {
    emit stateChanged(QStringLiteral("已连接"));
    QByteArray req = QString("GET %1 HTTP/1.1\r\nHost: %2:%3\r\nConnection: close\r\n\r\n")
                         .arg(m_path).arg(m_host).arg(m_port).toUtf8();
    m_sock->write(req);
}

void MjpegClient::onReadyRead() {
    m_buf += m_sock->readAll();
    extractFrames();
}

void MjpegClient::extractFrames() {
    // 鲁棒切帧：找 SOI(FFD8) 与配对 EOI(FFD9)
    for (;;) {
        int soi = m_buf.indexOf("\xFF\xD8");
        if (soi < 0) { m_buf.clear(); return; }
        int eoi = m_buf.indexOf("\xFF\xD9", soi + 2);
        if (eoi < 0) {
            if (soi > 0) m_buf.remove(0, soi);   // 保留半帧继续收
            return;
        }
        const QByteArray jpg = m_buf.mid(soi, eoi - soi + 2);
        m_buf.remove(0, eoi + 2);
        emit frameJpeg(jpg);
        QImage img;
        if (img.loadFromData(jpg, "JPEG") && !img.isNull()) {
            emit frame(img);
        }
    }
}

void MjpegClient::onError(QAbstractSocket::SocketError err) {
    emit stateChanged(QStringLiteral("连接断开(%1)，2 秒后重连…").arg(m_sock->errorString()));
    if (m_stop) return;
    m_buf.clear();
    QTimer::singleShot(2000, this, [this] {
        if (!m_stop) m_sock->connectToHost(m_host, quint16(m_port));
    });
}
