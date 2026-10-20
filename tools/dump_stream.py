"""监听 29000，把 scrcpy server 送来的原始字节存盘并分析 NAL 结构"""
import socket
import sys
import time

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind(("127.0.0.1", 29000))
s.listen(1)
s.settimeout(20)
print("listening on 29000 ...", flush=True)
try:
    c, addr = s.accept()
except socket.timeout:
    print("NO CONNECTION within 20s")
    sys.exit(2)
print("connected from", addr, flush=True)

buf = bytearray()
end = time.time() + 60
while time.time() < end:
    try:
        c.settimeout(2)
        chunk = c.recv(262144)
        if not chunk:
            print("peer closed", flush=True)
            break
        buf += chunk
    except socket.timeout:
        break
c.close()

open("raw_stream.bin", "wb").write(buf)
print(f"total {len(buf)} bytes")
print("first 48:", buf[:48].hex(" "))
# 找 Annex-B 起始码
startcodes = buf.find(b"\x00\x00\x00\x01"), buf.find(b"\x00\x00\x01")
print("startcode 4B at:", startcodes[0], " 3B at:", startcodes[1])
if startcodes[0] >= 0:
    nal = buf[startcodes[0] + 4]
    print("first NAL type:", nal & 0x1F)
