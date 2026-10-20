"""常驻监听 29000：scrcpy server 每次连入就存一段 raw 流"""
import socket
import time

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind(("127.0.0.1", 29000))
s.listen(2)
print("resident listener on 29000", flush=True)
n = 0
while True:
    c, addr = s.accept()
    n += 1
    print(f"[{time.strftime('%H:%M:%S')}] connected #{n}", flush=True)
    buf = bytearray()
    end = time.time() + 10
    c.settimeout(2)
    while time.time() < end:
        try:
            chunk = c.recv(262144)
            if not chunk:
                print("  peer closed", flush=True)
                break
            buf += chunk
        except socket.timeout:
            continue
        except ConnectionResetError:
            print("  reset", flush=True)
            break
    c.close()
    open(f"raw_{n}.bin", "wb").write(buf)
    print(f"  saved raw_{n}.bin {len(buf)} bytes", flush=True)
