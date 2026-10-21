# RKNN 问题处理分析 —— accuracy_analysis 联板全记录

> 一句话总结：**accuracy_analysis 必须从 ONNX 走量化构建（不能 load_rknn），必须在真板上跑，而"真板可达"这四個字背后是 一整条 WSL→Windows→网络→ARP 的排障链。** 本文记录这条链上的每个坑、根因与解法，供以后（任何板、任何 ROM）直接抄。
>
> 关联：[构建与验证手册](docs/2026-10-17-构建与验证手册.md) · [一帧的旅程](docs/2026-10-21-一帧的旅程-推理导学.md) · 工具脚本 `tools/rknn_acc.py`

---

## 0. 目标与最终可用形态

**目标**：RKNN LAB 增加"量化精度"能力——逐层对比 模拟器 vs 真板 的输出（余弦相似度），定位量化敏感层。

**最终可用形态**（全部验证通过）：

```bash
python3 tools/rknn_acc.py <model.onnx> <dataset.txt> <device> [meanstd] [out_dir]
# 例:
python3 tools/rknn_acc.py best.onnx dataset.txt 192.168.0.x:5555 0,0,0/255,255,255 /tmp/rknn_acc_out
# 产出: /tmp/rknn_acc_out/error_analysis.txt（逐层余弦表）
#       /tmp/rknn_acc_out/map_name_to_file.txt（层名→dump 文件映射）
#       /tmp/rknn_acc_out/runtime/dumps/*.npy（逐层张量，本项目 786 个）
```

**真跑结果**（helmet yolov5s，单图校准，静态 ARP 确定性链路）：94 层全部解析，**`sim_entire_cos ≥ 0.99` 的层 = 94/94（零低余弦层）**；`sim_single_cos≈1.0` 全程（模拟器逐层忠实）、`rt_single_cos=1.0`（板端逐层与模拟器一致）、`sim_entire_cos` 从 0.9998 缓降——量化噪声逐层累积的正常形态。单图校准能跑通全流程；正式量产级量化建议仍扩充 50~200 张多光照校准图。

## 1. API 误区：load_rknn 之后不能调 accuracy_analysis

**现象**：第一版脚本 `load_rknn(模型.rknn)` → `init_runtime` → 想直接分析。官方文档明确：**不可行**。

**正确流程**（accuracy_analysis 的本义是"对比量化前后逐层误差"，所以它自己会构建量化模型）：

```
config(mean_values, std_values, target_platform)
→ load_onnx(原始模型.onnx)                    # 不能是 .rknn
→ build(do_quantization=True, dataset=校准图列表.txt)
→ init_runtime(target='rk3576', device_id=设备)
→ accuracy_analysis(inputs=[图片], output_dir, target, device_id)
```

**两个连带细节**：
- `dataset.txt` 是校准图路径列表（每行一张）；量化质量取决于它的多样性（单图校准会有低余弦层）
- `config(mean_values=..., std_values=...)` 的参数**已经是列表**，再包一层会报 `invalid format`

## 2. 环境：toolkit2 只有 Linux 版，WSL 就是官方姿势

**查证**（PyPI JSON 接口权威确认）：rknn-toolkit2 官方仅发布 **manylinux x86_64 / aarch64** wheel（cp36~cp312，对应 Ubuntu 18.04~24.04），**没有任何 Windows wheel**。GitHub release 也不带。

结论：Windows 用户跑 toolkit2 的官方姿势 = **WSL2**（Rockchip 文档同款建议）。Windows 侧 python 3.14 再新也装不上。

## 3. 传输排障链（本次 90% 的时间耗在这里）

toolkit2 在 WSL，板在手上，"连上板"分三个候选路线，逐个的坑如下。

### 3.1 路线一：USB 直连（Windows adb 看得见，WSL 看不见）

Windows adb server 独占 USB；WSL 里的 adb（含 toolkit2 自带的私有 adb，位于
`~/.local/lib/python3.10/site-packages/rknn/3rdparty/platform-tools/adb/linux-x86_64/adb`）
看不到 Windows 侧 USB 设备。

**标准解法 = usbipd-win**（微软官方 USB-over-TCP 方案）：

```powershell
usbipd bind    --busid <BUSID>        # 管理员 PowerShell，一次性
usbipd attach  --wsl --busid <BUSID>  # 每次插板后
```

**坑 1**：attach 后 WSL adb 显示 `no permissions (missing udev rules)`——Rockchip vendor id（2207）不在默认 udev 规则里。一次性修复（WSL 终端，sudo 输自己的密码）：

```bash
echo 'SUBSYSTEM=="usb", ATTR{idVendor}=="2207", MODE="0666", GROUP="plugdev"' | sudo tee /etc/udev/rules.d/51-rk.rules
sudo udevadm control --reload-rules && sudo udevadm trigger
```

**坑 2**：`usbipd attach` 报 `Device busy (exported)`——之前某次 attach 的僵尸状态，`wsl --shutdown` 清掉即可（注意会杀掉所有 WSL 会话）。

**坑 3**：`usbipd bind` 报 `Access denied`——需要管理员 PowerShell。

### 3.2 路线二：网络 adb（adb tcpip / 无线调试）

前提：板端 adbd 监听 TCP。本板（RK3576，Android 14）实测：

| 尝试 | 结果 |
|---|---|
| `adb root` + `adb tcpip 5555`（多次，含全新开机） | 口头应答 "restarting in TCP mode"；早期观察到"5555 未监听"——**后被证伪**：persist 自听生效，`/proc/net/tcp6` 一直有 `:15B3`，"连不上"全是下面的 ARP 抢答 |
| `setprop service.adb.tcp.port 5555` + `persist.adb.tcp.port 5555` + `stop/start adbd` | 属性均生效；persist 变体经重启实测**开机自听成功** |
| 开发者选项"无线调试" | 有此选项且能显示 IP:端口（每次会变），但外部连接同样被 ARP 冲突遮蔽 |

**根因（实测钉死）**：不是 ROM 裁剪了 TCP——**是局域网里另一台设备抢答了板子的 IP（ARP 冲突）**。证据链：

1. 板上 `ss -tln`：5555 双栈监听存在（`[::]:5555`）
2. **板子本机** `nc 192.168.0.x 5555`：能连上（套接字双栈正常）
3. **外部**（Windows/WSL）连同一 IP:端口：积极拒绝（RST）
4. ARP 表：该 IP 的 MAC 是 `64:64:4a:xx`，**而板子 wlan0 MAC 是 `6c:05:d3:xx`**——外部流量全被发往陌生设备

**临时解**（管理员 PowerShell，把 IP 钉回板子真实 MAC；重启失效需重跑）：

```powershell
netsh interface ip add neighbors "<网卡名>" <板IP> <板MAC>
```

**长期解**：路由器后台找出抢答设备（MAC 前缀 64-64-4a）处理掉；或给板子做 DHCP 静态绑定换一个没人占的 IP。

> 教训：**"connection refused"不一定是对端服务没起——先查 ARP 表对不对得上对端真实 MAC。** IP 冲突的隐蔽性在于：本机一切正常、DHCP 不报错、Android 也不弹窗，只有"外部连不上"一个症状。

### 3.3 无线调试端口（38205 这类）

Android 11+ 的"无线调试"走独立传输（随机端口+配对码），不受上面 TCP 缺陷影响——但**同样被 ARP 冲突遮蔽**（外部连不上）。先解决 3.2 的冲突，它才有意义。

### 3.4 网络 adb 上的 accuracy_analysis：挂死与保活

即使网络 adb 连上了（`adb connect IP:5555` → device），accuracy_analysis 仍会在中途**挂死**：

**现象**：`init_runtime` 成功 → 打印 `adbd is already running as root` → 日志永远停在这里，进程僵死。

**根因**：accuracy_analysis 内部会执行 `adb root`——若 adbd 因此重启，**网络 TCP 传输随之中断**（USB 会自动重连，网络不会），toolkit2 已挂起的下一条 adb 子进程永远不会返回。实测：无保活时两次运行都冻在同一行；每 3 秒重连的看门狗也救不回**已经挂死**的子进程。

**解法**：**预防性保活**——accuracy_analysis 运行期间并行跑重连循环：

```bash
( for i in $(seq 1 300); do
    adb connect <板IP>:5555; adb root; sleep 3
  done ) &
python3 -u tools/rknn_acc.py ...   # -u 关闭缓冲，进度实时可见
```

**坑**：反复 `adb connect/root` 的连接循环本身会把 WSL 服务压到偶发超时（`Wsl/Service/0x8007274c`）——重试即可，不是死局。

## 4. rknn_server（板端代理）

- 位置：`/data/local/tmp/rknn_server`（来自 SDK `rknpu2/runtime/Android/rknn_server/arm64/`）
- 启动：`LD_LIBRARY_PATH=/data/local/tmp/rvs nohup /data/local/tmp/rknn_server >…/rknn_server.log 2>&1 &`（LD 指向配套 librknnrt.so）
- **板子断电/重启后进程会消失**，必须重新启动（二进制在数据分区，不会丢）
- 版本必须与 PC 端 toolkit2 配套（本项目双双 2.3.2）；版本错配的症状是 init_runtime/eval 静默失败
- adb root 状态影响它的工作方式：accuracy_analysis 内部会自查并保持 root

## 5. 排障速查表

| 症状 | 根因 | 处置 |
|---|---|---|
| `The extension of [x.rknn] is not '.onnx'` | accuracy_analysis 不能 load_rknn | 换 ONNX 源模型走量化构建 |
| `config ... mean_values is invalid format` | 参数已列表又包一层 | 直接传 parse_meanstd 的返回值 |
| 外部连 5555 积极拒绝，板上监听存在 | **ARP 被另一设备抢答** | 静态 ARP 或清除冲突设备 |
| 5555 无监听（/proc/net/tcp6 无 :15B3） | adbd 未进 TCP 模式 | root + tcpip；若 ROM 不支持→无线调试或 usbipd |
| accuracy_analysis 冻结在 `adbd is already running as root` | 网络传输被 adbd 重启切断 | 并行保活循环（3s connect+root） |
| WSL adb 显示 `no permissions` | 缺 udev 规则（vendor 2207） | 51-rk.rules 一行 + reload |
| `usbipd attach` 报 busy/exported | 僵尸 attach | `wsl --shutdown` 后重来 |
| `pip install rknn-toolkit2` 找不到版本 | PyPI 无此包/无 win wheel | 用官方仓库/本地 Linux wheel |
| 逐层余弦大量为 0 或异常偏低 | 解析器把欧氏距离（euc）当成余弦（cos） | 按 `|` 分列后取偶数位（cos 列），勿取行内最后一个数字 |

## 6. 板子重启后的手工恢复清单（accuracy_analysis 按序执行）

> 前提：板子已连 WiFi、与 PC 同网段；WSL toolkit2 已装（§2）；dataset.txt 已备。
> `persist.adb.tcp.port=5555` 已设置——理论上开机会自动监听，**先做第 1 步验证，不通再做第 2 步**。

```bash
# ① 验证 5555 是否开机自听（在 PC 或 WSL 上）
#    通了 → 直接跳第 3 步；拒绝 → 做第 ② 步（需要 USB 线连着板子）
adb connect <板IP>:5555

# ② USB 在位时：adbd 恢复 root + 重开 TCP
adb -s <板序列号> root
adb -s <板序列号> tcpip 5555

# ③ 起 rknn_server（每次开机都必须，进程不持久）
adb -s <设备> shell "LD_LIBRARY_PATH=/data/local/tmp/rvs nohup /data/local/tmp/rknn_server >/data/local/tmp/rknn_server.log 2>&1 &"

# ④ WSL 侧连板（toolkit2 私有 adb）
~/.local/lib/python3.10/site-packages/rknn/3rdparty/platform-tools/adb/linux-x86_64/adb connect <板IP>:5555

# ⑤ 保活循环（后台挂着；accuracy_analysis 内部 adb root 会断网络传输，无此步必挂死）
( for i in $(seq 1 300); do
    ~/.local/lib/python3.10/site-packages/rknn/3rdparty/platform-tools/adb/linux-x86_64/adb connect <板IP>:5555
    ~/.local/lib/python3.10/site-packages/rknn/3rdparty/platform-tools/adb/linux-x86_64/adb root
    sleep 3
  done ) &

# ⑥ 跑分析（-u 实时看进度）
cd <仓库根>
python3 -u tools/rknn_acc.py <onnx> <dataset.txt> <板IP>:5555

# ⑦ 看结果
cat /tmp/rknn_acc_out/error_analysis.txt
ls /tmp/rknn_acc_out/runtime/dumps/ | wc -l
```

若 ⑥ 卡死在 `adbd is already running as root` 之后 → ⑤ 的保活循环没在跑。

## 7. 方法论沉淀

- **卡点先搜后试**：本次最大的弯路（usbipd/ARP 的一长串尝试）始于没先搜"accuracy_analysis 别人怎么跑"——一搜就发现 load_rknn 限制和官方流程。先查官方文档与社区做法，再动手。
- **每个"不可能"都要留一次实测**：比如"外部连不上但本机能连"这一对矛盾现象，直接指向了 ARP 冲突。
- **诊断脚本要落盘进仓库**：`tools/journey_anchor_probe.py`、`tools/rknn_acc.py` 都是排障过程的沉淀，下次零成本复用。
