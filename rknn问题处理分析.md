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
# ⓪ 【仅 Windows 重启过后首次使用时】静态 ARP 钉回板子 MAC（管理员终端）
#    没有它：外部发往板子 IP 的包会被 ARP 抢答设备截走（见 §3.2）
netsh interface ip add neighbors "以太网 2" <板IP> <板MAC>

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

### 6.1 另一种场景：重启的是 Windows 电脑（板子没动）

板端一切照旧（rknn_server 进程、5555 自听都不受 PC 重启影响），只有 PC 侧两件事要重做：

```powershell
# ① 管理员终端：静态 ARP（Windows 重启清零，影子设备会重新抢答 .96）
netsh interface ip add neighbors "以太网 2" 192.168.0.96 6C-05-D3-06-57-57
```

- ② mediamtx.exe 双击启动（要用 RTSP 预览时）——它同样不自启
- WSL/toolkit2/数据（`assets/acc/` 在 WSL 磁盘 VHD 上）全部持久，无需任何操作

### 6.2 2026-10-25 环境变更：板子换到 192.168.3.x 网段，WSL adb 改原生直连

板子 WiFi 换网（.96→192.168.3.163）后 PC 局域网 IP 也漂了（192.168.0.86→192.168.3.119），旧方案三处全废：

1. **`~/bin/adb` 包装脚本**（WSL toolkit2 用的 adb）：原来 `ADB_SERVER_SOCKET=tcp:192.168.0.86:5037` 共享 Windows adb server——PC IP 漂移后该地址不可达，且 Windows server 默认只绑 127.0.0.1。**已改写为 WSL 原生 adb 直连**：`exec ~/platform-tools/adb "$@"`。注意旧脚本 `exec adb.exe ""` 把调用方参数全丢了（永远只发一个空参数），排障时"adb get-state 永远 offline"是它制造的假象。Windows 侧覆写该文件后须 `chmod +x`（Windows 写文件会丢执行位）。
2. **静态 ARP**（§6.1 ①）：.96 影子设备问题随换网自动消失，3.x 网段暂无 ARP 抢答，无需再钉。
3. **板子换网后首次使用**：`adb connect 192.168.3.163:5555`（WSL 原生 adb 自建 server）+ 检查 rknn_server（③ 命令不变）。

一键预检脚本模式（板连接 + rknn_server 存活检查 + 死了自动重启）已内置于 `tools/acc_compare.py` 启动逻辑与 GUI；手工版见本节 ①③ 两步。

### 6.3 2026-10-25 七通道实测结论 + ⑦ 板上 int16 根因勘透（最小复现实验链）

yolov8n 七通道全量对比实测（板 rk3576 / 驱动 0.9.8 / toolkit2+server+rt 全 2.3.2）：

- ①~⑥ 全过：sim fp16 0.9995 / sim int8 0.9933 / sim int16 1.0 / 板 fp16 0.999 / 板 int8 0.9851（mean IoU vs ONNX）
- ⑦ 板上 int16（w16a16i）：模型能 init、推理能跑，但 `rknn_outputs_get` 恒 `ACK_FAIL → RKNN_ERR_FAIL`。

**根因（最小复现实验链逐步锁定 + libiotrace 机制层实锤，非猜测）**：

| 实验 | 结果 | 排除/锁定 |
|---|---|---|
| mini(1 Conv+Relu)@64 int16 板推 | ✅ OK | int16 板推链路本身通 |
| mini+ReduceSum | ✅ OK | ReduceSum 无辜 |
| mini@640 / mini 20 层深 | ✅✅ OK | 输入尺寸、模型深度排除 |
| yolov8n 只留 1 输出（conv/rsum/sigmoid 各试） | ❌ 全炸 | 输出数量/类型排除 |
| yolov8n tap5%（前 11 节点） | ❌ 炸 | 坏点在最前面几层 |
| **mini(Conv+Sigmoid)@640** | ❌ **同款 ACK_FAIL** | **Sigmoid 坐实** |
| **mini(Conv+Concat)@640** | ❌ **同款 ACK_FAIL** | **Concat 坐实** |
| mini Mul/Add/MaxPool | ✅ OK | 这些无辜 |
| mini+Sigmoid + op_target Sigmoid→cpu | ❌ 仍炸 | **op_target 对 i16 不生效，无软件修法** |
| w8a16 | config 直接拒绝 | RK3576 不支持 |
| w4a16 全模型板推 | ✅ 跑通但仅 2 框 conf 0.35 | 4-bit 权重毁 CNN 精度，不如 int8，非出路 |
| **libiotrace 挂 rknn_server 跑 int16 模型** | **首次 SUBMIT ret=-1（6s 超时）** | **机制层实锤，见下** |

**机制层实锤（2026-10-25 libiotrace，修正早期"输出回传断"的表象判断）**：
int16 模型 load/init 全正常，输入同步完成后**第一条 SUBMIT（108 任务段）即 ret=-1**——
时间戳显示从提交到失败正好 6000ms = 驱动阻塞等待超时。即 **NPU 接到 int16 指令流后
挂死（不产生完成中断），驱动超时报错**；随后 librknnrt 走 POWER_OFF+MEM_DESTROY
清理现场，把错误上报为 "rknn_outputs_get ACK_FAIL"——**API 层的 ACK_FAIL 是清理
路径的表象，不是取输出环节的问题**。为何全 CPU 的 mini 模型能过：无 NPU 段提交，
自然不触发。Conv 在 build 期有 "Not Support u8->i16" 拦截（120 条全回退 CPU），
但 **Sigmoid/Concat 的 i16 指令编译无拦截、执行挂死**——这套栈（驱动 0.9.8+rt 2.3.2）
的 i16 NPU 指令路径是坏的。

**结论**：w16a16i 时 Conv 全部自动回退 CPU（build 日志 "Conv Not Support u8->i16"×120），但 **Sigmoid/Concat 仍被排上 NPU 以 i16 执行——RK3576 这代栈（驱动 0.9.8+rt 2.3.2）i16 的 NPU 算子输出回传是坏的**，任何含 Sigmoid/Concat 的 i16 模型（= 所有检测模型）板上必炸。注意与"CPU 回退路径不通"的旧假设相反：全 CPU 的模型（mini 系）输出全能拿回；坏的恰恰是留在 NPU 上的那部分。**高精度需求用板上 FP16（0.999），生产用 int8（0.9851）；板上 w16a16i 等瑞芯微修复**（可携最小复现提 issue：Conv+Sigmoid@640 + w16a16i + rk3576 即复现）。

**连带效应**：⑦ 失败后 rknn_server 进程仍活着但连接线程楔死（`futex_wait`），同脚本后续板上 accuracy_analysis 全部 Traceback。遇到"板上操作连环失败但 server 进程在"→ 先 `pkill -f rknn_server` 重启再判。逐层（accuracy_analysis）模拟列不受影响：int8 177 层最低 sim_single 0.99398、int16 177 层 0.99998。GUI 侧该失败被逐级捕获展示（⑦ 格子空+红字），不阻塞其余六通道。

### 6.4 2026-10-25 RK3588 Debian 板三档实测（Lite2 原生路径）

板：RK3588 Debian 12（WiFi 192.168.3.109，驱动 0.9.8 + librknnrt 2.3.2）。yolov8n + bus.jpg，
板端 rknn-toolkit-lite2 原生推理（绕开 rknn_server 通道），ONNX 基线 4 框：

| dtype | 框 | conf | 板端耗时 | 判定 |
|---|---|---|---|---|
| int8 | 4/4 全对 | 0.73~0.76 | **54.0 ms/帧** | ✅ 完美 |
| fp16 | 4/4 全对 | 0.85~0.88 | **72.9 ms/帧** | ✅ 完美 |
| int16(w16a16i) | 0 框 | — | 6400 ms/帧 | ❌ 全废 |

**int16 在 3588 上同样不可用，且与 3576 同病不同症**：板上日志
`failed to submit!, op id: 2, op name: Conv:/model.1/conv/Conv, task number: 64, run task counter: 40`
——NPU 任务提交失败（与 3576 的 SUBMIT 超时同族）；3588 原生路径不抛异常而是**返回常量垃圾输出**
（min==max），6400ms≈每帧走失败慢路径。注意 RK3588 编译器把 Conv 以 i16 编上 NPU（3576 是
Conv 回退 CPU、Sigmoid/Concat 上 NPU 挂）——**两代 RKNPU 的 i16 NPU 路径在 0.9.8 驱动栈上都是坏的**，
结论升级：非单板问题，是这套芯片+驱动代的 w16a16i 断层。高精度选 fp16（3588 上 72.9ms 也可用）。

**工具链记录**：① rknn_server 远程通道在 Debian 板 ERROR_PIPE——板上 server accept 后
"Create client failed"(ret=-1/-11)，链路本身通（手动 forward 保持连接）；SDK/系统版 server 均如此，
root cause 未定（unknowns），Lite2 原生路径是 Debian 板的正解；② 板上 pip3 需 apt 装
python3-pip + `pip3 install --break-system-packages rknn-toolkit-lite2 opencv-python-headless`；
③ onnx_infer.py 板侧 import 用 `sys.modules['onnxruntime']=假模块` 绕 ort 依赖；④
Debian 板起后台进程 `setsid xxx </dev/null &`（nohup& 会被会话回收）。

## 7. 方法论沉淀

- **卡点先搜后试**：本次最大的弯路（usbipd/ARP 的一长串尝试）始于没先搜"accuracy_analysis 别人怎么跑"——一搜就发现 load_rknn 限制和官方流程。先查官方文档与社区做法，再动手。
- **每个"不可能"都要留一次实测**：比如"外部连不上但本机能连"这一对矛盾现象，直接指向了 ARP 冲突。
- **诊断脚本要落盘进仓库**：`tools/journey_anchor_probe.py`、`tools/rknn_acc.py` 都是排障过程的沉淀，下次零成本复用。
