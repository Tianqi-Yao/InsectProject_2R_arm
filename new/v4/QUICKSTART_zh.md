# v4 快速上手

## v4是什么

v4是最简化的一条支线：**手动示教关节角度点位 + 回放**，没有摄像头标定、没有homography、没有IK/工作空间坐标——每个点位都是你亲手把机械臂拖到那个姿态记下来的，天然可达，回放时原样重放，不需要任何`calib.json`。适合"教几个固定姿态、定期回去巡检拍照"这种场景，比如巡检一批工位、定点拍照留档。

如果你需要的是可以在任意坐标点动、扫描、基于AprilTag自动标定的那一整套，看`../QUICKSTART_zh.md`（主线`new/`），不是这份文档。

## 1. 烧录ESP32驱动板固件

正常工作模式下（树莓派/Mac直接控制舵机）：

- 用Arduino IDE打开`../firmware/SerialBridge/SerialBridge.ino`并烧录。
- 这个固件让板子变成一个透明的USB↔舵机总线字节转发器——没有WiFi，没有自己的逻辑。主机直接跟舵机说SCServo协议，板子只是个透传的桥。

如果你需要检查接线、确认某个舵机有没有响应、或者改一个舵机的总线ID，临时改烧`../firmware/ServoJog/ServoJog.ino`：

- 它会自建一个WiFi热点`ArmServoCtrl`（密码`12345678`）。手机连上这个热点，浏览器打开`192.168.4.1`。
- 长按按钮可以点动各个关节；有OLED状态显示，还有一个"Set ID"输入框可以永久修改舵机的总线ID。
- 平时正常工作不要一直烧着这个固件——它和`SerialBridge`都要独占舵机的串口，同一时间只能用一个。测试完记得改烧回`SerialBridge.ino`。

## 2. 树莓派安装教程

假设树莓派已经装好Raspberry Pi OS、能SSH上去，接下来是v4这条支线专属的安装步骤。

### 2.1 启用摄像头接口

imx477（Pi HQ Camera）是定焦镜头，**没有自动对焦**——装好之后要拧镜头上的对焦环，手动对焦到工作距离（机械臂末端/拍摄对象跟相机之间的实际距离），拧好之后一般会有个锁紧螺丝把对焦环固定住，别让它之后自己转动。

确认摄像头排线接好、系统能识别到：

```bash
libcamera-hello
```

能看到实时预览画面就说明硬件+驱动没问题；如果这一步就失败了，先解决这个，不用往下走——后面Python这层出的任何相机问题，都要先排除掉是不是这一步没通过。

如果没装相机，或者暂时只想用示教/回放功能、不需要定点拍照，这一步和下面2.2都可以跳过——`main.py replay`不加`--photos`参数完全不需要相机。

### 2.2 安装Python依赖

```bash
sudo apt install -y python3-picamera2   # 相机库——用apt装，不要用pip（只有要拍照才需要）
cd new/v4
pip install -r requirements.txt
```

如果你用的是venv，创建时要加`--system-site-packages`，不然venv里看不到apt装的`picamera2`：

```bash
python3 -m venv --system-site-packages .venv
```

### 2.3 校验舵机SDK接口

不同来源的`scservo_sdk`包命名可能不一样，先确认你装的版本暴露出来的接口跟`arm_hardware.py`期望的一致：

```bash
python3 -c "import scservo_sdk as s; print([n for n in dir(s) if not n.startswith('_')])"
```

找一下有没有`PortHandler`和`PacketHandler`。如果你装的版本接口不一样，需要改的就是`arm_hardware.py`这一个文件——其他地方都不依赖具体的接口名字。

## 3. 示教点位（record）

```bash
python3 main.py record [--out PATH.json] [--port PORT]
```

流程：

1. 一启动，两个关节的力矩就会被释放——现在可以用手自由拖动机械臂了。
2. 把机械臂拖到你想记的第一个姿态，按回车，这个姿态就被记下来了一个点位。
3. 继续拖到下一个姿态，再按回车，重复多少次都行。
4. 输入`q`回车结束。结束后力矩会自动重新锁上（先把舵机的目标角度同步成当前实际角度，避免锁上的瞬间猛地弹回旧目标），记录的点位存到`recorded_path.json`（或者你`--out`指定的路径）。

## 4. 回放（replay）+ 定点拍照

```bash
python3 main.py replay [--in PATH.json] [--port PORT] [--dwell SECONDS]
python3 main.py replay --photos photos/ [--photo-delay SECONDS]
```

机械臂依次走到`record`存下来的每个点位，每到一个点位的节奏是：

1. **走到点位、彻底停稳**
2. **暂停`dwell_s`秒**（默认4秒）——如果加了`--photos`，这4秒里：
   - 先等`photo_delay_s`秒（默认2秒），让机械臂到点之后残留的震动沉降下来
   - 然后用imx477拍一张满幅静态照片（4056×3040）
   - 拍完再等剩下的`dwell_s - photo_delay_s`秒（默认也是2秒），凑满整个暂停时长
3. 走向下一个点位

不加`--photos`就是纯示教回放，跟拍照逻辑无关的这4秒空等——Mac开发机上不接相机也能这样测完整的示教/回放流程。

### 拍照怎么存

加了`--photos photos/`之后，每次跑`replay`都会在`photos/`下新建一个按时间戳命名的子目录（比如`photos/20260730_153000/`），这次回放的照片按点位顺序存成`point_001.jpg`、`point_002.jpg`……不同次回放各自一个子目录，不会互相覆盖。

### 调整暂停/拍照时机

`--dwell`和`--photo-delay`都可以调，但要满足`0 < photo-delay < dwell`（拍照必须发生在暂停时间内部），否则会直接报错、不会跑到一半才发现设置有问题。

## 5. 不需要硬件的纯逻辑测试

```bash
cd new/v4 && python3 -m pytest tests/
```
