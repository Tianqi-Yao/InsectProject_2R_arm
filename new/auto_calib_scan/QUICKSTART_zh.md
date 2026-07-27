# auto_calib_scan 快速上手

这个文件夹是完全独立的一份分支代码（`arm_core.py`/`arm_hardware.py`/`jog_controller.py`/`motion_planning/`/`main.py`都是从`../`复制过来的，不是引用），专门做"用AprilTag视觉自动标定机械臂运动学参数 + 卡片贴tag自动检测位置，然后驱动机械臂逐点扫描"这件事。跟`../fixed_path_scan/`（靠人点动教两个角点、参数是手工估的）是两条独立路线，互不调用，方便你自己对比。

## 0. 目前的调试阶段：电脑直连USB摄像头 + 直连舵机

嫌树莓派不方便调试的话，可以先在电脑上跑：机械臂通过ESP32透传桥USB线直接接电脑，摄像头用一个标准UVC的USB摄像头（比如Arducam 16MP）直接接电脑，不需要碰树莓派。等这一套跑通了、标定和扫描效果都满意了，再把整个`auto_calib_scan/`文件夹搬到树莓派上，把`calib.json`里两个配置改回树莓派对应的值就行，**代码不用改**。

## 1. 安装依赖

```bash
cd auto_calib_scan
pip install -r requirements.txt
```

如果以后要挪到树莓派上用真的CSI摄像头，那台机器上还需要：
```bash
sudo apt install -y python3-picamera2   # 用apt装，不要pip（仅树莓派需要）
```
现在在电脑上调试用USB摄像头，这一步可以先跳过。

确认舵机SDK暴露的接口跟`arm_hardware.py`期望的一致：
```bash
python3 -c "import scservo_sdk as s; print([n for n in dir(s) if not n.startswith('_')])"
```
找一下有没有`PortHandler`和`PacketHandler`。

## 2. 准备AprilTag

`tag36h11`家族，一共需要贴/放这几张：
- **4张**：贴在200x150mm工作纸面的四个角（标定纸本身，跟运动学标定用的是同一张）
- **1张**：装在机械臂末端执行器上
- **1张**：贴在你要扫描的卡片上——**贴在卡片中心，tag自身的边尽量跟卡片的边对齐**（`detect_card_rect()`直接假设tag的朝向就是卡片的朝向，没有再校准这个偏移）

## 3. 配置 calib.json

```bash
cp calib.example.json calib.json
```

编辑`calib.json`：

```json
"hardware": {
  "servo_port": "/dev/cu.usbserial-XXXX",
  "joint_ids": {"joint1": 1, "joint2": 2},
  "camera_backend": "usb",
  "usb_camera_index": 0
}
```

- `servo_port`：改成你这台电脑上ESP32桥接板实际的串口设备路径，可以用`ls /dev/cu.*`看一下插上USB线之后多出来的那个设备名。
- `camera_backend`：调试阶段填`"usb"`（用OpenCV读标准UVC摄像头）；以后挪到树莓派上用CSI摄像头时改回`"picamera2"`。
- `usb_camera_index`：`cv2.VideoCapture`的设备号，先填`0`试试，不对（比如读到的是电脑自带摄像头而不是Arducam）就依次试`1`、`2`……

如果你的机械臂servo2的机身固定在L1末端、但转动轴实际偏离L1连线一侧，现在就把这个偏移量（毫米）填进`kinematics.elbow_offset_mm`（拿卡尺/CAD图纸物理测量两个转轴的中心距，不要指望后面的视觉标定能测出这个值——这个量从数学上就没法只靠末端位置数据反推）。两者共线的话保持`0.0`。

## 4. 确认摄像头能读到画面

```bash
python3 main.py test-camera --watch
```

正常的话，终端会持续打印检测到的每个tag的id和像素坐标。确认能看到4张角tag、末端tag、卡片tag都能被稳定检测到（多试几个角度/距离/光照）。`Ctrl+C`退出。如果一直检测不到：先确认`usb_camera_index`选对了设备（试着改成别的数字），再检查光照和对焦。

## 5.（建议先做）设置机械安全限位

跟`../QUICKSTART_zh.md`第4步完全一样的流程，这里不重复展开，直接跑：

```bash
python3 main.py set-joint-limits
```

跟着终端提示，把每个关节的力矩关掉、用手转遍安全范围、`Ctrl+C`结束、输入`y`确认保存。这一步决定了后面所有点动/扫描命令的安全边界，务必在真正让机械臂自动运动之前做完。

## 6. 摄像头+标定可视化监视工具（同时能描联动死区边界）

```bash
python3 camera_view_gui.py
```

比`main.py test-camera --watch`更直观的版本：左边是USB摄像头实时画面，检测到的每个tag都会画圈标id（不限编号，贴错的、多余的tag也能看到）；右边是可达区域着色图+黄色扫描框+机械臂实时姿态，如果`homography`已经跑过，检测到的每个tag还会在这张图上换算成mm坐标画出来，方便直接用眼睛确认角tag是不是真的落在标定纸四角附近、卡片tag是不是落在黄框里面。

这个工具同时也融合了`../manual_test/trace_boundary_gui.py`的功能——**启动后会立即释放两个关节的力矩**（随时可以徒手描边界），退出时自动同步位置、重新上锁。按键：

- `p`：截图（`camera_view.png`）
- `b`：开始/停止描联动死区边界——徒手转动机械臂走一圈安全区域的完整边界，窗口实时画出这条轨迹
- `c`：清空当前描的轨迹
- `s`/回车：把描出来的边界（跟第5步测出来的独立范围取并集）存进`calib.json`，同时截图`joint_limits_trace.png`
- `r`：回放已保存的边界——重新上电，机械臂按保存的顶点依次走一圈，回到起点，验证边界是不是真的在你以为的地方（**这会驱动真实机械臂**，watch closely）
- `k`/`shift+k`：把机械臂折到一个已知的L1-L2夹角（默认90°，量角器读数），不用相机直接修`servo2_offset_deg`——第7步还没跑视觉标定、或者想快速验证一下时能用
- `q`/ESC：退出

`b`/`r`/`s`这三个键需要第5步的`set-joint-limits`已经跑过（`joint_limits_deg`里要先有`joint1`/`joint2`的独立范围）——没跑过的话按这几个键只会提示"run main.py set-joint-limits first"，不会报错，左边相机画面和右边的可达区域/标定视图不受影响，随时能看。`c`和`k`/`shift+k`不受这个限制。

## 7. 视觉标定（这是这个文件夹存在的核心原因）

```bash
python3 main.py homography
```
检测标定纸四个角的tag，拟合像素↔毫米的映射关系，保存到`calib.json`。

```bash
python3 main.py calibrate
```
自动生成一组目标点，逐个驱动机械臂过去（平滑运动），每个点停稳后读真实编码器角度、同时拍照检测末端tag位置，收集够样本后做非线性最小二乘拟合，算出`L1`/`L2`/机械臂底座位置/舵机零点偏移。会打印每个点的误差和整体RMS误差（<1mm很好，1-3mm可用，>3mm建议检查tag贴得牢不牢、有没有对焦问题），最后问你要不要保存，输入`y`确认。

**这一步就是解决"扫描没有紧贴卡片"这个问题的关键**——运动学参数标定准了，后面卡片检测算出来的坐标才能跟机械臂实际能到达的物理位置对得上。

## 8. 配置卡片信息

编辑`calib.json`的`card`段：

```json
"card": {
  "tag_id": 20,
  "width_mm": 85.6,
  "height_mm": 54.0
}
```

- `tag_id`：贴在卡片上那张tag的编号（跟标定纸四角、末端执行器的tag编号不能重复）。
- `width_mm`/`height_mm`：卡片本身的实际宽高，拿尺子量——这两个数不是靠视觉测出来的，卡片比tag大很多，没法从tag检测反推卡片边界。

## 9. 跑卡片扫描工具

```bash
python3 card_gui.py
```

打开后会自动拍一张照检测卡片tag，成功的话面板上会显示卡片的检测位置和朝向，同时画出蛇形扫描节点预览（能到达的绿色，超出机械臂能力范围的红色）。

按键：
- `d`：重新检测一次卡片（挪动过卡片、或者一开始没检测到的时候用）
- `[`/`]`：列数 -/+，`;`/`'`：行数 -/+（都不能少于2）
- `,`/`.`：每个节点停留时间 -/+ 0.2秒
- `s`：保存`card_scan_config.json` + 截图`card_scan_preview.png`
- `p`：只截图
- `g`/回车：卡片已检测到、且所有节点都可达时，开始扫描——机械臂依次走到每个节点、停留设定时间（`card_core.default_on_arrive`占位钩子被调用的时机，目前只打印一行log，真正的拍照代码以后接上去替换这个函数即可）
- `q`/ESC：退出（扫描中按下则先中止扫描，机械臂停在当前位置）

如果面板上出现"WARNING: kinematics look uncalibrated"，说明第7步的`calibrate`还没跑或者跑了但没保存——先回去做完标定，扫描才会准。

## 10. 以后挪到树莓派

把整个`auto_calib_scan/`文件夹搬过去，只需要改`calib.json`里两处：
```json
"hardware": {
  "servo_port": "/dev/serial0（或树莓派上实际的设备路径）",
  "camera_backend": "picamera2",
  "usb_camera_index": 0
}
```
其余（标定结果、卡片配置、扫描参数）都不用变。
