# 2R 机械臂

一个2连杆（2R）平面机械臂项目，配合摄像头视觉自动标定和平滑的、由运动规划器驱动的运动控制。两个Feetech STS3215-HS总线舵机在水平面内驱动机械臂，工作范围是一张200x150mm的工作纸面；一个俯拍的树莓派+IMX477相机通过检测AprilTag来判断实际的物理位置，因此连杆的真实长度、底座位置、舵机零点偏移/转向都不需要手工测量——全部由视觉数据自动拟合得出。安装和使用步骤见`QUICKSTART_zh.md`。

这是这套代码库的第二次、重构后的版本（见下方"历史沿革"）——在经历了几轮由硬件问题驱动的补丁式修改后（波特率修复、一个舵机方向的bug、三套各自独立实现、后来又互相脱节的点动逻辑），整理归并成了更少、更清晰的文件。如果你现在看的是`../software/`或`../manual_test/`，那些是已归档的上一版本；这个（`new/`）目录才是当前版本。

## 硬件

- 2× Feetech STS3215-HS 串行总线舵机（磁编码器，真实位置反馈——不是开环PWM）
- Waveshare ESP32 舵机驱动板
- 树莓派 + IMX477（Raspberry Pi HQ Camera）
- AprilTag（`tag36h11`家族）：4个固定在200x150mm工作纸面的四角，1个装在末端执行器上

## 目录结构

```
arm_core.py            核心逻辑：IK/FK、单应性(homography)、最小二乘拟合、
                        开机自检、calib.json的读写与schema——这是唯一一个
                        需要你从头到尾通读理解的文件。
arm_hardware.py         黑盒硬件层：舵机总线寄存器I/O、相机采集、AprilTag检测。
                        这里没有任何决策逻辑——不需要关心它的内部实现。
motion_planning/        可插拔的轨迹规划算法（见下方"运动规划"一节）。
                        trapezoidal.py 是默认实现；想换一种算法，只需新加
                        一个文件+一行import即可替换。
jog_controller.py       ArmController：下面三个前端工具共用的唯一实时运动
                        控制器（以前是三套各自独立、逐渐脱节的实现）。
main.py                 命令行入口：test-servo / test-camera / homography /
                        calibrate / selfcheck / set-joint-limits / jog。
manual_test/
  run.py                curses（终端）点动+扫描测试工具，跳过相机/标定流程——
                        用来快速肉眼确认机械臂动作是否符合预期。
  gui.py                同上，但用pygame做了可视化：把真实（编码器读数）姿态
                        和当前指令目标姿态并排画出来对比。
firmware/
  SerialBridge/          正式运行用的ESP32固件：USB↔舵机总线的透明字节转发，
                        让主机能直接跟舵机说SCServo协议。
  ServoJog/              布线/改ID用的调试固件：自建WiFi热点+网页点动界面，
                        跟SerialBridge互斥（两者都要独占舵机串口）。
tests/                   纯逻辑pytest测试集——全都不需要连接硬件
                        （覆盖arm_core、motion_planning、jog_controller）。
calib.example.json       示例calib.json，展示完整的schema。真正的calib.json
                        是运行时生成的，且被gitignore（这是每台设备各自的
                        运行状态，不是源码）。
```

## 运动规划

以前的"运动平滑"是三层各自独立、手工调出来的拼凑：舵机寄存器自带的速度/加速度限制、一对点动专用的速度/加速度常量、还有一个扫描专用的"每隔N毫秒硬塞一个新目标点、不等到位"的土办法——每个前端各自维护一份，这正是`run.py`和`gui.py`的扫描网格大小/速度参数会悄悄脱节的原因（见git历史）。

现在只有一个统一的规划器接口（`motion_planning.TrajectoryPlanner`），由`jog_controller.ArmController`以固定的控制周期频率（默认50Hz）驱动它，点动和多点扫描都走这一套。默认实现（`motion_planning/trapezoidal.py`）是标准的、关节空间下双关节同步的梯形速度曲线——绝大多数机器人/CNC控制器采用的基础方法。相邻的扫描路径点如果方向基本一致，会以巡航速度平滑过渡而不完全停顿（对应`arm_core.py`里的`MotionConfig.blend_threshold`），这正是解决扫描抖动问题的关键。

想换一种算法（比如jerk-limited的S形曲线）：在`motion_planning/`下新建一个文件实现`TrajectoryPlanner`接口，给类加上`@register("your_algo")`装饰器，在`motion_planning/__init__.py`末尾加一行import，再把`calib.json`里的`motion.planner_name`改成`"your_algo"`即可，不需要动其他任何文件。

## 设计要点

- **为什么用视觉做标定**：连杆真实长度和电机底座位置在装配时没法精确手工测量。与其猜测，不如让机械臂自动遍历一组舵机角度，同时相机观察末端的AprilTag；`scipy.optimize.least_squares`联合拟合出真实的L1、L2、底座位置、舵机偏移（`servo1_dir`/`servo2_dir`——舵机原始角度增大的方向是否跟我们的数学约定一致——这个是手工确认的固定硬件事实，不参与拟合：方向反转是一种镜像变换，offset/连杆长度这些参数怎么调都补不出这种反转）。
- **为什么`elbow_offset_mm`是一个固定常数而不是拟合参数，尽管它看起来就是个普通长度**：有些机械臂上，servo2的机身固定在L1末端，但它的转动轴实际上偏离L1连线一侧（不是设计选择，是舵机本身有物理体积导致的必然结果）。这个垂直方向的偏移会改变运动学公式（`ArmParams.elbow_offset_mm`、`fk_from_servo_angles`、`ik_solve`）。很容易会想：视觉标定既然已经能拟合L1/L2/底座/偏移，为什么不把这个也一起拟合？因为它跟L1之间存在可以证明的精确退化关系：视觉数据只能确定`reach = hypot(L1, elbow_offset_mm)`这一个组合值，永远没法确定L1和elbow_offset_mm各自具体是多少——因为在固定`reach`的前提下调整L1和elbow_offset_mm的分配，效果会被`servo1_offset_deg`和`servo2_offset_deg`的相应反向调整完全抵消掉。这一点已经用数值实验验证过：拿真实值elbow_offset_mm=28生成的模拟数据去拟合，结果拟合出了完全不同的elbow_offset_mm，配上相应偏移过的L1，残差误差却几乎完美——不管采多少、多宽范围的数据都一样。所以这个值只能靠独立的物理测量（卡尺或CAD图纸，测两个转轴的中心距）来确定——完整推导见`ArmParams`的文档字符串。
- **为什么要做开机自检**：设备在户外运行、每天都会重启，所以每次开机都会重新用相机核对一遍标定是否还有效。轻微的漂移会自愈（采用新读数、记日志、继续工作）；漂移超过阈值就会停机并触发报警（目前只是占位钩子）,直到有人重新标定为止。
- **为什么calib.json也是硬件/运动配置的唯一数据源**（不只是运动学参数）：之前三个工具各自硬编码同一套物理常量，这正是它们互相脱节的根源。`arm_core.HardwareConfig`/`MotionConfig`和`calib_hardware_config()`/`calib_motion_config()`保证每个前端读的是同一份文件。
- **为什么机械死区保护是两层而不是一层**：IK只检查几何上是否可达（连杆长度），它完全不知道存在物理障碍/死区这回事。`main.py set-joint-limits`让你用手测出每个关节的安全范围，然后**同时**写入舵机自己的硬件Min/Max Angle Limit寄存器（`arm_hardware.py`——舵机固件本身会拒绝转过这个边界，不管上层软件发了什么指令，哪怕这个项目本身有bug）和calib.json的`joint_limits_deg`（`ik_solve`/`jog_controller`会检查的软件层限位，能更早给出更清楚的报错）。真正兜底的是硬件那一层——如果软件层出了bug，靠的就是它；软件层的价值在于能在那之前给出更明确的提示。
- **为什么联动/相对死区是手画的闭合多边形，而不是采样出来的区间曲线**：在一个2连杆机械臂上，远端连杆的安全活动范围有可能**连续地**随近端连杆当前位置变化（比如joint2跟一个固定障碍物之间的间隙，会随joint1的转动平滑变化，而不是"在/不在"某个区域这种非黑即白的关系）。之前试过两种采集方式，都在真实硬件上被证明不行：沿边缘描一条线、猜哪一侧受限——退化成一堆几乎单点的错误区间；扫满内部、按joint1分箱取每箱min/max——实测出来的包络线也不对。`joint_limits_deg`里的`coupled_boundary`换了个更简单的思路：一串(joint1,joint2)顶点，用手围着安全区域的整个边界画一圈闭合轮廓（两个关节扭矩同时松开，沿边界走一整圈，回到起点附近——`manual_test/trace_boundary_gui.py`里的`b`键，或者`main.py set-joint-limits`的终端等效流程）。不做任何分箱/平滑/推导——画出来的轨迹本身就是边界，原样存盘。一个姿态只要落在这个多边形**内部**就算通过（`arm_core.py`里的`_point_in_polygon()`），用的是**缠绕数（winding number）**规则而不是更简单的奇偶（ray-casting）规则——原因是手画的轨迹会自然地回头重走已经走过的路（犹豫、抖动、来回），奇偶规则在"整圈被走了偶数遍"这种巧合下会把判定翻转成"在外面"，纯属奇偶性的巧合；缠绕数不管绕了几圈，只要不是零就判定"在里面"，不会有这个问题。这套表示法还能处理任意形状（包括凹形）的安全区域，这是"每个joint1对应一个joint2区间"这种旧模型从数据结构上就做不到的。这个实时GUI窗口也能在真正跑摄像头视觉拟合之前，快速发现`servo2_offset_deg`不对的问题：把机械臂折到一个已知的L1-L2夹角，按`k`键（对应`arm_core.servo2_offset_from_known_elbow_angle`），就能直接从这一个姿态反解出这个值。联动边界依然只能靠软件保护：舵机自己的硬件角度限位寄存器是严格意义上每个舵机各自独立的——一个STS3215的寄存器没有任何办法表达"我的限位取决于另一个舵机的位置"。如果这种联动碰撞风险比较严重，真正靠得住的硬件级兜底只有物理机械挡块；软件这层检查提供的是更早、更清楚的拒绝提示，不是一个不依赖代码本身没有bug的绝对保证。
- **为什么点动/扫描区域要跟标定表本身的尺寸解耦**：`manual_test/gui.py`/`run.py`以前点动/扫描用的就是`workspace.width_mm`/`height_mm`——AprilTag标定表自己的尺寸。但这个表格的四个角是标签实际贴在哪就是哪，不保证这个矩形完全落在机械臂真正可达+安全的区域（`joint_limits_deg`）里——而且没法靠软件改`workspace.width_mm`/`height_mm`/`corner_world_mm`来"修"，因为这几个数字必须跟标签的物理位置对得上，改了数字不动标签，homography坐标系就错位了。`MotionConfig`的`scan_center_x_mm`/`scan_center_y_mm`/`scan_width_mm`/`scan_height_mm`/`scan_rotation_deg`（`calib_scan_area()`）描述的是同一个坐标系里一个独立的、可选的子矩形——用"中心点+尺寸+旋转角"而不是min/max边界来表示，专门是因为它可以倾斜：一个斜着放的矩形，比轴对齐的矩形更能贴合不规则形状的可达区域。没配置过就退回整张表、不旋转；`generate_scan_path()`加了对应的`center_x_mm`/`center_y_mm`/`rotation_deg`参数。`manual_test/scan_area_gui.py`提供可视化拟合：把`ik_solve()`判定可达的每个工作空间点都涂色（跟其他工具已经在用的三层检查完全一样——IK可达性、独立关节范围、联动死区多边形），让你在这张图上直接拖矩形的角(调整大小)、拖顶边上方的专用手柄(绕中心旋转)、或者拖矩形内部(整体挪动)，而不是手改数字。

## 历史沿革

- `../software/`、`../manual_test/`、`../sim/`、`../ServoDriverST/`都是已归档的旧版本——保留仅供参考，不再运行。`sim/`和`ServoDriverST/`更早，完全是STS3215+视觉这套硬件出现之前的东西（MG90S开环舵机，没有相机）；`software/`/`manual_test/`则是这套设计重构前的上一版。
