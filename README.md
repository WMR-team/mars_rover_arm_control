# Mars Rover Arm Control

## 概述

该仓库用于 MuJoCo + Pinocchio 的火星车-机械臂协同控制与运动学实验。当前包含：
- Zhurong 车体（六轮阿克曼转向）
- Zhurong + Franka Emika Panda 机械臂的联合模型
- 末端轨迹跟踪、目标可达性判断与基座移动协同

<div align="center" style="margin: 20px 0;">
  <img src="assets/车臂绘制圆形.gif"
    alt="intro img"
    title="Mujoco Arm Trajectory"
    width="800"
    style="max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);"
    loading="lazy"/>
</div>


## 功能亮点

- MuJoCo 仿真 + Pinocchio 运动学/逆运动学
- 末端轨迹生成（circle/heart/figure8/line_y）
- 目标超出工作空间时，底盘自动移动到可达区域
- 轨迹与末端路径可视化
- 可选 ROS1 `/cmd_vel` 订阅、关节状态发布（PlotJuggler）


## 目录结构

```
mars_rover_arm_control/
  controllers/
    zhurong_mars_rover/                 # 仅底盘模型脚本
    zhurong_mars_rover_franka_emika/    # 车臂协同控制
      arm_end_control.py               # 主入口
      control_api.py                   # 车体/机械臂控制与任务协调
      configs/config.yaml              # 运行参数
  planetary_robot_zoo/                 # 子模块：模型资产
  utils/                               # IK、线程、轨迹、ROS 工具
```


## 环境依赖

- Ubuntu 20.04 / 22.04
- Python 3.10
- MuJoCo (Python 绑定)
- pinocchio (建议 conda 安装)
- numpy, scipy, pyyaml
- 可选：ROS1 Noetic (`rospy`, `geometry_msgs`, `sensor_msgs`)


## Quickstart(推荐)

### 1) 克隆并初始化子模块

```bash
git clone <your_repo_url>
cd mars_rover_arm_control
git submodule update --init --recursive
```

### 2) 安装 Python 依赖

```bash
# conda
conda install -c conda-forge pinocchio

# pip
pip install mujoco numpy scipy pyyaml
```

### 3) 运行 MuJoCo 车臂协同控制

```bash
python controllers/zhurong_mars_rover_franka_emika/arm_end_control.py
```


## 配置说明

运行参数在：
- `controllers/zhurong_mars_rover_franka_emika/configs/config.yaml`

常用选项：
- `trajectory_type`：`circle` / `heart` / `figure8` / `line_y`
- `trajectory_center` / `trajectory_scale` / `trajectory_period`
- `arm_workspace_min` / `arm_workspace_max`
- `control_hz` / `sim_hz` / `view_hz`


## ROS1 相关(可选)

如果需要通过 ROS1 `/cmd_vel` 控制底盘，或发布关节状态：

```bash
source /opt/ros/noetic/setup.bash
```

并在配置中开启：
- `ros_joint_pub_enable: True`


## 开发与贡献

### 代码格式化

- Python：`black`
- C/C++：`clang-format`
- 建议启用 pre-commit(提交时自动格式化/检查)：

```bash
pip install pre-commit
pre-commit install
```


## 引用

```
@InProceedings{,
  author    = {},
  title     = {},
  booktitle = {},
  year      = {}
}
```