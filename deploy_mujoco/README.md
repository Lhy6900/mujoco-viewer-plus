# MuJoCo 多环境仿真框架

模块化的 MuJoCo 仿真框架，支持多环境、Ghost渲染、外力系统和奖励可视化。

## 🚀 快速开始

```bash
python bmwoyaw_multienv.py
```

## 📁 目录结构

```
deploy_mujoco/
├── bmwoyaw_multienv.py    # 主程序
├── bmconfig.py             # 配置文件（用户主要修改）
├── core/                   # 核心模块（环境、策略、观测、控制）
├── visualization/          # 可视化模块（Ghost、外力、奖励）
├── utils/                  # 工具模块（数学、日志、数据加载）
├── docs/                   # 详细文档
├── tests/                  # 测试脚本
├── legacy/                 # 旧代码（已废弃）
└── module/                 # 旧模块（已废弃）
```

## 📖 详细文档

请查看 [docs/README.md](docs/README.md)

## 🔧 修改配置

编辑 `bmconfig.py` 文件即可修改所有参数：

- 策略类型
- 环境数量
- 奖励权重
- 外力参数
- 可视化选项

## 🎮 快捷键

- `↑` / `↓`: 切换主视角环境
- `M`: 显示/隐藏其他环境
- `Ctrl+G`: 切换 Ghost 显示
- `Ctrl+F`: 触发/停止外力
- `Esc`: 退出

## 📝 版本

- **v2.0**: 模块化重构（2025-01-12）
- **v1.0**: 初始版本
