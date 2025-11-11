"""
Reward Plotter for MuJoCo Viewer
Displays real-time reward curves using MuJoCo's native MjvFigure API
参考 deploy_robot-main-v1.0/deploy_robot/sim/viewer_plus/reward_plotter.py
"""

import mujoco
import numpy as np
from collections import deque
from typing import Dict, List, Tuple


class RewardPlotter:
    """实时奖励绘图器，使用 MuJoCo MjvFigure 在 viewer 侧边显示奖励曲线"""

    def __init__(self, history_length: int = 300):
        """
        初始化奖励绘图器
        
        Args:
            history_length: 保存的历史数据点数量（默认300步，约6秒@50Hz）
        """
        self.history_length = history_length
        self._histories: Dict[str, deque] = {}  # 存储每个奖励项的历史数据
        self._figures: Dict[str, mujoco.MjvFigure] = {}  # 存储每个奖励项的图表对象
        
        print(f"[RewardPlotter] 初始化完成，历史长度: {history_length}")

    def register_terms(self, term_names: List[str]) -> None:
        """
        注册要跟踪和绘制的奖励项
        
        Args:
            term_names: 奖励项名称列表
        """
        for name in term_names:
            if name in self._histories:
                continue  # 已经注册过，跳过
            
            # 为每个奖励项创建历史数据队列
            self._histories[name] = deque(maxlen=self.history_length)
            
            # 创建 MjvFigure 对象
            fig = mujoco.MjvFigure()
            mujoco.mjv_defaultFigure(fig)
            
            # 设置图表属性
            fig.title = name[:40]  # 标题（截断长名称）
            # 注意：MjvFigure 只有 title，没有 xlabel 和 ylabel
            
            # 设置初始范围：fig.range 是 2x2 数组 [[xmin, xmax], [ymin, ymax]]
            fig.range[0][0] = -self.history_length  # x 轴最小值
            fig.range[0][1] = 0  # x 轴最大值（当前步）
            fig.range[1][0] = -0.01  # y 轴最小值（初始值）
            fig.range[1][1] = 0.01  # y 轴最大值（初始值）
            
            # 配置线条样式
            fig.linepnt[0] = 0  # 初始数据点数量
            
            # 设置颜色 - 背景和面板
            fig.figurergba[:] = [1.0, 1.0, 1.0, 0.3]  # 白色半透明背景（30%）
            fig.panergba[:] = [1.0, 1.0, 1.0, 0.5]    # 白色半透明面板（50%）
            
            # 设置线条颜色 - 根据奖励项名称使用不同颜色
            color_map = {
                "smoothness": [1.0, 0.0, 0.0],              # 红色 - 平滑度
                "pos_tracking_global": [0.0, 1.0, 0.0],     # 绿色 - 全局位置跟踪
                "pos_tracking_local": [0.0, 0.0, 1.0],      # 蓝色 - 局部位置跟踪
                "quat_tracking_global": [1.0, 1.0, 0.0],    # 黄色 - 全局姿态跟踪
                "quat_tracking_local": [1.0, 0.0, 1.0],     # 品红 - 局部姿态跟踪
            }
            line_color = color_map.get(name, [0.5, 0.5, 1.0])  # 默认：浅蓝色
            fig.linergb[0][:] = line_color  # 设置第一条线的颜色（RGB）
            
            # 设置网格和文本颜色
            fig.gridrgb[:] = [0.7, 0.7, 0.7]  # 灰色网格
            fig.textrgb[:] = [0.0, 0.0, 0.0]  # 黑色文字
            
            self._figures[name] = fig
            print(f"[RewardPlotter] 注册奖励项: {name}, 颜色: {line_color}")

    def update(self, rewards: Dict[str, float]) -> None:
        """
        更新奖励历史数据和图表
        
        Args:
            rewards: 奖励字典 {term_name: value}
        """
        for k, v in rewards.items():
            # 如果是新的奖励项，自动注册
            if k not in self._histories:
                self.register_terms([k])
            
            # 添加新数据点
            self._histories[k].append(float(v))
            
            # 更新对应的图表数据
            hist = list(self._histories[k])
            fig = self._figures[k]
            n = len(hist)
            
            if n == 0:
                continue
            
            # 将数据写入 linedata 数组
            # 格式：x0, y0, x1, y1, x2, y2, ...
            for i, val in enumerate(hist):
                fig.linedata[0][2 * i] = -n + i  # x 坐标（相对当前时刻）
                fig.linedata[0][2 * i + 1] = val  # y 坐标（奖励值）
            
            # 设置数据点数量
            fig.linepnt[0] = n
            
            # 自动缩放 y 轴（添加 10% 的边距）
            if n > 0:
                vals = np.array(hist)
                ymin = float(np.min(vals))
                ymax = float(np.max(vals))
                yspan = max(ymax - ymin, 1e-6)  # 避免除零
                padding = yspan * 0.1
                fig.range[1][0] = ymin - padding  # y 轴最小值
                fig.range[1][1] = ymax + padding  # y 轴最大值

    def get_figures_for_viewer(self, viewport: mujoco.MjrRect) -> List[Tuple]:
        """
        计算图表布局并返回 (viewport, figure) 对列表
        
        布局策略：在屏幕右侧 1/3 处垂直堆叠图表
        
        Args:
            viewport: viewer 的视口尺寸
            
        Returns:
            [(viewport, figure), ...] 列表，用于 viewer 渲染
        """
        if not self._figures:
            return []
        
        term_names = list(self._figures.keys())
        num_figs = len(term_names)
        
        # 在屏幕右侧保留 1/3 空间用于显示图表
        plot_width = viewport.width // 3
        plot_left = viewport.left + (viewport.width - plot_width)
        
        # 垂直方向平均分配空间（留出小间隙）
        gap = 5  # 图表之间的间隙（像素）
        total_gap = gap * (num_figs + 1)
        available_height = viewport.height - total_gap
        fig_height = available_height // num_figs
        fig_height = max(fig_height, 50)  # 最小高度 50 像素
        
        result = []
        # 从顶部向下排列图表
        for i, name in enumerate(term_names[:12]):  # 最多显示 12 个图表
            vp = mujoco.MjrRect(
                left=plot_left,
                bottom=viewport.bottom + gap + i * (fig_height + gap),
                width=plot_width,
                height=fig_height
            )
            result.append((vp, self._figures[name]))
        
        return result

    def clear(self) -> None:
        """清空所有历史数据"""
        for hist in self._histories.values():
            hist.clear()
        print("[RewardPlotter] 已清空所有历史数据")

    def get_current_values(self) -> Dict[str, float]:
        """获取当前最新的奖励值"""
        values = {}
        for name, hist in self._histories.items():
            if hist:
                values[name] = hist[-1]
        return values


# 测试代码
if __name__ == "__main__":
    print("=" * 60)
    print("RewardPlotter 测试")
    print("=" * 60)
    
    # 创建绘图器
    plotter = RewardPlotter(history_length=100)
    
    # 注册奖励项
    reward_terms = [
        "smoothness",
        "pos_tracking_global",
        "pos_tracking_local",
        "quat_tracking_global",
        "quat_tracking_local"
    ]
    plotter.register_terms(reward_terms)
    
    # 模拟更新数据
    print("\n模拟添加数据...")
    for step in range(20):
        rewards = {
            "smoothness": -0.01 * np.random.rand(),
            "pos_tracking_global": -0.05 * np.random.rand(),
            "pos_tracking_local": -0.03 * np.random.rand(),
            "quat_tracking_global": -0.02 * np.random.rand(),
            "quat_tracking_local": -0.01 * np.random.rand(),
        }
        plotter.update(rewards)
        
        if step % 5 == 0:
            current = plotter.get_current_values()
            print(f"Step {step}: {current}")
    
    # 检查数据
    print("\n检查历史数据长度:")
    for name, hist in plotter._histories.items():
        print(f"  {name}: {len(hist)} 个数据点")
    
    # 检查图表范围
    print("\n检查图表 Y 轴范围:")
    for name, fig in plotter._figures.items():
        ymin = fig.range[1][0]
        ymax = fig.range[1][1]
        print(f"  {name}: [{ymin:.4f}, {ymax:.4f}]")
    
    print("\n✅ RewardPlotter 测试完成！")
