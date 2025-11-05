import numpy as np
from shapely.geometry import Polygon

class HybridAStarPlanner:
    def __init__(self, vehicle_params):
        self.vehicle_length = vehicle_params['length']  # 车辆长度
        self.vehicle_width = vehicle_params['width']    # 车辆宽度
        self.max_steer = vehicle_params['max_steer']    # 最大转向角
        self.step_size = 0.5  # 路径采样步长

    def plan(self, start_state, goal_state, obstacles):
        """
        规划从起点到终点的路径
        :param start_state: (x, y, heading) 起点状态
        :param goal_state: (x, y, heading) 终点状态
        :param obstacles: 障碍物列表（每个为Polygon或LinearRing）
        :return: 路径列表 [(x0,y0,h0), (x1,y1,h1), ...]
        """
        # 此处实现Hybrid A*核心逻辑：
        # 1. 初始化搜索树
        # 2. 扩展节点（考虑车辆运动学约束生成子节点）
        # 3. 碰撞检测（利用obstacles判断节点是否安全）
        # 4. 到达目标时回溯路径
        # （实际实现需参考Hybrid A*算法细节）
        path = self._dummy_path(start_state, goal_state)  # 临时虚拟路径
        return path

    def _dummy_path(self, start, goal):
        """临时虚拟路径（仅用于示例，需替换为真实算法）"""
        x = np.linspace(start[0], goal[0], 50)
        y = np.linspace(start[1], goal[1], 50)
        h = np.linspace(start[2], goal[2], 50)
        return list(zip(x, y, h))

    def is_collision_free(self, state, obstacles):
        """检查当前状态（位置+朝向）是否与障碍物碰撞"""
        x, y, h = state
        # 根据车辆尺寸和朝向生成车辆矩形
        half_len = self.vehicle_length / 2
        half_width = self.vehicle_width / 2
        corners = [
            (x + half_len*np.cos(h) - half_width*np.sin(h), y + half_len*np.sin(h) + half_width*np.cos(h)),
            (x + half_len*np.cos(h) + half_width*np.sin(h), y + half_len*np.sin(h) - half_width*np.cos(h)),
            (x - half_len*np.cos(h) + half_width*np.sin(h), y - half_len*np.sin(h) - half_width*np.cos(h)),
            (x - half_len*np.cos(h) - half_width*np.sin(h), y - half_len*np.sin(h) + half_width*np.cos(h))
        ]
        vehicle_poly = Polygon(corners)
        for obs in obstacles:
            if vehicle_poly.intersects(obs):
                return False
        return True