import sys
sys.path.append("..")
sys.path.append(".")
from env.vehicle import State
from shapely.geometry import LinearRing, Polygon
from enum import Enum

class Status(Enum):
    CONTINUE = 1
    ARRIVED = 2
    COLLIDED = 3
    OUTBOUND = 4
    OUTTIME = 5
    NOPATH = 6

class Monitor:
    def __init__(self, path, dest, obstacles):
        self.dest_box = self.generate_box(dest[0], dest[1], dest[2])
        self.obstacles = obstacles
        self.collision_index = None  # 存储第一个碰撞时刻的索引

        self.vehicle_centers = []
        self.vehicle_boxes = []
        x_list = path.x_list
        y_list = path.y_list
        yaw_list = path.yaw_list
        leng = len(x_list)
        for i in range(leng):
            x = x_list[i]
            y = y_list[i]
            yaw = yaw_list[i]
            self.vehicle_boxes.append(self.generate_box(x, y, yaw))
            self.vehicle_centers.append((x, y, yaw))

    def generate_box(self, x, y, yaw) -> LinearRing:
        rb, rf, lf, lb  = list(State([x, y, yaw, 0, 0]).create_box().coords)[:-1]
        box = LinearRing((rb, rf, lf, lb))
        return box
    
    def check_arrived(self):
        vehicle_polygon = Polygon(self.vehicle_boxes[-1])
        dest_polygon = Polygon(self.dest_box)
        union_polygon = vehicle_polygon.intersection(dest_polygon)
        if union_polygon.area / dest_polygon.area > 0.95:
            return True
        return False
    
    def detect_collision(self):
        """检测第一个碰撞时刻，返回(是否有碰撞, 碰撞时刻索引)"""
        if not self.obstacles:
            return False
        for i, vehicle in enumerate(self.vehicle_boxes):
            for obstacle in self.obstacles:
                if vehicle.intersection(obstacle):
                    self.collision_index = i
                    return True  # 返回第一个碰撞的时刻索引
        return False

    def detect_outbound(self):
        if not self.obstacles:
            return False

        # 计算地图边界：从所有障碍物的坐标中找出最小外接矩形
        all_x = []
        all_y = []
        for obstacle in self.obstacles:
            coords = list(obstacle.coords)[:-1]
            xs = [coord[0] for coord in coords]
            ys = [coord[1] for coord in coords]
            all_x.extend(xs)
            all_y.extend(ys)

        x_min, x_max = min(all_x), max(all_x)
        y_min, y_max = min(all_y), max(all_y)

        # 检查每个车辆边界框是否超出边界
        for (x, y, yaw) in self.vehicle_centers:
                if x < x_min or x > x_max or y < y_min or y > y_max:
                    return True
        return False
    
    def check_time_exceeded(self):
        # TODO
        return False

    def check(self):
        if self.detect_collision():
            return Status.COLLIDED
        # if self.detect_outbound():
        #     return Status.OUTBOUND
        if self.check_arrived():
            return Status.ARRIVED
        # if self.check_time_exceeded():
        #     return Status.OUTTIME
        return Status.OUTTIME

if __name__ == '__main__':

    pass