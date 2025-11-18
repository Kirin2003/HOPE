import sys
sys.path.append("..")
sys.path.append(".")
from env.vehicle import State, Status
from shapely.geometry import LinearRing, Polygon

class Monitor:
    def __init__(self, path, dest, obstacles):
        self.dest_box = self.generate_box(dest[0], dest[1], dest[2])
        print('generate dest_box')
        self.obstacles = obstacles

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
        if not self.obstacles:
            return False
        for vehicle in self.vehicle_boxes:
            for obstacle in self.obstacles:
                if vehicle.intersection(obstacle):
                    print(f'vehicle coords: {list(vehicle.coords)}')
                    print(f'obstacle coords: {list(obstacle.coords)}')
                    return True, vehicle, obstacle
        return False

    def detect_outbound(self):
        if not self.obstacles:
            return False

        # 计算地图边界：从所有障碍物的坐标中找出最小外接矩形
        all_x = []
        all_y = []
        for obstacle in self.obstacles:
            coords = list(obstacle.coords)
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
        if self.detect_outbound():
            return Status.OUTBOUND
        if self.check_arrived():
            return Status.ARRIVED
        if self.check_time_exceeded():
            return Status.OUTTIME
        return Status.CONTINUE

def plot_collision(vehicle, obstacle, case_id=-1, figure_save_path=None):
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    plt.figure(figsize=(10, 10))
    obs_coords = np.array(obstacle[:-1])
    plt.fill(obs_coords[:, 0], obs_coords[:, 1], color='gray', alpha=0.7)
    vehicle_coords = np.array(vehicle[:-1])
    plt.fill(vehicle_coords[:, 0], vehicle_coords[:, 1], color='red', alpha=0.7)
    plt.grid(True, alpha=0.3)
    plt.axis("equal")
    plt.legend()
    if figure_save_path:
        filename = f'{case_id}.png'
        filepath = os.path.join(figure_save_path, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f'碰撞图已保存至: {filepath}')
        plt.close()
    else:
        plt.show()

if __name__ == '__main__':

    vehicle = [(1.1547149462734203, 8.577770294481812), (5.756612593856238, 9.482556479816074), (5.382351357278739, 11.386113289263978), (0.7804537096959214, 10.481327103929715), (1.1547149462734203, 8.577770294481812)]
    obstacle = [(-15.0, 11.363295299983607), (15.0, 11.363295299983607), (15.0, 11.463295299983606), (-15.0, 11.463295299983606), (-15.0, 11.363295299983607)]
    plot_collision(vehicle, obstacle)