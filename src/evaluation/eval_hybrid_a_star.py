import sys
sys.path.append("..")
sys.path.append(".")
import time
from model.hybridAstar.hybrid_a_star import *
import os
from tqdm import trange
from env.generator import *
from configs import *
from shapely.geometry import Polygon, LinearRing
from env.vehicle import State
from env.generator import visual_case

class Monitor:
    def __init__(self, planning_path, dest, obstacles):
        self.dest_box = self._generate_box(dest[0], dest[1], dest[2])
        self.obstacles = obstacles

        self.vehicle_centers = []
        self.vehicle_boxes = []
        x_list = path.x_list
        y_list = path.y_list
        yaw_list = path.yaw_list
        leng = len(planning_path)
        for i in range(leng):
            x = x_list[i]
            y = y_list[i]
            yaw = yaw_list[i]
            self.vehicle_boxes.append(self._generate_box(x, y, yaw))
            self.vehicle_centers.append((x, y, yaw))

    def _generate_box(self, x, y, yaw) -> LinearRing:
        rb, rf, lf, lb  = list(State([x, y, yaw, 0, 0]).create_box().coords)[:-1]
        box = LinearRing((rb, rf, lf, lb))
        return box
    
    def _check_arrived(self):
        vehicle_polygon = Polygon(self.vehicle_boxes[-1])
        dest_polygon = Polygon(self.dest_box)
        union_area = vehicle_polygon.intersection(dest_polygon).area
        if union_area / dest_polygon.area > 0.95:
            return True
        return False
    
    def _detect_collision(self):
        if not self.obstacles:
            return False
        for vehicle in self.vehicle_boxes:
            vehicle_polygon = Polygon(vehicle)
            for obstacle in self.obstacles:
                obstacle_polygon = Polygon(obstacle)
                if vehicle_polygon.intersection(obstacle_polygon):
                    return True
        return False

    def _detect_outbound(self):
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
    
    def _check_time_exceeded(self):
        # TODO
        return False

def animation_case(x_list, y_list, yaw_list, ox, oy):
    start = time.time()
    import matplotlib.animation as animation
    def update_frame(i):
        plt.cla() # 清除上帧
        plt.plot(ox, oy, ".k") # 障碍点（静态）
        plt.plot(x_list, y_list, "-r", label="Hybrid A* path") # 整条轨迹（参考）
        plt.grid(True)
        plt.axis("equal")
        if i < len(x_list):
            plot_car(x_list[i], y_list[i], yaw_list[i]) # 在 (ix,iy) 处画车轮廓（带旋转）
        return []
    
    fig = plt.figure()
    anim = animation.FuncAnimation(
        fig, 
        update_frame, 
        frames=len(x_list), 
        interval=100,
    )
    
    anim.save(f'{figure_save_path}/{i}.gif', writer='pillow', fps=10)
    plt.close(fig)
    end = time.time()
    print("animation time:{:.4f} s".format(end-start))

def plot_case(x_list, y_list, yaw_list, obstacles, idx = -1, save_path=None):
    plt.figure(figsize=(10, 10))
    # 绘制障碍物
    for obs in obstacles:
        obs_coords = np.array(obs.coords[:-1])  # 移除重复的第一个点
        
        plt.fill(obs_coords[:, 0], obs_coords[:, 1], color='gray', alpha=0.7)

    plt.plot(x_list, y_list, "-r", label="Hybrid A* path") # 整条轨迹（参考）
    plt.grid(True, alpha=0.3)
    plt.axis("equal")
    plt.legend()
    if save_path:
        filename = f'{idx}.png'
        filepath = os.path.join(save_path, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f'地图已保存至: {filepath}')
        plt.close()
    else:
        plt.show()


if __name__ == '__main__':
    case_dir = 'log/eval/20251113_143214/data'
    case_files = [f for f in os.listdir(case_dir) if os.path.isfile(os.path.join(case_dir, f))]
    # episode = len(case_files)

    failed_case_record = []
    log_path = 'log/eval/20251113_143214/hybridAstar'

    current_time = time.localtime()
    timestamp = time.strftime("%Y%m%d_%H%M%S", current_time)
    save_path = './log/eval/%s/' % timestamp
    # os.makedirs(save_path, exist_ok=True)
    figure_save_path = f'{log_path}/figure'
    os.makedirs(figure_save_path, exist_ok=True)

    episode = 10

    for i in trange(episode):
        success = False
        case_path = os.path.join(case_dir, case_files[i])
        case_data = load_case(case_path)
        start, dest, obstacles = case_data['start'], case_data['dest'], case_data['obstacles']
        ox, oy = obstacles_to_xy_lists(obstacles)
        path = hybrid_a_star_planning(start, dest, ox, oy)
        x_list = path.x_list
        y_list = path.y_list
        yaw_list = path.yaw_list

        if len(x_list) == 0:
            visual_case(case_data, save_path = figure_save_path)
        else:
            planning_path = (x_list, y_list, yaw_list)

            monitor = Monitor(planning_path, dest, obstacles)
            success = monitor._check_arrived() & (not monitor._detect_collision()) & (not monitor._detect_outbound()) & (not monitor._check_time_exceeded())

            if not success:
                failed_case_record.append(i)
                plot_case(x_list, y_list, yaw_list, obstacles, i, figure_save_path)
                
    print('#'*15)
    print('success rate: {:.4f}'.format(1 - len(failed_case_record) / episode))
    print('failed cases: ', failed_case_record)
    with open(log_path+"failed_cases.txt", "w") as f:
        f.write(",".join(map(str, failed_case_record)))
    