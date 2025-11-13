import sys
sys.path.append("..")
sys.path.append(".")
import time
from model.hybridAstar.hybrid_a_star import *
import os
from tqdm import trange
from env.generator import *
from configs import *

if __name__ == '__main__':
    case_dir = 'log/eval/20251113_143214/data'
    case_files = [f for f in os.listdir(case_dir) if os.path.isfile(os.path.join(case_dir, f))]
    episode = len(case_files)

    show_animation = True

    current_time = time.localtime()
    timestamp = time.strftime("%Y%m%d_%H%M%S", current_time)
    save_path = './log/eval/%s/' % timestamp
    os.makedirs(save_path, exist_ok=True)

    # TODO trange(1) for test
    for i in trange(1):
        case_path = os.path.join(case_dir, case_files[i])
        case_data = load_case(case_path)
        start, dest, obstacles = case_data['start'], case_data['dest'], case_data['obstacles']
        ox, oy = obstacles_to_xy_lists(obstacles)
        path = hybrid_a_star_planning(start, dest, ox, oy)
        x = path.x_list
        y = path.y_list
        yaw = path.yaw_list

        if show_animation:
            start = time.time()
            import matplotlib.animation as animation
            def update_frame(i):
                plt.cla() # 清除上帧
                plt.plot(ox, oy, ".k") # 障碍点（静态）
                plt.plot(x, y, "-r", label="Hybrid A* path") # 整条轨迹（参考）
                plt.grid(True)
                plt.axis("equal")
                if i < len(x):
                    plot_car(x[i], y[i], yaw[i]) # 在 (ix,iy) 处画车轮廓（带旋转）
                return []
            
            fig = plt.figure()
            anim = animation.FuncAnimation(
                fig, 
                update_frame, 
                frames=len(x), 
                interval=100,
            )
            
            anim.save(f'{save_path}/{i}.gif', writer='pillow', fps=10)
            plt.close(fig)
            end = time.time()
            print("animation time:{:.4f} s".format(end-start))