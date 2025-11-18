import sys
sys.path.append("..")
sys.path.append(".")
import time
from model.hybridAstar.hybrid_a_star import *
import os
from tqdm import trange
from env.generator import *
from configs import *
from env.generator import visual_case
from evaluation.monitor import Monitor
from env.vehicle import State, Status
from evaluation.visual_utils import plot_case, animation_case


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
        # start: [x, y, yaw]
        # dest: [x, y, yaw]
        # obstacles: [obstacle (LinearRing), ...]
        start, dest, obstacles = case_data['start'], case_data['dest'], case_data['obstacles']
        ox, oy = obstacles_to_xy_lists(obstacles)
        path = hybrid_a_star_planning(start, dest, ox, oy)
        x_list = path.x_list
        y_list = path.y_list
        yaw_list = path.yaw_list

        if len(x_list) == 0:
            visual_case(case_data, save_path = figure_save_path)
        else:
            monitor = Monitor(path, dest, obstacles)
            # failed
            vehicle_status = monitor.check()
            if vehicle_status != Status.ARRIVED:
                print(f'case {i} failed: {vehicle_status.name}')
                failed_case_record.append(i)
                plot_case(x_list, y_list, yaw_list, obstacles, dest, i, figure_save_path)  
                # animation_case(x_list, y_list, yaw_list, ox, oy, i, figure_save_path)
    print('#'*15)
    print('success rate: {:.4f}'.format(1 - len(failed_case_record) / episode))
    print('failed cases: ', failed_case_record)
    with open(log_path+"failed_cases.txt", "w") as f:
        f.write(",".join(map(str, failed_case_record)))
    