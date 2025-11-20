import sys
sys.path.append("..")
sys.path.append(".")
import time
from model.hybridAstar.hybrid_a_star import *
import os
from tqdm import trange
from env.generator import *
from configs import *
from evaluation.monitor import Monitor
from evaluation.monitor import Status
from evaluation.visual_utils import * 
from collections import defaultdict

if __name__ == '__main__':
    case_dir = 'log/eval/vertical_complex/data'
    case_files = [f for f in os.listdir(case_dir) if os.path.isfile(os.path.join(case_dir, f))]
    # episode = len(case_files)

    log_path = 'log/eval/vertical_complex/hybridAstar_2'
    os.makedirs(log_path, exist_ok=True)
    figure_save_path = f'{log_path}/figure'
    os.makedirs(figure_save_path, exist_ok=True)

    status_counter = defaultdict(int)
    failed_case_record = []

    episode = 20

    time1 = 0

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

        # 没有找到可行路径
        if len(x_list) == 0:
            vehicle_status = Status.NOPATH
            status_counter[Status.NOPATH] += 1
            failed_case_record.append(i)
            title = f'Case_{i}_{vehicle_status.name}'
            plot_case(case_data['start'], case_data['dest'], case_data['obstacles'], title=title, save_path = f'{figure_save_path}/{title}.png')
        else:
            monitor = Monitor(path, dest, obstacles)
            vehicle_status = monitor.check()
            status_counter[vehicle_status] += 1
            # failed
            if vehicle_status != Status.ARRIVED:
                print(f'case {i} failed: {vehicle_status.name}')
                failed_case_record.append(i)
                title = f'Case_{i}_{vehicle_status.name}'
                plot_planning_result(x_list, y_list, yaw_list, obstacles, dest, collision_idx=monitor.collision_index, title=title, save_path=f'{figure_save_path}/{title}.png')

    time1 = time.time() - time1
    print('#'*15)
    print('average time per case: {:.4f}'.format(time1 / episode))
    print('success rate: {:.4f}'.format(1 - len(failed_case_record) / episode))
    print('failed cases: ', failed_case_record)

    with open(log_path+"/failed_cases.txt", "w") as f:
        f.write(",".join(map(str, failed_case_record)))
    
    import csv
    with open(log_path+'/result.csv', 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([Status.ARRIVED.name, Status.COLLIDED.name, Status.NOPATH.name, Status.OUTTIME.name])
        writer.writerow([
            status_counter[Status.ARRIVED],
            status_counter[Status.COLLIDED],
            status_counter[Status.NOPATH],
            status_counter[Status.OUTTIME]
        ])

    # case_dir = 'log/eval/20251114_043028/data'
    # case_files = [f for f in os.listdir(case_dir) if os.path.isfile(os.path.join(case_dir, f))]
    # log_path = 'log/eval/20251114_043028/hybridAstar'
    # for i in trange(6,11):
    #     case_path = os.path.join(case_dir, case_files[i])
    #     case_data = load_case(case_path)
    #     start, dest, obstacles = case_data['start'], case_data['dest'], case_data['obstacles']
    #     ox, oy = obstacles_to_xy_lists(obstacles)
    #     path = hybrid_a_star_planning(start, dest, ox, oy)
    #     title=f'Grid_Case_{i}'
    #     plot_planning_result_in_grid_env(path.x_list, path.y_list, path.yaw_list, ox, oy, dest, title=title, save_path=f'{log_path}/{title}.png')
    #                                 #  , dest, ox, oy, title='Case_6_grid')