import sys
sys.path.append("..")
sys.path.append(".")
from typing import DefaultDict
import pickle

import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

from env.vehicle import Status
from env.map_level import get_map_level
from configs import *
import os
from shapely.geometry import LinearRing

def save_parking_map(env, save_path=None, show_trajectory=None, episode_idx=None, result=None):
    """
    保存泊车地图为图片

    Args:
        env: 环境对象，包含地图信息
        save_path: 保存路径，如果为None则不保存
        show_trajectory: 轨迹点列表[(x, y), ...]，可选
        episode_idx: 回合索引，用于文件命名
        result: 泊车结果（Status枚举），可选
    """
    plt.figure(figsize=(10, 10))

    # 绘制障碍物
    for obs in env.map.obstacles:
        # LinearRing 直接使用 coords，Polygon 使用 exterior.coords
        if isinstance(obs.shape, LinearRing):
            obs_coords = np.array(obs.shape.coords[:-1])  # 移除重复的第一个点
        else:
            obs_coords = np.array(obs.shape.exterior.coords[:-1])  # 移除重复的第一个点
        plt.fill(obs_coords[:, 0], obs_coords[:, 1], color='gray', alpha=0.7)

    # 绘制起点
    start_coords = np.array(env.map.start.create_box().coords[:-1])
    plt.fill(start_coords[:, 0], start_coords[:, 1], color='green', alpha=0.8, label='Start')

    # 绘制终点
    dest_coords = np.array(env.map.dest.create_box().coords[:-1])
    plt.fill(dest_coords[:, 0], dest_coords[:, 1], color='red', alpha=0.8, label='Destination')

    # 如果有轨迹，绘制轨迹
    if show_trajectory is not None and len(show_trajectory) > 0:
        trajectory = np.array(show_trajectory)
        plt.plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=2, alpha=0.6, label='Trajectory')
        plt.plot(trajectory[0, 0], trajectory[0, 1], 'bo', markersize=8, label='Start Point')
        plt.plot(trajectory[-1, 0], trajectory[-1, 1], 'ro', markersize=8, label='End Point')

    # 设置图形属性
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # 设置标题，包含泊车结果
    # case_type = "Bay Parking" if env.map.case_id == 0 else "Parallel Parking"

    title = f'Parking Map - Case {env.map.case_id} - {result.name}'
    plt.title(title, fontsize=12, fontweight='bold')

    # 如果提供了保存路径，则保存图片
    if save_path is not None:
        # 确保目录存在
        os.makedirs(save_path, exist_ok=True)

        # 生成文件名
        if episode_idx is not None:
            filename = f'map_episode_{episode_idx:03d}_case_{env.map.case_id}.png'
        else:
            filename = f'map_case_{env.map.case_id}.png'

        filepath = os.path.join(save_path, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f'地图已保存至: {filepath}')
        plt.close()
    else:
        # 不保存，只显示
        plt.show()

def eval_with_case(env, agent, case_dir, log_path='', multi_level=False, post_proc_action=True, save_map=False, save_trajectory=False, map_save_path=None, trajectory_save_path=None):
    succ_rate_case = DefaultDict(list)
    if multi_level:
        succ_rate_level = DefaultDict(list)
        step_num_level = DefaultDict(list)
        path_length_level = DefaultDict(list)
    reward_case = DefaultDict(list)
    reward_record = []
    succ_record = []
    success_step_record = []
    step_record = DefaultDict(list)
    path_length_record = DefaultDict(list)
    eval_record = []
    failed_case_record = []

    # 如果需要保存地图或轨迹，创建保存目录
    map_save_path = ''
    if save_map:
        map_save_path = log_path + '/saved_maps'
        os.makedirs(map_save_path, exist_ok=True)

    case_files = [f for f in os.listdir(case_dir) if os.path.isfile(os.path.join(case_dir, f))]
    episode = len(case_files)

    for i in trange(episode):
        obs = env.reset_with_case(i+1, os.path.join(case_dir, case_files[i]))
        agent.reset()
        done = False
        total_reward = 0
        step_num = 0
        path_length = 0
        last_xy = (env.vehicle.state.loc.x, env.vehicle.state.loc.y)
        last_obs = obs['target']
        # 记录轨迹点
        trajectory = [(env.vehicle.state.loc.x, env.vehicle.state.loc.y)]
        while not done:
            step_num += 1
            if post_proc_action:
                action, _ = agent.choose_action(obs)
            else:
                action, _ = agent.get_action(obs)
            # 避免智能体在“卡住原地不动”
            if (last_obs == obs['target']).all():
                action = env.action_space.sample()
            last_obs = obs['target']
            next_obs, reward, done, info = env.step(action)
            total_reward += reward
            obs = next_obs
            # path_length表示泊车轨迹长度
            # np.linalg.norm(np.array(last_xy)-np.array((env.vehicle.state.loc.x, env.vehicle.state.loc.y)))是当前一步移动的距离（欧式距离）
            # 记录轨迹点
            trajectory.append((env.vehicle.state.loc.x, env.vehicle.state.loc.y))
            path_length += np.linalg.norm(np.array(last_xy)-np.array((env.vehicle.state.loc.x, env.vehicle.state.loc.y)))
            last_xy = (env.vehicle.state.loc.x, env.vehicle.state.loc.y)

            # 如果环境规划器给出了一条可行路径，则将其提供给 RL agent
            if info['path_to_dest'] is not None:
                agent.set_planner_path(info['path_to_dest'])
            if done:
                if info['status']==Status.ARRIVED:
                    succ_record.append(1)
                else:
                    # debug
                    done = True
                    succ_record.append(0)
                    # 如果泊车失败，保存视频（仅当启用按需视频保存时）
                    if hasattr(env, 'save_video_on_failure_only') and env.save_video_on_failure_only:
                        env.save_recorded_frames_to_video()

                    eval_record.append({'case_id':env.map.case_id,
                            'status':info['status'].name,
                            'step_num':step_num,
                            'reward':total_reward,
                            'path_length':path_length,
                            'episode_num': i + 1
                            })

        # 回合结束后保存地图和轨迹
        if info['status']!= Status.ARRIVED:
            save_parking_map(env, save_path=map_save_path, show_trajectory=trajectory, episode_idx=i, result=info['status'])
            failed_case_record.append(i)
        # save_parking_map(env, save_path=map_save_path, show_trajectory=trajectory, episode_idx=i, result=info['status'])
            
        reward_record.append(total_reward)
        succ_rate_case[env.map.case_id].append(succ_record[-1])
        if step_num < 200:
            path_length_record[env.map.case_id].append(path_length)
        reward_case[env.map.case_id].append(reward_record[-1])
        if multi_level:
            succ_rate_level[env.map.map_level].append(succ_record[-1])
            if step_num < 200:
                path_length_level[env.map.map_level].append(path_length)
            step_num_level[env.map.map_level].append(step_num)
        if info['status']==Status.OUTBOUND:
            step_record[env.map.case_id].append(200)
        else:
            step_record[env.map.case_id].append(step_num)
        if succ_record[-1] == 1:
            success_step_record.append(step_num)

    print('#'*15)
    print('EVALUATE RESULT:')
    print('success rate: ', np.mean(succ_record))
    print('failed cases:', failed_case_record)
    with open(log_path+"failed_cases.txt", "w") as f:
        f.write(",".join(map(str, failed_case_record)))
    print('average reward: ', np.mean(reward_record))
    print('-'*10)
    print('success rate per case: ')
    case_ids = [int(k) for k in succ_rate_case.keys()]
    case_ids.sort()
    if len(case_ids) < 10:
        print('-'*10)
        print('average reward per case: ')
        for k in case_ids:
            env.reset(k)
            print('case %s (%s) :'%(k,get_map_level(env.map.start, env.map.dest, env.map.obstacles))\
                , np.mean(succ_rate_case[k]))
        for k in case_ids:
            print('case %s :'%k, np.mean(reward_case[k]), np.mean(step_record[k]), '+-(%s)'%np.std(step_record[k]))

    if multi_level:
        print('success rate per level: ')
        for k in succ_rate_level.keys():
            print('%s (case num %s):'%(k, len(succ_rate_level[k])) + '%s '%np.mean(succ_rate_level[k]))
    
    if log_path is not None:
        def plot_time_ratio(node_list):
            max_node = TOLERANT_TIME
            raw_len = len(node_list)
            filtered_node_list = []
            for n in node_list:
                if n != max_node:
                    filtered_node_list.append(n)
            filtered_node_list.sort()
            ratio_list = [i/raw_len for i in range(1,len(filtered_node_list)+1)]
            plt.plot(filtered_node_list, ratio_list)
            plt.xlabel('Search node')
            plt.ylabel('Accumulate success rate')
            fig = plt.gcf()
            fig.savefig(log_path+'/success_rate.png')
            plt.close()
        all_step_record = []
        for k in step_record.keys():
            all_step_record.extend(step_record[k])
        plot_time_ratio(all_step_record)

        # save eval result
        f_record = open(log_path+'/record.data', 'wb')
        pickle.dump(eval_record, f_record)
        f_record.close()

        import json
        with open(log_path+'/eval_record.json', 'w', encoding='utf-8') as f:
            json.dump(eval_record, f, indent=4, ensure_ascii=False)

        f_record_txt = open(log_path+'/result.txt', 'w', newline='')
        f_record_txt.write('success rate: %s\n'%np.mean(succ_record))
        f_record_txt.write('step num: %s '%np.mean(success_step_record)+'+-(%s)\n'%np.std(success_step_record))
        if multi_level:
            f_record_txt.write('\n')
            for k in succ_rate_level.keys():
                f_record_txt.write('%s (case num %s):'%(k, len(succ_rate_level[k])) + '%s \n'%np.mean(succ_rate_level[k]))
                f_record_txt.write('step num: %s '%np.mean(step_num_level[k])+'+-(%s)\n'%np.std(step_num_level[k]))
                f_record_txt.write('path length: %s '%np.mean(path_length_level[k])+'+-(%s)\n'%np.std(path_length_level[k]))
        if len(case_ids) < 10:
            for k in case_ids:
                f_record_txt.write('\ncase %s : '%k + 'success rate: %s \n'%np.mean(succ_rate_case[k]))
                f_record_txt.write('step num: %s '%np.mean(step_record[k])+'+-(%s)\n'%np.std(step_record[k]))
                f_record_txt.write('path length: %s '%np.mean(path_length_record[k])+'+-(%s)\n'%np.std(path_length_record[k]))
        f_record_txt.close()

        status_count = {}
        for record in eval_record:
            status_name = record['status']
            status_count[status_name] = status_count.get(status_name, 0) + 1

        import csv
        with open(log_path+'/result.csv', 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['ARRIVED', 'COLLIDED', 'OUTBOUND', 'OUTTIME'])
            writer.writerow([
                status_count.get('ARRIVED', 0),
                status_count.get('COLLIDED', 0),
                status_count.get('OUTBOUND', 0),
                status_count.get('OUTTIME', 0)
            ])

    return np.mean(succ_record)


def eval(env, agent, episode=10, log_path='', multi_level=False, post_proc_action=True, save_map=False, save_trajectory=False, map_save_path=None, trajectory_save_path=None):

    succ_rate_case = DefaultDict(list)
    if multi_level:
        succ_rate_level = DefaultDict(list)
        step_num_level = DefaultDict(list)
        path_length_level = DefaultDict(list)
    reward_case = DefaultDict(list)
    reward_record = []
    succ_record = []
    success_step_record = []
    step_record = DefaultDict(list)
    path_length_record = DefaultDict(list)
    eval_record = []

    # 如果需要保存地图或轨迹，创建保存目录
    map_save_path = ''
    if save_map:
        map_save_path = log_path + '/saved_maps'
        os.makedirs(map_save_path, exist_ok=True)

    for i in trange(episode):
        obs = env.reset(i+1)
        agent.reset()
        done = False
        total_reward = 0
        step_num = 0
        path_length = 0
        last_xy = (env.vehicle.state.loc.x, env.vehicle.state.loc.y)
        last_obs = obs['target']
        # 记录轨迹点
        trajectory = [(env.vehicle.state.loc.x, env.vehicle.state.loc.y)]
        while not done:
            step_num += 1
            if post_proc_action:
                action, _ = agent.choose_action(obs)
            else:
                action, _ = agent.get_action(obs)
            # 避免智能体在“卡住原地不动”
            if (last_obs == obs['target']).all():
                action = env.action_space.sample()
            last_obs = obs['target']
            next_obs, reward, done, info = env.step(action)
            total_reward += reward
            obs = next_obs
            # path_length表示泊车轨迹长度
            # np.linalg.norm(np.array(last_xy)-np.array((env.vehicle.state.loc.x, env.vehicle.state.loc.y)))是当前一步移动的距离（欧式距离）
            # 记录轨迹点
            trajectory.append((env.vehicle.state.loc.x, env.vehicle.state.loc.y))
            path_length += np.linalg.norm(np.array(last_xy)-np.array((env.vehicle.state.loc.x, env.vehicle.state.loc.y)))
            last_xy = (env.vehicle.state.loc.x, env.vehicle.state.loc.y)

            # 如果环境规划器给出了一条可行路径，则将其提供给 RL agent
            if info['path_to_dest'] is not None:
                agent.set_planner_path(info['path_to_dest'])
            if done:
                if info['status']==Status.ARRIVED:
                    succ_record.append(1)
                else:
                    # debug
                    done = True
                    succ_record.append(0)
                    # 如果泊车失败，保存视频（仅当启用按需视频保存时）
                    if hasattr(env, 'save_video_on_failure_only') and env.save_video_on_failure_only:
                        env.save_recorded_frames_to_video()

                    eval_record.append({'case_id':env.map.case_id,
                            'status':info['status'].name,
                            'step_num':step_num,
                            'reward':total_reward,
                            'path_length':path_length,
                            'episode_num': i + 1
                            })

        # 回合结束后保存地图和轨迹
        if info['status']!= Status.ARRIVED:
            save_parking_map(env, save_path=map_save_path, show_trajectory=trajectory, episode_idx=i, result=info['status'])
            
        reward_record.append(total_reward)
        succ_rate_case[env.map.case_id].append(succ_record[-1])
        if step_num < 200:
            path_length_record[env.map.case_id].append(path_length)
        reward_case[env.map.case_id].append(reward_record[-1])
        if multi_level:
            succ_rate_level[env.map.map_level].append(succ_record[-1])
            if step_num < 200:
                path_length_level[env.map.map_level].append(path_length)
            step_num_level[env.map.map_level].append(step_num)
        if info['status']==Status.OUTBOUND:
            step_record[env.map.case_id].append(200)
        else:
            step_record[env.map.case_id].append(step_num)
        if succ_record[-1] == 1:
            success_step_record.append(step_num)

    print('#'*15)
    print('EVALUATE RESULT:')
    print('success rate: ', np.mean(succ_record))
    print('average reward: ', np.mean(reward_record))
    print('-'*10)
    print('success rate per case: ')
    case_ids = [int(k) for k in succ_rate_case.keys()]
    case_ids.sort()
    if len(case_ids) < 10:
        print('-'*10)
        print('average reward per case: ')
        for k in case_ids:
            env.reset(k)
            print('case %s (%s) :'%(k,get_map_level(env.map.start, env.map.dest, env.map.obstacles))\
                , np.mean(succ_rate_case[k]))
        for k in case_ids:
            print('case %s :'%k, np.mean(reward_case[k]), np.mean(step_record[k]), '+-(%s)'%np.std(step_record[k]))

    if multi_level:
        print('success rate per level: ')
        for k in succ_rate_level.keys():
            print('%s (case num %s):'%(k, len(succ_rate_level[k])) + '%s '%np.mean(succ_rate_level[k]))
    
    if log_path is not None:
        def plot_time_ratio(node_list):
            max_node = TOLERANT_TIME
            raw_len = len(node_list)
            filtered_node_list = []
            for n in node_list:
                if n != max_node:
                    filtered_node_list.append(n)
            filtered_node_list.sort()
            ratio_list = [i/raw_len for i in range(1,len(filtered_node_list)+1)]
            plt.plot(filtered_node_list, ratio_list)
            plt.xlabel('Search node')
            plt.ylabel('Accumulate success rate')
            fig = plt.gcf()
            fig.savefig(log_path+'/success_rate.png')
            plt.close()
        all_step_record = []
        for k in step_record.keys():
            all_step_record.extend(step_record[k])
        plot_time_ratio(all_step_record)

        # save eval result
        f_record = open(log_path+'/record.data', 'wb')
        pickle.dump(eval_record, f_record)
        f_record.close()

        import json
        with open(log_path+'/eval_record.json', 'w', encoding='utf-8') as f:
            json.dump(eval_record, f, indent=4, ensure_ascii=False)

        f_record_txt = open(log_path+'/result.txt', 'w', newline='')
        f_record_txt.write('success rate: %s\n'%np.mean(succ_record))
        f_record_txt.write('step num: %s '%np.mean(success_step_record)+'+-(%s)\n'%np.std(success_step_record))
        if multi_level:
            f_record_txt.write('\n')
            for k in succ_rate_level.keys():
                f_record_txt.write('%s (case num %s):'%(k, len(succ_rate_level[k])) + '%s \n'%np.mean(succ_rate_level[k]))
                f_record_txt.write('step num: %s '%np.mean(step_num_level[k])+'+-(%s)\n'%np.std(step_num_level[k]))
                f_record_txt.write('path length: %s '%np.mean(path_length_level[k])+'+-(%s)\n'%np.std(path_length_level[k]))
        if len(case_ids) < 10:
            for k in case_ids:
                f_record_txt.write('\ncase %s : '%k + 'success rate: %s \n'%np.mean(succ_rate_case[k]))
                f_record_txt.write('step num: %s '%np.mean(step_record[k])+'+-(%s)\n'%np.std(step_record[k]))
                f_record_txt.write('path length: %s '%np.mean(path_length_record[k])+'+-(%s)\n'%np.std(path_length_record[k]))
        f_record_txt.close()

        status_count = {}
        for record in eval_record:
            status_name = record['status']
            status_count[status_name] = status_count.get(status_name, 0) + 1

        import csv
        with open(log_path+'/result.csv', 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['ARRIVED', 'COLLIDED', 'OUTBOUND', 'OUTTIME'])
            writer.writerow([
                status_count.get('ARRIVED', 0),
                status_count.get('COLLIDED', 0),
                status_count.get('OUTBOUND', 0),
                status_count.get('OUTTIME', 0)
            ])

    return np.mean(succ_record)
