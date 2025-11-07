import sys
sys.path.append("..")
sys.path.append(".")
import time
import os
import argparse

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

# from model.MultiModalPPO_AF import PPO
from model.agent.ppo_agent import PPOAgent as PPO
from model.agent.sac_agent import SACAgent as SAC
from model.agent.parking_agent import ParkingAgent, RsPlanner
from env.car_parking_base import CarParking
from env.env_wrapper import CarParkingWrapper
from env.vehicle import VALID_SPEED
from evaluation.eval_utils import eval
from configs import *


if __name__=="__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, default='./model/ckpt/HOPE_SAC0.pt')
    parser.add_argument('--eval_episode', type=int, default=100)
    parser.add_argument('--verbose', type=bool, default=True)
    parser.add_argument('--visualize', type=bool, default=False)
    args = parser.parse_args()

    checkpoint_path = args.ckpt_path
    print('ckpt path: ',checkpoint_path)
    verbose = args.verbose

    relative_path = '.'
    current_time = time.localtime()
    timestamp = time.strftime("%Y%m%d_%H%M%S", current_time)
    save_path = relative_path+'/log/eval/%s/' % timestamp
    os.makedirs(save_path, exist_ok=True)
    video_path = save_path + '/videos'
    os.makedirs(video_path, exist_ok=True)

    # 准备测试环境（地图和障碍物）
    if args.visualize:
        raw_env = CarParking(fps=100, verbose=verbose, video_path=video_path, save_video_on_failure_only=True)
    else:
        raw_env = CarParking(fps=100, verbose=verbose, render_mode='rgb_array', video_path=video_path, save_video_on_failure_only=True)
    env = CarParkingWrapper(raw_env)

    # 记录日志，从配置文件里读取agent类型
    
    configs_file = os.path.join(save_path, 'configs.txt')
    Agent_type = PPO if 'ppo' in checkpoint_path.lower() else SAC
    writer = SummaryWriter(save_path)
    print("You can track the training process by command 'tensorboard --log-dir %s'" % save_path)

    # seed = SEED
    seed = int(time.time())
    with open(configs_file, 'w') as f:
        f.write('checkpoint_path: '+str(checkpoint_path))
        f.write('\nseed: '+str(seed))
    env.action_space.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    actor_params = ACTOR_CONFIGS
    critic_params = CRITIC_CONFIGS
    configs = {
        "discrete": False,
        "observation_shape": env.observation_shape,
        "action_dim": env.action_space.shape[0],
        "hidden_size": 64,
        "activation": "tanh",
        "dist_type": "gaussian",
        "save_params": False,
        "actor_layers": actor_params,
        "critic_layers": critic_params,
    }
    print('observation_space:',env.observation_space)

    # 被测算法，RL_agent+RS_planner
    rl_agent = Agent_type(configs)
    if checkpoint_path is not None:
        rl_agent.load(checkpoint_path, params_only=True)
        print('load pre-trained model!')

    step_ratio = env.vehicle.kinetic_model.step_len*env.vehicle.kinetic_model.n_step*VALID_SPEED[1]
    rs_planner = RsPlanner(step_ratio)
    parking_agent = ParkingAgent(rl_agent, rs_planner)

    eval_episode = args.eval_episode
    # PPO 评估用随机动作（需要后处理如采样），SAC 评估用确定性动作（均值）
    choose_action = True if isinstance(rl_agent, PPO) else False
    with torch.no_grad():
        # eval on extreme
        # env.set_level()只设置难度等级，在eval()里个episode前会调用env.reset()生成符合该难度的环境数据
        env.set_level('Extrem')
        log_path = save_path+'/extreme'
        os.makedirs(log_path, exist_ok=True)
        eval(env, parking_agent, episode=eval_episode, log_path=log_path, post_proc_action=choose_action, save_map=True, save_trajectory=True)

        # # eval on dlp
        # env.set_level('dlp')
        # log_path = save_path+'/dlp'
        # os.makedirs(log_path, exist_ok=True)
        # eval(env, parking_agent, episode=eval_episode, log_path=log_path, multi_level=True, post_proc_action=choose_action, save_map=True, save_trajectory=True)
        #
        # # eval on complex
        # env.set_level('Complex')
        # log_path = save_path+'/complex'
        # os.makedirs(log_path, exist_ok=True)
        # eval(env, parking_agent, episode=eval_episode, log_path=log_path, post_proc_action=choose_action, save_map=True, save_trajectory=True)
        #
        # # eval on normalize
        # env.set_level('Normal')
        # log_path = save_path+'/normalize'
        # os.makedirs(log_path, exist_ok=True)
        # eval(env, parking_agent, episode=eval_episode, log_path=log_path, post_proc_action=choose_action, save_map=True, save_trajectory=True)

    env.close()