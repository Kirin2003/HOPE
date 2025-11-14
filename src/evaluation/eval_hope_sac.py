import sys
sys.path.append("..")
sys.path.append(".")
import time
import os
import argparse

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from model.agent.ppo_agent import PPOAgent as PPO
from model.agent.sac_agent import SACAgent as SAC
from model.agent.parking_agent import ParkingAgent, RsPlanner
from env.car_parking_base import CarParking
from env.env_wrapper import CarParkingWrapper
from env.vehicle import VALID_SPEED
from evaluation.eval_utils import eval, eval_with_case
from configs import *

if __name__ == '__main__':
    relative_path = '.'
    current_time = time.localtime()
    timestamp = time.strftime("%Y%m%d_%H%M%S", current_time)
    save_path = relative_path+'/log/eval/%s/' % timestamp
    os.makedirs(save_path, exist_ok=True)
    video_path = save_path + '/videos'
    os.makedirs(video_path, exist_ok=True)

    # 准备测试环境
    raw_env = CarParking(fps=100, verbose=True, render_mode='rgb_array', video_path=video_path, save_video_on_failure_only=True)
    env = CarParkingWrapper(raw_env)

    # 准备被测算法
    checkpoint_path = './model/ckpt/HOPE_SAC0.pt'
    Agent_type = SAC
    choose_action = False

    seed = int(time.time())
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

    rl_agent = Agent_type(configs)
    if checkpoint_path is not None:
        rl_agent.load(checkpoint_path, params_only=True)
        print('load pre-trained model!')
    
    step_ratio = env.vehicle.kinetic_model.step_len*env.vehicle.kinetic_model.n_step*VALID_SPEED[1]
    rs_planner = RsPlanner(step_ratio)
    parking_agent = ParkingAgent(rl_agent, rs_planner)

    case_dir = 'log/eval/20251113_143214/data'
    log_path = 'log/eval/20251113_143214/hope_sac'
    os.makedirs(log_path, exist_ok=True)
    with torch.no_grad():
        eval_with_case(env, parking_agent, case_dir=case_dir, log_path=log_path, post_proc_action=choose_action, save_map=True, save_trajectory=True)
    
    env.close()