#!/usr/bin/env python3
"""
测试视频保存功能的示例脚本

这个脚本演示了如何使用CarParking环境的视频录制功能。
每次调用reset()会创建一个新的视频文件。
"""

import sys
import os
sys.path.append('../')

from env.car_parking_base import CarParking

def main():
    # 创建环境
    print("创建CarParking环境...")
    env = CarParking(render_mode="human")

    # 运行几个episode，每个episode会自动保存一个视频
    for episode in range(3):
        print(f"\n=== Episode {episode + 1} ===")

        # 重置环境（这会自动创建新的视频文件）
        observation = env.reset()

        # 运行episode直到完成
        done = False
        step_count = 0
        max_steps = 500

        while not done and step_count < max_steps:
            # 随机选择动作（这里用随机动作作为示例）
            # 实际使用时，您应该使用训练好的策略
            action = env.action_space.sample()

            # 执行动作
            observation, reward_info, status, info = env.step(action)

            # 检查是否结束
            if status.value in ['ARRIVED', 'COLLIDED', 'OUTBOUND', 'OUTTIME']:
                print(f"Episode结束，状态: {status.value}")
                done = True

            step_count += 1

        if step_count >= max_steps:
            print(f"Episode达到最大步数: {max_steps}")

    # 关闭环境（释放视频资源）
    print("\n关闭环境...")
    env.close()

    print("\n视频保存完成！")
    print(f"视频文件保存在: {os.path.abspath(env.video_output_dir if hasattr(env, 'video_output_dir') else 'videos')} 目录")

if __name__ == "__main__":
    main()
