import sys
sys.path.append("..")
sys.path.append(".")
import os
import time
import matplotlib.pyplot as plt
from math import cos, sin
import numpy as np
from model.hybridAstar.car import VRX, VRY
from scipy.spatial.transform import Rotation as Rot

def rot_mat_2d(angle):
    """
    Create 2D rotation matrix from an angle

    Parameters
    ----------
    angle :

    Returns
    -------
    A 2D rotation matrix

    Examples
    --------
    >>> angle_mod(-4.0)


    """
    return Rot.from_euler('z', angle).as_matrix()[0:2, 0:2]

def plot_car(x, y, yaw, color='k', label=None):
    car_color = color
    c, s = cos(yaw), sin(yaw)
    rot = rot_mat_2d(-yaw)
    car_outline_x, car_outline_y = [], []
    for rx, ry in zip(VRX, VRY):
        converted_xy = np.stack([rx, ry]).T @ rot
        car_outline_x.append(converted_xy[0]+x)
        car_outline_y.append(converted_xy[1]+y)

    # arrow_x, arrow_y, arrow_yaw = c * 1.5 + x, s * 1.5 + y, yaw
    # plot_arrow(arrow_x, arrow_y, arrow_yaw)

    plt.plot(car_outline_x, car_outline_y, car_color, label=label)

def plot_case(start, dest, obstacles, title=None, save_path=None):
    plt.figure(figsize=(10, 10))
    # 绘制起点
    start_x, start_y, start_yaw = start
    plot_car(start_x, start_y, start_yaw, color='g', label='start position')
    # 绘制目标车位
    dest_x, dest_y, dest_yaw = dest
    plot_car(dest_x, dest_y, dest_yaw, color='b', label='target parking lot')

    # 绘制障碍物
    for obs in obstacles:
        obs_coords = np.array(obs.coords[:-1])  # 移除重复的第一个点
        plt.fill(obs_coords[:, 0], obs_coords[:, 1], color='gray', alpha=0.7)
    
    plt.grid(True, alpha=0.3)
    plt.axis("equal")
    plt.legend()
    plt.title(title)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'save case in: {save_path}')
        plt.close()
    else:
        plt.show()

def animation_planning_result(x_list, y_list, yaw_list, ox, oy, case_id, figure_save_path ):
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
    
    anim.save(f'{figure_save_path}/{case_id}.gif', writer='pillow', fps=10)
    plt.close(fig)
    end = time.time()
    print("animation time:{:.4f} s".format(end-start))

def plot_planning_result(x_list, y_list, yaw_list, obstacles, dest, collision_idx=None, title=None, save_path=None, ):
    plt.figure(figsize=(10, 10))
    # 绘制障碍物
    for obs in obstacles:
        obs_coords = np.array(obs.coords[:-1])  # 移除重复的第一个点

        plt.fill(obs_coords[:, 0], obs_coords[:, 1], color='gray', alpha=0.7)

    # 起点（绿色）
    plot_car(x_list[0], y_list[0], yaw_list[0], color='g', label='start position')
    # 目标车位（蓝色）
    plot_car(dest[0], dest[1], dest[2], color='b', label='target parking lot')
    # 整条轨迹（红色）
    plt.plot(x_list, y_list, "r", label="planning path")

    # 如果指定了碰撞索引，绘制碰撞时刻的汽车（橙色）
    if collision_idx is not None and 0 <= collision_idx < len(x_list):
        plot_car(x_list[collision_idx], y_list[collision_idx], yaw_list[collision_idx], color='orange', label='collision point')

    plt.grid(True, alpha=0.3)
    plt.axis("equal")
    plt.legend()
    plt.title(title)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'save case in: {save_path}')
        plt.close()
    else:
        plt.show()