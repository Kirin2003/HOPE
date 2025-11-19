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
    """
    根据车辆位置(x, y)和航向角绘制车辆轮廓
    
    参数:
        x: 车辆中心x坐标
        y: 车辆中心y坐标  
        yaw: 车辆航向角（弧度）
        color: 轮廓颜色，默认为黑色
        label: 图例标签（可选）
    """
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
    """
    可视化路径规划测试场景
    
    绘制完整的测试环境，包括起始位置、目标停车位和障碍物布局，
    用于验证路径规划算法的运行环境。

    参数:
        start: 起始位置 (x, y, yaw)
        dest: 目标停车位 (x, y, yaw) 
        obstacles: 障碍物列表，每个障碍物为LinearRing类型
        title: 图像标题（可选）
        save_path: 保存路径（可选），如果提供则保存图像
    """
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

def plot_case_in_grid_env(start, dest, ox, oy, title=None, save_path=None):
    """
    绘制离散网格环境下的路径规划测试场景
    
    在网格化环境中可视化起始位置、目标停车位和离散障碍物分布，
    适用于基于网格的路径规划算法场景展示。

    参数:
        start: 起始位置，格式为(x, y, yaw)的元组
        dest: 目标停车位，格式为(x, y, yaw)的元组
        ox: 障碍物的x坐标列表
        oy: 障碍物的y坐标列表
        title: 图像标题（可选）
        save_path: 图像保存路径（可选），如提供则保存图像文件
    """
    plt.figure(figsize=(10, 10))
    # 绘制起点
    start_x, start_y, start_yaw = start
    plot_car(start_x, start_y, start_yaw, color='g', label='start position')
    # 绘制目标车位
    dest_x, dest_y, dest_yaw = dest
    plot_car(dest_x, dest_y, dest_yaw, color='b', label='target parking lot')
    # 绘制障碍物（离散）
    plt.plot(ox, oy, ".k") 

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



def animation_planning_result(x_list, y_list, yaw_list, ox, oy, figure_save_path ):
    """
    生成路径规划结果的动态演示动画
    
    通过逐帧绘制车辆在规划路径上的运动过程，直观展示路径规划效果。
    动画包含静态障碍物、全局参考路径和动态车辆姿态。

    参数:
        x_list: x坐标序列，描述路径轨迹
        y_list: y坐标序列，描述路径轨迹  
        yaw_list: 航向角序列，描述车辆朝向
        ox: 障碍物x坐标列表
        oy: 障碍物y坐标列表
        figure_save_path: 动画保存路径
    """
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
    
    anim.save(figure_save_path, writer='pillow', fps=10)
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

def plot_planning_result_in_grid_env(x_list, y_list, yaw_list, ox, oy, dest, collision_idx=None, title=None, save_path=None, ):
    plt.figure(figsize=(10, 10))
    # 绘制障碍物
    plt.plot(ox, oy, '.k')
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