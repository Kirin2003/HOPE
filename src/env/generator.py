import sys
sys.path.append("../")
sys.path.append(".")
from configs import *
import matplotlib.pyplot as plt
from numpy.random import randn, random
from env.vehicle import State
from math import pi, cos, sin
import time
from configs import *

# params for generating parking case
prob_huge_obst = 0.5
n_non_critical_car = 3
prob_non_critical_car = 0.7

current_time = time.localtime()
timestamp = time.strftime("%Y%m%d_%H%M%S", current_time)

DEBUG = False

def random_gaussian_num(mean, std, clip_low, clip_high):
    rand_num = randn()*std + mean
    return np.clip(rand_num, clip_low, clip_high)

def random_uniform_num(clip_low, clip_high):
    rand_num = random()*(clip_high - clip_low) + clip_low
    return rand_num

"""
在一个扇形区域内随机生成一个点坐标
即：以 (origin_x, origin_y) 为圆心，从：
  角度范围：[angle_min, angle_max]
  半径范围：[radius_min, radius_max]
中按照高斯随机的方式采样一个位置(rand_pos_x, rand_pos_y)。
  rand_pos_x = origin_x + r * cos(angle)
  rand_pos_y = origin_y + r * sin(angle)
"""
def get_rand_pos(origin_x, origin_y, angle_min, angle_max, radius_min, radius_max):
    angle_mean = (angle_max+angle_min)/2
    angle_std = (angle_max-angle_min)/4
    angle_rand = random_gaussian_num(angle_mean, angle_std, angle_min, angle_max)
    radius_rand = random_gaussian_num((radius_min+radius_max)/2, (radius_max-radius_min)/4, radius_min, radius_max)
    return origin_x+cos(angle_rand)*radius_rand, origin_y+sin(angle_rand)*radius_rand


def random_generate(case_idx):
    '''
    Generate the parameters that a bay parking case need.
    
    Returns
    ----------
        `start` (list): [x, y, yaw]
        `dest` (list): [x, y, yaw]
        `obstacles` (list): [ obstacle (`LinearRing`) , ...]
    '''
    # TODO test
    map_level = 'Complex'

    origin = (0., 0.)
    bay_half_len = 15.
    # params related to map level
    max_BAY_PARK_LOT_WIDTH = MAX_PARK_LOT_WIDTH_DICT[map_level]
    min_BAY_PARK_LOT_WIDTH = MIN_PARK_LOT_WIDTH_DICT[map_level]
    bay_PARK_WALL_DIST = BAY_PARK_WALL_DIST_DICT[map_level]
    n_obst = N_OBSTACLE_DICT[map_level]
    max_lateral_space = max_BAY_PARK_LOT_WIDTH - WIDTH
    min_lateral_space = min_BAY_PARK_LOT_WIDTH - WIDTH

    generate_success = True
    # generate obstacle on back
    obstacle_back = LinearRing(( 
        (origin[0]+bay_half_len, origin[1]),
        (origin[0]+bay_half_len, origin[1]-1), 
        (origin[0]-bay_half_len, origin[1]-1), 
        (origin[0]-bay_half_len, origin[1])))

    # generate dest
    # 泊车终点的角度，单位是弧度
    dest_yaw = random_gaussian_num(pi/2, pi/36, pi*5/12, pi*7/12)
    rb, _, _, lb  = list(State([origin[0], origin[1], dest_yaw, 0, 0]).create_box().coords)[:-1]
    # rb 和 lb 是 车体后边的两个角，min(rb[1], lb[1]) 给出 车体在 y 轴方向最靠负方向的点。
    # 再取 -min(...) 就变成“为了让车完全不跌出 y=0 以下，至少需要把车向 +y 方向移动这么远”。
    # 然后加上 MIN_DIST_TO_OBST = 预留一点和墙/障碍之间的最小安全距离。
    min_dest_y = -min(rb[1], lb[1]) + MIN_DIST_TO_OBST
    # (dest_x,dest_y)是终点坐标
    dest_x = origin[0]
    dest_y = random_gaussian_num(min_dest_y+0.4, 0.2, min_dest_y, min_dest_y+0.8)
    car_rb, car_rf, car_lf, car_lb  = list(State([dest_x, dest_y, dest_yaw, 0, 0]).create_box().coords)[:-1]
    dest_box = LinearRing((car_rb, car_rf, car_lf, car_lb))


    non_critical_vehicle = []
    # generate obstacle on left
    # the obstacle can be another vehicle or just a simple obstacle
    if random()<prob_huge_obst: # generate simple obstacle
        max_dist_to_obst = max_lateral_space/5*4
        min_dist_to_obst = max_lateral_space/5*1
        # 左边障碍物的右前角
        left_obst_rf = get_rand_pos(*car_lf, pi*11/12, pi*13/12, min_dist_to_obst, max_dist_to_obst)
        # 左边障碍物的右后角
        left_obst_rb = get_rand_pos(*car_lb, pi*11/12, pi*13/12, min_dist_to_obst, max_dist_to_obst)
        obstacle_left = LinearRing((
            left_obst_rf,
            left_obst_rb,
            (origin[0]-bay_half_len, origin[1]), # 左边障碍物的左后方 TODO 不理解
            (origin[0]-bay_half_len, left_obst_rf[1]))) # 左边障碍物的左前方
    else: # generate another vehicle as obstacle on left
        max_dist_to_obst = max_lateral_space/5*4
        min_dist_to_obst = max_lateral_space/5*1
        left_car_x = origin[0] - (WIDTH + random_uniform_num(min_dist_to_obst, max_dist_to_obst))
        left_car_yaw = random_gaussian_num(pi/2, pi/36, pi*5/12, pi*7/12)
        rb, _, _, lb  = list(State([left_car_x, origin[1], left_car_yaw, 0, 0]).create_box().coords)[:-1]
        min_left_car_y = -min(rb[1], lb[1]) + MIN_DIST_TO_OBST
        left_car_y = random_gaussian_num(min_left_car_y+0.4, 0.2, min_left_car_y, min_left_car_y+0.8)
        obstacle_left = State([left_car_x, left_car_y, left_car_yaw, 0, 0]).create_box()

        # generate other parking vehicle
        for _ in range(n_non_critical_car):
            left_car_x -= (WIDTH + MIN_DIST_TO_OBST + random_uniform_num(min_dist_to_obst, max_dist_to_obst))
            left_car_y += random_gaussian_num(0, 0.05, -0.1, 0.1)
            left_car_yaw = random_gaussian_num(pi/2, pi/36, pi*5/12, pi*7/12)
            obstacle_left_ = State([left_car_x, left_car_y, left_car_yaw, 0, 0]).create_box()
            if random()<prob_non_critical_car:
                non_critical_vehicle.append(obstacle_left_)


    # generate obstacle on right
    dist_dest_to_left_obst = dest_box.distance(obstacle_left)
    min_dist_to_obst = max(min_lateral_space-dist_dest_to_left_obst, 0)+MIN_DIST_TO_OBST
    max_dist_to_obst = max(max_lateral_space-dist_dest_to_left_obst, 0)+MIN_DIST_TO_OBST
    if random()<prob_huge_obst: # generate simple obstacle
        right_obst_lf = get_rand_pos(*car_rf, -pi/12, pi/12, min_dist_to_obst, max_dist_to_obst)
        right_obst_lb = get_rand_pos(*car_rb, -pi/12, pi/12, min_dist_to_obst, max_dist_to_obst)
        obstacle_right = LinearRing((
            (origin[0]+bay_half_len, right_obst_lf[1]),
            (origin[0]+bay_half_len, origin[1]),
            right_obst_lb,
            right_obst_lf))
    else: # generate another vehicle as obstacle on right
        right_car_x = origin[0] + (WIDTH + random_uniform_num(min_dist_to_obst, max_dist_to_obst))
        right_car_yaw = random_gaussian_num(pi/2, pi/36, pi*5/12, pi*7/12)
        rb, _, _, lb  = list(State([right_car_x, origin[1], right_car_yaw, 0, 0]).create_box().coords)[:-1]
        min_right_car_y = -min(rb[1], lb[1]) + MIN_DIST_TO_OBST
        right_car_y = random_gaussian_num(min_right_car_y+0.4, 0.2, min_right_car_y, min_right_car_y+0.8)
        obstacle_right = State([right_car_x, right_car_y, right_car_yaw, 0, 0]).create_box()

        # generate other parking vehicle
        for _ in range(n_non_critical_car):
            right_car_x += (WIDTH + MIN_DIST_TO_OBST + random_uniform_num(min_dist_to_obst, max_dist_to_obst))
            right_car_y += random_gaussian_num(0, 0.05, -0.1, 0.1)
            right_car_yaw = random_gaussian_num(pi/2, pi/36, pi*5/12, pi*7/12)
            obstacle_right_ = State([right_car_x, right_car_y, right_car_yaw, 0, 0]).create_box()
            if random()<prob_non_critical_car:
                non_critical_vehicle.append(obstacle_right_)

    dist_dest_to_right_obst = dest_box.distance(obstacle_right)
    if dist_dest_to_right_obst+dist_dest_to_left_obst<min_lateral_space or \
        dist_dest_to_right_obst+dist_dest_to_left_obst>max_lateral_space or \
        dist_dest_to_left_obst<MIN_DIST_TO_OBST or \
        dist_dest_to_right_obst<MIN_DIST_TO_OBST:
        generate_success = False
    # check collision
    obstacles = [obstacle_back, obstacle_left, obstacle_right]
    obstacles.extend(non_critical_vehicle)
    for obst in obstacles:
        if obst.intersects(dest_box):
            generate_success = False

    # generate obstacles out of start range
    max_obstacle_y = max([np.max(np.array(obs.coords)[:,1]) for obs in obstacles])+MIN_DIST_TO_OBST
    other_obstcales = []
    if random()<0.2: # in this case only a wall will be generate
        other_obstcales = [LinearRing((
        (origin[0]-bay_half_len, bay_PARK_WALL_DIST+max_obstacle_y+MIN_DIST_TO_OBST),
        (origin[0]+bay_half_len, bay_PARK_WALL_DIST+max_obstacle_y+MIN_DIST_TO_OBST),
        (origin[0]+bay_half_len, bay_PARK_WALL_DIST+max_obstacle_y+MIN_DIST_TO_OBST+0.1),
        (origin[0]-bay_half_len, bay_PARK_WALL_DIST+max_obstacle_y+MIN_DIST_TO_OBST+0.1)))]
    else:
        other_obstacle_range = LinearRing((
        (origin[0]-bay_half_len, bay_PARK_WALL_DIST+max_obstacle_y),
        (origin[0]+bay_half_len, bay_PARK_WALL_DIST+max_obstacle_y),
        (origin[0]+bay_half_len, bay_PARK_WALL_DIST+max_obstacle_y+8),
        (origin[0]-bay_half_len, bay_PARK_WALL_DIST+max_obstacle_y+8)))
        valid_obst_x_range = (origin[0]-bay_half_len+2, origin[0]+bay_half_len-2)
        valid_obst_y_range = (bay_PARK_WALL_DIST+max_obstacle_y+2, bay_PARK_WALL_DIST+max_obstacle_y+6)
        for _ in range(n_obst):
            obs_x = random_uniform_num(*valid_obst_x_range)
            obs_y = random_uniform_num(*valid_obst_y_range)
            obs_yaw = random()*pi*2
            obs_coords = np.array(State([obs_x, obs_y, obs_yaw, 0, 0]).create_box().coords[:-1])
            obs = LinearRing(obs_coords+0.5*random(obs_coords.shape))
            if obs.intersects(other_obstacle_range):
                continue
            obst_invalid = False
            for other_obs in other_obstcales:
                if obs.intersects(other_obs):
                    obst_invalid = True
                    break
            if obst_invalid:
                continue
            other_obstcales.append(obs)

    # merge two kind of obstacles
    obstacles.extend(other_obstcales)


    # generate start position
    start_box_valid = False
    valid_start_x_range = (origin[0]-bay_half_len/2, origin[0]+bay_half_len/2)
    valid_start_y_range = (max_obstacle_y+1, bay_PARK_WALL_DIST+max_obstacle_y-1)
    while not start_box_valid:
        start_box_valid = True
        start_x = random_uniform_num(*valid_start_x_range)
        start_y = random_uniform_num(*valid_start_y_range)
        start_yaw = random_gaussian_num(0, pi/6, -pi/2, pi/2)
        start_yaw = start_yaw+pi if random()<0.5 else start_yaw
        start_box = State([start_x, start_y, start_yaw, 0, 0]).create_box()
        # check collision
        for obst in obstacles:
            if obst.intersects(start_box):
                if DEBUG:
                    print('start box intersect with obstacles, will retry to generate.')
                start_box_valid = False
        # check overlap with dest box
        if dest_box.intersects(start_box):
            if DEBUG:
                print('start box intersect with dest box, will retry to generate.')
            start_box_valid = False

    # randomly drop the obstacles
    for obs in obstacles:
        if random()<DROUP_OUT_OBST:
            obstacles.remove(obs)

    if DEBUG:
        ax.add_patch(plt.Polygon(xy=list(State([start_x, start_y, start_yaw, 0, 0]).create_box().coords), color='g'))
        print(generate_success)
        if generate_success:
            path = './log/figure/'
            num_files = len([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])
            fig = plt.gcf()
            fig.savefig(path+f'image_{num_files}.png')
        plt.clf()

    if generate_success:
        # save in files
        case_data = {
            'start': [start_x, start_y, start_yaw],
            'dest': [dest_x, dest_y, dest_yaw],
            'obstacles': obstacles
        }
        case_dir = f'{LOG_DIR}/{timestamp}/data'
        os.makedirs(case_dir, exist_ok=True)
        case_path = f'{case_dir}/case_{case_idx}.npz'
        save_case(case_data, case_path)
    else:
        # print(1)
        return random_generate(case_idx)

def save_case(case_data, filename):
    np.savez_compressed(
        filename,
        start=np.array(case_data['start']),
        dest=np.array(case_data['dest']),
        obstacles=[np.array(obs.coords) for obs in case_data['obstacles']]
    )

def load_case(filename):
    data = np.load(filename)
    case_data = {
        'start': data['start'].tolist(),
        'dest': data['dest'].tolist(),
    }
    case_data['obstacles'] = [
        LinearRing(obs_array) 
        for obs_array in data['obstacles']
    ]
    return case_data

def visual_case(case_data):
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.set_xlim(-20,20)
    ax.set_ylim(-20,20)
    plt.axis('off')

    dest_x, dest_y, dest_yaw = case_data['dest']
    obstacles = case_data['obstacles']
    start_x, start_y, start_yaw = case_data['start']
    car_rb, car_rf, car_lf, car_lb  = list(State([dest_x, dest_y, dest_yaw, 0, 0]).create_box().coords)[:-1]
    ax.add_patch(plt.Polygon(xy=list([car_rb, car_rf, car_lf, car_lb]), color='b'))

    for obs in obstacles:
        ax.add_patch(plt.Polygon(xy=list(obs.coords), color='gray'))

    ax.add_patch(plt.Polygon(xy=list(State([start_x, start_y, start_yaw, 0, 0]).create_box().coords), color='g'))

    # Add arrow to indicate vehicle direction at start position
    plot_arrow(ax, start_x, start_y, start_yaw)
    plot_arrow(ax, dest_x, dest_y, dest_yaw)
    
    path = './log/eval/figure/'
    os.makedirs(path, exist_ok=True)
    num_files = len([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])
    fig = plt.gcf()
    fig.savefig(path+f'image_{num_files}.png')
    plt.clf()

def plot_arrow(ax, x, y, yaw, length = 3, width = 2, facecolor = 'r', edgecolor = 'k'):
    dx = length * cos(yaw)
    dy = length * sin(yaw)
    ax.annotate('', xy=(x - dx, y - dy),
                xytext=(x, y),
                arrowprops=dict(arrowstyle='->', facecolor=facecolor, edgecolor=edgecolor, lw=width))

def obstacles_to_xy_lists(obstacles, resolution=XY_GRID_RESOLUTION):
    """
    将 LinearRing 障碍物列表转换为 x, y 坐标列表
    障碍物为实心多边形，内部所有栅格点都记录到ox和oy中

    参数:
        obstacles (list): [obstacle (LinearRing), ...]
        resolution (float): 栅格分辨率，用于填充多边形内部 [m]

    返回:
        ox (list): x坐标列表
        oy (list): y坐标列表
    """
    # 使用shapely判断点是否在多边形内部
    from shapely.geometry import Point, Polygon

    ox, oy = [], []

    for obstacle in obstacles:
        # 将LinearRing转换为Polygon，这样才能有内部区域
        polygon = Polygon(obstacle)

        # 获取边界框
        min_x, min_y = polygon.bounds[0], polygon.bounds[1]
        max_x, max_y = polygon.bounds[2], polygon.bounds[3]

        # 在边界框内进行栅格化
        x_grid = np.arange(min_x, max_x + resolution, resolution)
        y_grid = np.arange(min_y, max_y + resolution, resolution)

        # 对每个栅格点检查是否在多边形内部
        for x in x_grid:
            for y in y_grid:
                # 使用Polygon.contains()判断点是否在多边形内部
                if polygon.contains(Point(x, y)):
                    ox.append(x)
                    oy.append(y)

    return ox, oy

if __name__ == '__main__':
    """test random_generate() function, average time: 0.004s"""
    case_num = 1000
    gen_time = time.time()
    for i in range(case_num):
        random_generate(i)
    gen_time = time.time() - gen_time
    print(f'average time for generation: {gen_time / case_num:.4f} s')

    """test visual_case() function, average time: 0.3 s"""
    # case_path = 'log/eval/20251113_143214/data/case_0.npz'
    # case_data = load_case(case_path)
    # visual_time = time.time()
    # visual_case(case_data)
    # visual_time = time.time() - visual_time
    # print(f'average time for generation: {visual_time:.4f} s')
    
    """test obstacles_to_xy_lists() function"""
    case_path = 'log/eval/20251113_143214/data/case_0.npz'
    case_data = load_case(case_path)
    start, dest = case_data['start'], case_data['dest']
    obstacles = case_data['obstacles']
    ox, oy = obstacles_to_xy_lists(obstacles)
    