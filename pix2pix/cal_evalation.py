"""
通过.ill文件，计算评价指标：采光均匀度、照度均值、采光满足率（>300lx）
可用于CNN回归的single value
"""

import os
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
import pandas as pd
import time

if __name__ == "__main__":
    # img_path = '采光'
    light_path = 'regular_result'
    # img_path = '采光/new'
    # light_path = 'light/new'
    evaluation_path = 'regular_evaluation/'
    folders = os.listdir(light_path)

    cnt = 0
    # for folder in folders[: 23]:
    for folder in folders:
        cnt += 1
        print([cnt, folder])
        # 读取文件
        room_file = light_path + '/' + folder + '/' + 'room.csv'
        file = open(room_file, 'r', encoding="gbk")  # 读取以utf-8
        context = file.read()  # 读取成str
        room_loc = context.split("\n")[0:-1]
        trans = 10
        for i in range(len(room_loc)):
            room_loc[i] = room_loc[i].split(",")[0:-1]
            room_loc[i] = [float(i) * trans for i in room_loc[i]]
        file.close()

        # if len(os.listdir(light_path + '/' + folder)) != len(room_loc) + 3:
        #     print(folder + '! not match!')

        min_lux = 2000
        all_num = 0
        sum_lux = 0
        satisfy_num = 0

        for i in range(len(room_loc)):
            light_file = light_path + '/' + folder + '/' + str(i+1) + '.csv'
            # df = pd.read_csv(loc_file)
            file = open(light_file, 'r', encoding="gbk")  # 读取以utf-8
            context = file.read()  # 读取成str
            light = context.split("\n")[0:-1]
            for j in range(len(light)):
                light[j] = light[j].split(",")[0:-1]
                light[j] = [float(k) for k in light[j]]
            file.close()

            # 数据统计
            light_array = np.array(light)
            min_room = np.min(light_array[light_array > 0])
            if min_room < min_lux:
                min_lux = min_room
            a2 = sum(sum(light_array > 0))
            all_num += a2
            a1 = sum(sum(light_array))
            sum_lux += a1
            a3 = sum(sum(light_array > 300))
            satisfy_num += a3
            # print(min_lux, satisfy_num, all_num)

        mean_lux = sum_lux / all_num
        v_uniform = min_lux / mean_lux
        v_satisfy = satisfy_num / all_num

        print(v_uniform, mean_lux, v_satisfy)
        v_array = np.array([v_uniform, mean_lux, v_satisfy])

        """结果存储"""
        save_evaluation_path = evaluation_path + folder + '.npy'
        np.save(save_evaluation_path, v_array)

