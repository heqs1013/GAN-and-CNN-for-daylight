import os
from PIL import Image
import numpy as np
import time
import math
import random

if __name__ == "__main__":
    time_start = time.time()
    root_folder = "采光"
    folders = os.listdir(root_folder)
    cnt = 0

    color = [(8, 251, 19), (249, 131, 5), (247, 248, 6), (7, 225, 252), (255, 0, 255), (249, 5, 5), (192, 255, 62),
             (132, 112, 255)]
    room_name = ["阳台", "客厅", "厨房", "卫生", "衣帽", "卧室", "书房", "储藏"]
    window_narrow = [0.5, 0.6, 0.7, 0.8]

    if 1:
        folder = '100'

    # for folder in folders:
        cnt += 1
        # if cnt < 82:
        #     continue
        # print(cnt)
        print(str(cnt) + ',' + folder)

        # 读取识别结果
        room_file = root_folder + '/' + folder + '/' + 'room.csv'
        window_file = root_folder + '/' + folder + '/' + 'window.csv'
        door_file = root_folder + '/' + folder + '/' + 'door.csv'
        trans = 10

        file = open(room_file, 'r', encoding="gbk")  # 读取以utf-8
        context = file.read()  # 读取成str
        room_loc = context.split("\n")[0:-1]
        for i in range(len(room_loc)):
            room_loc[i] = room_loc[i].split(",")[0:-1]
            room_loc[i] = [float(i) * trans for i in room_loc[i]]
        file.close()

        file = open(door_file, 'r', encoding="gbk")  # 读取以utf-8
        context = file.read()  # 读取成str
        door_loc = context.split("\n")[0:-1]
        for i in range(len(door_loc)):
            door_loc[i] = door_loc[i].split(",")
            door_loc[i] = [float(i) * trans for i in door_loc[i]]
        file.close()

        # 生成图形标记
        """
        notation: 0表示不透光结构或外部,1表示房间内或开门处
        room_bound：单个房间边界
        mask：表示第n步扩散边界
        result：房间内部，到窗的曼哈顿距离
        """
        nx = 192
        ny = 256
        notation = np.ndarray((ny, nx))
        notation[:, :] = 0
        room_bound = np.ndarray((ny, nx))
        room_bound[:, :] = 0
        mask = np.ndarray((ny, nx))
        mask[:, :] = -1
        result = np.ndarray((ny, nx))
        result[:, :] = 0

        # 标记房间内部为1(notation, result)
        for room in room_loc:
            room_bound[:, :] = 0
            room_x = room[::2]
            room_y = room[1::2]
            room_y = [256-i for i in room_y]
            if len(room_x) != len(room_y):
                print("room_x != room_y")
            # 边界
            for i in range(-1, len(room_x)-1):
                x1 = round(room_x[i])
                x2 = round(room_x[i+1])
                y1 = round(room_y[i])
                y2 = round(room_y[i+1])
                minx = min(x1, x2)
                maxx = max(x1, x2)
                miny = min(y1, y2)
                maxy = max(y1, y2)
                room_bound[miny:maxy+1, minx:maxx+1] = 1
            # 内部
            minx = round(min(room_x))
            maxx = round(max(room_x))
            miny = round(min(room_y))
            maxy = round(max(room_y))
            for x in range(minx+1, maxx):
                for y in range(miny+1, maxy):
                    if room_bound[y, x] == 0 and sum(room_bound[y, minx:x]) > 0 and sum(room_bound[y, x:maxx+1]) > 0 \
                            and sum(room_bound[miny:y, x]) > 0 and sum(room_bound[y:maxy+1, x]) > 0:
                        notation[y, x] = 1
                        result[y, x] = 1

        # Image.fromarray(notation * 255).show()

        # 标记门为1(notation)
        for door in door_loc:
            if door[0] == door[2]:
                minx = round(min(door[0], door[0] + door[4]))-1
                maxx = round(max(door[0], door[0] + door[4]))+1
            else:
                minx = round(min(door[0], door[2]))+1
                maxx = round(max(door[0], door[2]))-1
            if door[1] == door[3]:
                miny = round(min(256 - door[1], 256 - door[1] - door[4]))-1
                maxy = round(max(256 - door[1], 256 - door[1] - door[4]))+1
            else:
                miny = round(min(256 - door[1], 256 - door[3]))+1
                maxy = round(max(256 - door[1], 256 - door[3]))-1
            notation[miny:maxy+1, minx:maxx+1] = 1

        # 数据增强，大于0.5m的窗宽度减小
        data_aug = 1
        dir_list = []

        for win_r in window_narrow:
            # read original
            file = open(window_file, 'r', encoding="gbk")  # 读取以utf-8
            context = file.read()  # 读取成str
            window_loc = context.split("\n")[0:-1]
            for i in range(len(window_loc)):
                window_loc[i] = window_loc[i].split(",")
                window_loc[i] = [float(i) * trans for i in window_loc[i]]
            file.close()

            # change
            for i in range(len(window_loc)):
                if max(abs(window_loc[i][0] - window_loc[i][2]), abs(window_loc[i][1] - window_loc[i][3])) > 5:
                    if window_loc[i][0] == window_loc[i][2]:
                        mid = (window_loc[i][1] + window_loc[i][3]) / 2
                        window_loc[i][1] = win_r * window_loc[i][1] + (1 - win_r) * mid
                        window_loc[i][3] = win_r * window_loc[i][3] + (1 - win_r) * mid
                    else:
                        mid = (window_loc[i][0] + window_loc[i][2]) / 2
                        window_loc[i][0] = win_r * window_loc[i][0] + (1 - win_r) * mid
                        window_loc[i][2] = win_r * window_loc[i][2] + (1 - win_r) * mid

            # save to new folder
            new_dir = root_folder + '/new_win/' + folder + '_' + str(data_aug)
            dir_list.append(new_dir)
            data_aug += 1

            if not os.path.exists(new_dir):
                os.makedirs(new_dir)
            win_file_name = new_dir + '/window.csv'
            win_file = open(win_file_name, 'w')
            for window in window_loc:
                win_file.write(str(window[0] / trans) + ',' + str(window[1] / trans) + ',' +
                               str(window[2] / trans) + ',' + str(window[3] / trans) + '\n')
            win_file.close()

            # 标记窗为0(mask)
            mask[:, :] = -1
            for window in window_loc:
                minx = round(min(window[0], window[2]))
                maxx = round(max(window[0], window[2]))
                miny = round(min(256 - window[1], 256 - window[3]))
                maxy = round(max(256 - window[1], 256 - window[3]))
                mask[miny:maxy+1, minx:maxx+1] = 0
            #     notation[miny:maxy+1, minx:maxx+1] = 0.5
            #
            # Image.fromarray(notation * 255).show()

            # 距离扩散标记（mask）
            num = 1
            step = 0
            while num > 0:
                idx_y = np.where(mask == step)[0]
                idx_x = np.where(mask == step)[1]
                step += 1
                num = 0
                for j in range(len(idx_y)):
                    x = idx_x[j]
                    y = idx_y[j]

                    if notation[y - 1, x] > 0 and mask[y - 1, x] < 0:
                        mask[y - 1, x] = step
                        num += 1
                    if notation[y + 1, x] > 0 and mask[y + 1, x] < 0:
                        mask[y + 1, x] = step
                        num += 1
                    if notation[y, x - 1] > 0 and mask[y, x - 1] < 0:
                        mask[y, x - 1] = step
                        num += 1
                    if notation[y, x + 1] > 0 and mask[y, x + 1] < 0:
                        mask[y, x + 1] = step
                        num += 1

            # for y in range(mask.shape[0]):
            #     for x in range(mask.shape[1]):
            #         if mask[y, x] > 0:
            #             mask[y, x] = 3 - math.log(mask[y, x])
            # Image.fromarray(mask * 2.55).show()

            # 存储结果（result）和蒙版
            save_mask = np.ndarray((ny, nx, 3))
            save_mask[:, :, 0] = result * 255
            save_mask[:, :, 1] = result * 255
            save_mask[:, :, 2] = result * 255
            img = Image.fromarray(np.uint8(save_mask))
            # img.show()
            img.save(new_dir + '/result_mask.jpg')

            save_result = result * mask * 2.55
            img = Image.fromarray(np.uint8(save_result))
            # img.show()
            img.save(new_dir + '/result.jpg')

        # 另存room和door
        file = open(room_file, 'r', encoding="gbk")  # 读取以utf-8
        context = file.read()  # 读取成str
        for new_dir in dir_list:
            filename_w = new_dir + '/room.csv'
            w = open(filename_w, 'w')
            for line in context:
                w.write(line)
            w.close()
        file.close()

        file = open(door_file, 'r', encoding="gbk")  # 读取以utf-8
        context = file.read()
        for new_dir in dir_list:
            filename_w = new_dir + '/door.csv'
            w = open(filename_w, 'w')
            for line in context:
                w.write(line)
            w.close()
        file.close()