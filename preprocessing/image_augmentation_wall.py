import os
from PIL import Image
import numpy as np
import time
import random

if __name__ == "__main__":
    time_start = time.time()
    root_folder = "采光/new_win/"
    output_folder = "采光/new_extra/"
    # folders = os.listdir(root_folder)
    cnt = 0

    color = [(8, 251, 19), (249, 131, 5), (247, 248, 6), (7, 225, 252), (255, 0, 255), (249, 5, 5), (192, 255, 62),
             (132, 112, 255)]
    room_name = ["阳台", "客厅", "厨房", "卫生", "衣帽", "卧室", "书房", "储藏"]
    wall_expand = [6, 10, 16, 20]
    wall_name = ['a', 'b', 'c', 'd']

    folders = []
    for f in ['1', '2', '3', '4']:
        for folder in ['103']:
            folders.append(folder + '_' + f)

    # for folder in ['103']:
    for folder in folders:
        cnt += 1
        print(str(cnt) + ',' + folder)
        # if cnt != 17:
        #     continue

        # 读取识别结果
        room_file = root_folder + '/' + folder + '/' + 'room.csv'
        window_file = root_folder + '/' + folder + '/' + 'window.csv'
        door_file = root_folder + '/' + folder + '/' + 'door.csv'
        trans = 10

        file = open(door_file, 'r', encoding="gbk")  # 读取以utf-8
        context = file.read()  # 读取成str
        door_loc = context.split("\n")[0:-1]
        for i in range(len(door_loc)):
            door_loc[i] = door_loc[i].split(",")
            door_loc[i] = [float(i) * trans for i in door_loc[i]]
        file.close()

        # 数据增强，外墙延伸
        data_aug = 1
        dir_list = []

        for wall in wall_expand:
            new_dir = output_folder + folder + '_' + wall_name[data_aug - 1]
            dir_list.append(new_dir)
            data_aug += 1
            # 读取原始文件
            file = open(room_file, 'r', encoding="gbk")  # 读取以utf-8
            context = file.read()  # 读取成str
            room_loc = context.split("\n")[0:-1]
            for i in range(len(room_loc)):
                room_loc[i] = room_loc[i].split(",")[0:-1]
                room_loc[i] = [float(i) * trans for i in room_loc[i]]
            file.close()

            file = open(window_file, 'r', encoding="gbk")  # 读取以utf-8
            context = file.read()  # 读取成str
            window_loc = context.split("\n")[0:-1]
            for i in range(len(window_loc)):
                window_loc[i] = window_loc[i].split(",")
                window_loc[i] = [float(i) * trans for i in window_loc[i]]
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
            window_plus = np.ndarray((ny, nx))
            window_plus[:, :] = 0
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

            # 标记窗为1(window_plus)
            for window in window_loc:
                minx = round(min(window[0], window[2]))
                maxx = round(max(window[0], window[2]))
                miny = round(min(256 - window[1], 256 - window[3]))
                maxy = round(max(256 - window[1], 256 - window[3]))
                if miny == maxy:
                    continue
                window_plus[miny:maxy+1, minx:maxx+1] = 1

            # 外墙延伸(notation, result)
            for i in range(len(room_loc)):
                j = -1
                while j < int(len(room_loc[i]) / 2)-1:
                    x1 = round(room_loc[i][j*2])
                    x2 = round(room_loc[i][j*2+2])
                    y1 = round(256 - room_loc[i][j*2+1])
                    y2 = round(256 - room_loc[i][j*2+3])
                    minx = min(x1, x2)
                    maxx = max(x1, x2)
                    miny = min(y1, y2)
                    maxy = max(y1, y2)
                    if minx == maxx and miny != maxy:
                        j += 1
                        continue
                        x = room_loc[i][2 * j]
                        # 向左
                        if sum(sum(notation[miny:maxy+1, :minx])) == 0:
                            room_loc[i][2 * j] -= wall
                            room_loc[i][2 * j + 2] -= wall
                            notation[miny + 1:maxy, minx - wall + 1:minx + 1] = 1
                            result[miny + 1:maxy, minx - wall + 1:minx + 1] = 1

                            # change window
                            for k in range(len(window_loc)):
                                if window_loc[k][0] == window_loc[k][2] == x:
                                    if max(window_loc[k][1], window_loc[k][3]) <= max(room_loc[i][2*j+1], room_loc[i][2*j+3]) \
                                            and min(window_loc[k][1], window_loc[k][3]) >= min(room_loc[i][2*j+1], room_loc[i][2*j+3]):
                                        window_loc[k][0] -= wall
                                        window_loc[k][2] -= wall
                        # 向右
                        elif sum(sum(notation[miny:maxy+1, minx:])) == 0:
                            room_loc[i][2 * j] += wall
                            room_loc[i][2 * (j+1)] += wall
                            notation[miny + 1:maxy, minx:minx + wall] = 1
                            result[miny + 1:maxy, minx:minx + wall] = 1

                            # change window on the wall
                            for k in range(len(window_loc)):
                                if window_loc[k][0] == window_loc[k][2] == x:
                                    if max(window_loc[k][1], window_loc[k][3]) <= max(room_loc[i][2*j+1], room_loc[i][2*j+3]) \
                                            and min(window_loc[k][1], window_loc[k][3]) >= min(room_loc[i][2*j+1], room_loc[i][2*j+3]):
                                        window_loc[k][0] += wall
                                        window_loc[k][2] += wall
                    elif miny == maxy and minx != maxx:
                        if maxx - minx < 5:
                            j += 1
                            continue
                        y = room_loc[i][2 * j + 1]
                        # 向上
                        if sum(sum(notation[:miny, minx:maxx+1])) == 0:
                            # 判断是否遮挡窗
                            if sum(sum(window_plus[miny - wall + 1:miny + 1, :minx + 1])) > 0:
                                win_x = minx + 1
                                win_break = False
                                while win_x > 0:
                                    win_x -= 1
                                    if sum(window_plus[miny - wall + 1:miny + 1, win_x]) > 0:
                                        if sum(sum(notation[miny - wall + 1:miny + 1, :win_x])) > 0:
                                            print(['取消延伸，遮挡窗:', i, j])
                                            win_break = True
                                            break
                                if win_break:
                                    j += 1
                                    continue
                            if sum(sum(window_plus[miny - wall + 1:miny + 1, maxx:])) > 0:
                                win_x = maxx - 1
                                win_break = False
                                while win_x < nx - 1:
                                    win_x += 1
                                    if sum(window_plus[miny - wall + 1:miny + 1, win_x]) > 0:
                                        if sum(sum(notation[miny - wall + 1:miny + 1, win_x:])) > 0:
                                            print(['取消延伸，遮挡窗:', i, j])
                                            win_break = True
                                            break
                                if win_break:
                                    j += 1
                                    continue
                            # 填补空线
                            if room_loc[i][2 * j - 1] > room_loc[i][2 * j + 1]:
                                minx -= 1
                            if room_loc[i][2 * j + 5 - len(room_loc[i])] > room_loc[i][2 * j + 1]:
                                maxx += 1
                            room_loc[i][2 * j + 1] += wall
                            room_loc[i][2 * j + 3] += wall
                            notation[miny - wall + 1:miny + 1, minx + 1:maxx] = 1
                            result[miny - wall + 1:miny + 1, minx + 1:maxx] = 1
                            # 坐标去重
                            # if round(room_loc[i][2 * j + 2]) == round(room_loc[i][2 * j + 4 - len(room_loc[i])]) and \
                            #         round(room_loc[i][2 * j + 3]) == round(room_loc[i][2 * j + 5 - len(room_loc[i])]):
                            #     room_loc[i][2 * j + 1] = room_loc[i][2 * j + 5 - len(room_loc[i])]
                            #     del room_loc[i][2*j+2:2*j+4]

                            # change window on the wall
                            for k in range(len(window_loc)):
                                if window_loc[k][1] == window_loc[k][3] == y:
                                    if max(window_loc[k][0], window_loc[k][2]) <= max(room_loc[i][2*j], room_loc[i][2*j+2]) \
                                            and min(window_loc[k][0], window_loc[k][2]) >= min(room_loc[i][2*j], room_loc[i][2*j+2]):
                                        window_loc[k][1] += wall
                                        window_loc[k][3] += wall
                        # 向下
                        elif sum(sum(notation[miny:, minx:maxx + 1])) == 0:
                            # 判断是否遮挡窗
                            if sum(sum(window_plus[miny:miny + wall, :minx + 1])) > 0:
                                win_x = minx + 1
                                win_break = False
                                while win_x > 0:
                                    win_x -= 1
                                    if sum(window_plus[miny:miny + wall, win_x]) > 0:
                                        if sum(sum(notation[miny:miny + wall, :win_x])) > 0:
                                            print(['取消延伸，遮挡窗:', i, j])
                                            win_break = True
                                            break
                                if win_break:
                                    j += 1
                                    continue
                            if sum(sum(window_plus[miny:miny + wall, maxx:])) > 0:
                                win_x = maxx - 1
                                win_break = False
                                while win_x < nx - 1:
                                    win_x += 1
                                    if sum(window_plus[miny:miny + wall, win_x]) > 0:
                                        if sum(sum(notation[miny:miny + wall, win_x:])) > 0:
                                            print(['取消延伸，遮挡窗:', i, j])
                                            win_break = True
                                            break
                                if win_break:
                                    j += 1
                                    continue
                            # 填补空线
                            if room_loc[i][2 * j - 1] < room_loc[i][2 * j + 1]:
                                maxx += 1
                            if room_loc[i][2 * j + 5 - len(room_loc[i])] < room_loc[i][2 * j + 1]:
                                minx -= 1
                            room_loc[i][2 * j + 1] -= wall
                            room_loc[i][2 * j + 3] -= wall
                            notation[miny:miny + wall, minx + 1:maxx] = 1
                            result[miny:miny + wall, minx + 1:maxx] = 1
                            # 坐标去重
                            # if round(room_loc[i][2 * j + 2]) == round(room_loc[i][2 * j + 4 - len(room_loc[i])]) and \
                            #         round(room_loc[i][2 * j + 3]) == round(room_loc[i][2 * j + 5 - len(room_loc[i])]):
                            #     room_loc[i][2 * j + 1] = room_loc[i][2 * j + 5 - len(room_loc[i])]
                            #     del room_loc[i][2 * j + 2:2 * j + 4]

                            # change window on the wall
                            for k in range(len(window_loc)):
                                if window_loc[k][1] == window_loc[k][3] == y:
                                    if max(window_loc[k][0], window_loc[k][2]) <= max(room_loc[i][2*j], room_loc[i][2*j+2]) \
                                            and min(window_loc[k][0], window_loc[k][2]) >= min(room_loc[i][2*j], room_loc[i][2*j+2]):
                                        window_loc[k][1] -= wall
                                        window_loc[k][3] -= wall
                        j += 1
                    elif minx == maxx and miny == maxy:
                        j += 1
                        print("点重合")
                    else:
                        j += 1
                        print("error! 起步非正交")
                        print([minx, maxx, miny, maxy])
                        # break
            # save to new folder
            if not os.path.exists(new_dir):
                os.makedirs(new_dir)
            win_file_name = new_dir + '/window.csv'
            win_file = open(win_file_name, 'w')
            for opening in window_loc:
                win_file.write(str(opening[0] / trans) + ',' + str(opening[1] / trans) + ',' +
                               str(opening[2] / trans) + ',' + str(opening[3] / trans) + '\n')
            win_file.close()

            new_room_file = new_dir + '/room.csv'
            new_room = open(new_room_file, 'w')
            for room in room_loc:
                for r in room:
                    new_room.write(str(r / trans) + ',')
                new_room.write('\n')
            new_room.close()

            # 标记窗为0(mask)
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
            # mask = 100 - mask
            # Image.fromarray(mask * 2.55).show()
            a = mask.max()
            mask[mask > 100] = 100
            b = mask.max()

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

        # 另存door
        file = open(door_file, 'r', encoding="gbk")  # 读取以utf-8
        context = file.read()  # 读取成str
        for new_dir in dir_list:
            filename_w = new_dir + '/door.csv'
            w = open(filename_w, 'w')
            for line in context:
                w.write(line)
            w.close()
        file.close()