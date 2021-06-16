import os
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
import pandas as pd
import time

if __name__ == "__main__":
    # img_path = '采光'
    # light_path = 'light'
    img_path = '采光/new'
    light_path = 'light/new'
    lux_path = 'lux/'
    folders = os.listdir(light_path)
    max_lux = 2000

    x_hist = []

    # 自定义色带
    color_map = np.ndarray((1100, 3))
    color_num = 1100-1
    cnt = 0
    while cnt < 236:
        color_map[cnt, :] = (255, cnt, 0)
        cnt += 1
    while cnt < 256:
        trans = 255 - int((cnt - 234) / 2)
        color_map[cnt, :] = (trans, cnt, 0)
        cnt += 1
    while cnt < 494:
        color_map[cnt, :] = (500 - cnt, 255, 0)
        cnt += 1
    while cnt < 508:
        trans = int((507 - cnt) / 2)
        color_map[cnt, :] = (trans, 255, cnt - 493)
        cnt += 1
    while cnt < 745:
        color_map[cnt, :] = (0, 255, cnt - 493)
        cnt += 1
    while cnt < 750:
        color_map[cnt, :] = (0, 1000 - cnt, cnt - 494)
        cnt += 1
    while cnt < 986:
        color_map[cnt, :] = (0, 1000 - cnt, 255)
        cnt += 1
    while cnt < 1000:
        trans = int((cnt - 984) / 2)
        color_map[cnt, :] = (trans, 1000 - cnt, 255)
        cnt += 1
    while cnt < color_num + 1:
        color_map[cnt, :] = (cnt - 992, 0, 255)
        cnt += 1

    # 内置色带
    # colors = np.ndarray((256, 3))
    # for i in range(256):
    #     for j in range(3):
    #         colors[i, j] = int(255 * cm.hsv(i)[j])

    # for folder in ['103']:
    # for folder in folders[: 23]:
    for folder in folders:
        print(folder)
        # 读取文件
        img_file = img_path + '/' + folder + '/' + 'result.jpg'
        img = Image.open(img_file)
        if len(np.array(img).shape) == 2:
            mark = np.array(img)
        else:
            mark = np.array(img)[:, :, 0]
        result = np.ndarray((mark.shape[0], mark.shape[1], 3))
        result[:, :] = (0, 0, 0)
        result_grey = np.ndarray((mark.shape[0], mark.shape[1]))
        result_grey[:, :] = 0
        result_lux = np.ndarray((mark.shape[0], mark.shape[1]))
        result_lux[:, :] = -1

        room_file = img_path + '/' + folder + '/' + 'room.csv'
        file = open(room_file, 'r', encoding="gbk")  # 读取以utf-8
        context = file.read()  # 读取成str
        room_loc = context.split("\n")[0:-1]
        trans = 10
        for i in range(len(room_loc)):
            room_loc[i] = room_loc[i].split(",")[0:-1]
            room_loc[i] = [float(i) * trans for i in room_loc[i]]
        file.close()

        if len(os.listdir(light_path + '/' + folder)) != len(room_loc):
            print(folder + '! not match!')

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

            # 坐标定位
            room_x = room_loc[i][::2]
            room_y = room_loc[i][1::2]
            room_y = [256 - i for i in room_y]
            minx = round(min(room_x))
            maxx = round(max(room_x))
            miny = round(min(room_y))
            maxy = round(max(room_y))

            # 对应每个房间
            if maxy + 1 - miny == len(light):
                maxy += 1
            elif maxy + 2 - miny == len(light):
                light = light[: -1]
            elif maxy - 1 - miny == len(light):
                miny += 1
            elif maxy - miny != len(light):
                print(['y: ', maxy + 1 - miny - len(light), len(light)])
                continue

            if maxx + 1 - minx == len(light[0]):
                maxx += 1
            elif maxx - 1 - minx == len(light[0]):
                minx += 1
            elif maxx - minx != len(light[0]):
                print(['x: ', maxx + 1 - minx - len(light[0]), len(light[0])])
                continue

            for y in range(miny, maxy):
                for x in range(minx, maxx):
                    # 彩色映射
                    if light[miny - y - 1][x - minx] == 0:
                        mark[miny - y - 1, x - minx] = 0
                        # mark[miny - y - 1, x - minx, :] = (0, 0, 0)
                        continue

                    # 自定义色带
                    # s = max(1 - light[miny - y - 1][x - minx] / 2000, 0)
                    # color = int(color_num * s)
                    # result[y, x, :] = color_map[color]

                    # 内置色带
                    # s = max(1 - light[miny - y - 1][x - minx] / 2000, 0)
                    # color = cm.hsv(int(255 * s))
                    # for j in range(3):
                    #     result[y, x, j] = int(255 * color[j])

                    # 黑白映射
                    s = int(255 * min(light[miny - y - 1][x - minx] / max_lux, 1))
                    # result[y, x, :] = (s, s, s)
                    result_grey[y, x] = s
                    result_lux[y, x] = light[miny - y - 1][x - minx]

                # 数据统计
                # for ill in light[miny - y - 1]:
                #     if ill > 0:
                #         x_hist.append(ill)

        # img_result = Image.fromarray(np.uint8(result))
        # img_result.show()
        # img_result.save(img_path + '/' + folder + '/light.jpg')
        # img_mark = Image.fromarray(mark)
        # img_mark.show()
        # img_mark.save(img_path + '/' + folder + '/input.jpg')

        """图片拼合"""
        # 三通道版本
        # img_all = np.ndarray((256, 512, 3))
        # img_all[:, :32] = (0, 0, 0)
        # img_all[:, 32:224, 0] = mark
        # img_all[:, 32:224, 1] = mark
        # img_all[:, 32:224, 2] = mark
        # img_all[:, 224:288] = (0, 0, 0)
        # img_all[:, 288:480] = result
        # img_all[:, 480:] = (0, 0, 0)
        # img_save = Image.fromarray(np.uint8(img_all))

        # 单通道版本
        img_all = np.ndarray((256, 512))
        img_all[:, :32] = 0
        img_all[:, 32:224] = mark
        img_all[:, 224:288] = 0
        img_all[:, 288:480] = result_grey
        img_all[:, 480:] = 0
        img_save = Image.fromarray(np.uint8(img_all))
        # img_save.show()
        # img_save.save('datasets_output/grey_png/' + folder + '.png')
        save_lux_path = lux_path + folder + '.txt'
        np.savetxt(save_lux_path, result_lux, fmt='%.4e')

        """绘制照度分布直方图"""
        # plt.hist(x_hist, bins=200)
        # plt.xlabel('Illuminance(lux)')
        # plt.ylabel('Frequency')
        # plt.title('Hist of illuminance')
        # plt.show()
