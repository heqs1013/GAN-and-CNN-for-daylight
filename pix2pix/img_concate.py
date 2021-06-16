import os
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy

if __name__ == "__main__":
    path_test = 'test_200/'
    path_truth = 'datasets/facades_rgb/test/'
    # mark_path = 'datasets_output/grey_png/test/'
    path_lux = 'lux/test/'
    path_rgb = 'datasets_output/compare/'
    path_save_fig = 'datasets_output/figs6/'
    path_save_lux = 'lux/test_gt/'
    folders_1 = os.listdir(path_test)
    folders_2 = os.listdir(path_truth)
    # folders_3 = os.listdir(mark_path)
    folders_4 = os.listdir(path_lux)
    folders_5 = os.listdir(path_rgb)

    size_map = 1100
    max_lux = 2000
    img_shape = (256, 256)
    r_lux2colormap = size_map / max_lux
    r_lux2grey = 255 / max_lux
    r_grey2lux = max_lux / 255

    # 自定义色带
    color_map = np.ndarray((size_map, 3))
    color_num = size_map - 1
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
    # color_map_list = []
    # for x in color_map:
    #     color_map_list.append(x[0] * 1000000 + x[1] * 1000 + x[2])

    # img_colormap = np.ndarray((1100, 100, 3))
    # for i in range(1100):
    #     img_colormap[i, :] = color_map[i]
    # colormap_show = Image.fromarray(np.uint8(img_colormap))
    # plt.imshow(colormap_show)
    # plt.show()
    # colormap_show.show()

    error_mean = np.ndarray((len(folders_1), 1))
    for i in [43, 6, 73]:
    # for i in range(len(folders_1)):
        file_test = folders_1[i]
        # file_truth = folders_2[i]
        if i < 9:
            file_truth = '0' + file_test[7:9] + '.jpg'
            file_rgb = '0' + file_test[7:9] + '.jpg'
        else:
            file_truth = '00' + file_test[7:9] + '.jpg'
            file_rgb = '00' + file_test[7:9] + '.jpg'
        # img_mark = folders_3[i]
        file_lux = folders_4[i]
        # file_rgb = folders_5[i]

        print(file_truth, file_test)

        """灰度图像误差"""
        # img_origin = cv2.imread(truth_path + img_truth)
        # img_truth = img_origin[:, 256:, :]
        # img_predict = cv2.imread(test_path + img_test)
        # diff_cv = cv2.absdiff(img_truth, img_predict)

        # plt.subplot(1, 3, 1)
        # plt.title('ground_truth')
        # plt.imshow(img_truth)

        # plt.subplot(1, 3, 2)
        # plt.title('predict')
        # plt.imshow(img_predict)

        # plt.subplot(1, 3, 3)
        # plt.title('error')
        # plt.imshow(diff_cv)

        # error_mean[i] = diff_cv.mean()
        # print(diff_cv.mean())
        # plt.show()

        """黑白转为照度值"""
        """映射为伪彩色"""
        img1 = Image.open(path_truth + file_truth)
        """格式观察，需pip install scipy==1.2.1"""
        # a = scipy.misc.imread(img1)
        img_truth_color = np.array(img1)[:, 256:]
        img_truth_grey = np.array(img1)[:, :256]
        img2 = Image.open(path_test + file_test)
        # img_test = np.array(img2)
        img_test = np.array(img2.convert('L'))
        # img3 = Image.open((mark_path + img_mark))
        # img_mid = np.array(img3)[:, 256:]
        value_lux = np.ndarray((256, 256))
        value_lux[:, :] = -1
        value_lux_predict = np.ndarray((256, 256))
        value_lux_predict[:, :] = -1
        lux_array = np.loadtxt(path_lux + file_lux)
        value_lux[:, 32:224] = lux_array
        img4_cv = cv2.imread(path_rgb + file_rgb)
        img_rgb_cv = img4_cv[:, 512:]
        img4 = Image.open(path_rgb + file_rgb)
        img_rgb = np.array(img4)[:, 512:]
        img5 = cv2.imread(path_truth + file_truth)
        img_truth = img5[:, 256:]
        # a = img_right[:, :, 1] - img_right[:, :, 0].max()
        # b = img_right[:, :, 1] - img_right[:, :, 2].max()

        origin_color = np.ndarray((img_shape[0], img_shape[1], 3))
        predict_color = np.ndarray((img_shape[0], img_shape[1], 3))
        diff_color = np.ndarray((img_shape[0], img_shape[1], 3))
        diff_color_rgb = np.ndarray((img_shape[0], img_shape[1], 3))

        for x in range(img_shape[1]):
            for y in range(img_shape[0]):
                if value_lux[y, x] == -1:
                # if img_mid[y, x] < 10:
                    origin_color[y, x, :] = (0, 0, 0)
                    predict_color[y, x, :] = (0, 0, 0)
                    diff_color[y, x, :] = (0, 0, 0)
                    value_lux_predict[y, x] = -1
                else:
                    predict_color[y, x, :] = color_map[color_num - int(img_test[y, x] * color_num / 255)]
                    value_lux_predict[y, x] = img_test[y, x] * r_grey2lux
                    # diff_color[y, x, :] = color_map[color_num - min(int(abs(
                    #     (value_lux[y, x] - img_test[y, x] * r_grey2lux) * r_lux2colormap)), color_num)]
                    diff_color[y, x] = abs(img_truth_color[y, x] - predict_color[y, x])

        """画图"""
        # imshow直接处理矩阵颜色有偏差，奇怪
        img_origin_color = Image.fromarray(np.uint8(img_truth_color))
        img_origin_grey = Image.fromarray(np.uint8(img_truth_grey))
        plt.subplot(2, 3, 1)
        plt.title('ground_truth')
        plt.imshow(img_origin_color)
        plt.subplot(2, 3, 4)
        plt.title('ground_truth')
        plt.imshow(img_origin_grey)

        img_predict = Image.fromarray(np.uint8(predict_color))
        plt.subplot(2, 3, 2)
        plt.title('predict(grey)')
        plt.imshow(img_predict)

        img_diff = Image.fromarray(np.uint8(diff_color))
        plt.subplot(2, 3, 3)
        plt.title('error(grey)')
        plt.imshow(img_diff)

        img_predict_rgb = Image.fromarray(np.uint8(img_rgb))
        plt.subplot(2, 3, 5)
        plt.title('predict(rgb)')
        plt.imshow(img_predict_rgb)

        # img_diff_rgb = Image.fromarray(np.uint8(diff_color_rgb))
        img_diff_rgb = cv2.absdiff(img_truth, img_rgb_cv)
        plt.subplot(2, 3, 6)
        plt.title('error(rgb)')
        plt.imshow(img_diff_rgb)

        plt.show()
        # plt.savefig(path_save_fig + file_truth + '.png')
        # np.savetxt(path_save_lux + file_lux + '.txt', value_lux_predict, fmt='%.4e')

        # 拼合
        # img_all = np.ndarray((256, 768, 3))
        # img_all[:, :512, :] = img_left
        # img_all[:, 512:, :] = img_right
        # img_save = Image.fromarray(np.uint8(img_all))
        # img_save.save('datasets_output/compare/' + img_test)

        """色彩转为照度值"""
        # nx = img_truth.shape[1]
        # ny = img_truth.shape[0]
        # light_value = np.ndarray((ny, nx))
        # light_truth = np.ndarray((ny, nx))

        # num_error = 0
        # for x in range(nx):
        #     for y in range(ny):
        #         pix = img_left[y][x]
        #         pix_truth = img_right[y][x]
        #         if pix_truth[0] == 0 and pix_truth[1] == 0 and pix_truth[2] == 0:
        #             # 非模拟区域，照度值记为-1
        #             light_truth[y][x] = -1
        #         elif color_map_list.count(pix_truth[0] * 1000000 + pix_truth[1] * 1000 + pix_truth[2]):
        #             # 有对应值
        #             light_truth[y][x] = color_map_list.index(pix_truth[0] * 1000000 + pix_truth[1] * 1000 + pix_truth[2])
        #         else:
        #             # 没有对应值，匹配最近
        #             min_diff = int(pix_truth[0]) + int(pix_truth[1]) + int(pix_truth[2])
        #             min_idx = -1
        #             for j in range(len(color_map)):
        #                 c = color_map[j]
        #                 diff = abs(pix_truth[0] - c[0]) + abs(pix_truth[1] - c[1]) + abs(pix_truth[2] - c[2])
        #                 if diff < min_diff:
        #                     min_diff = diff
        #                     min_idx = j
        #             light_truth[y][x] = min_idx
        #             # print('error! ground truth无法对应照度值')
        #             # print(pix_truth, light_truth[y][x])
        #             num_error += 1
        #
        #         if pix[0] == 0 and pix[1] == 0 and pix[2] == 0:
        #             # 非模拟区域，照度值记为-1
        #             light_value[y][x] = -1
        #         elif color_map_list.count(pix[0] * 1000000 + pix[1] * 1000 + pix[2]):
        #             # 有对应值
        #             light_value[y][x] = color_map_list.index(pix[0] * 1000000 + pix[1] * 1000 + pix[2])
        #         else:
        #             # 没有对应值，匹配最近
        #             min_diff = abs(pix[0] - color_map[0][0]) + abs(pix[1] - color_map[0][1]) + abs(pix[2] - color_map[0][2])
        #             min_idx = -1
        #             for j in range(len(color_map)):
        #                 c = color_map[j]
        #                 diff = abs(pix[0] - c[0]) + abs(pix[1] - c[1]) + abs(pix[2] - c[2])
        #                 if diff < min_diff:
        #                     min_diff = diff
        #                     min_idx = j
        #             light_value[y][x] = min_idx
        #
        # # 统计误差
        # light_diff = light_value - light_truth
        # max_diff = max(light_diff)
        # mean_diff = light_diff.mean
        # print(['预测误差：', mean_diff, max_diff])

    np.savetxt('error_mean.csv', error_mean, delimiter = ',')
