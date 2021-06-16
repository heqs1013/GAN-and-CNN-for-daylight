import os
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import lpips
import torch
import scipy

if __name__ == "__main__":
    path_test = 'test_200/'
    path_truth = 'datasets/facades_rgb/test/'
    # mark_path = 'datasets_output/grey_png/test/'
    path_lux = 'lux/test/'
    path_rgb = 'datasets_output/compare/'
    path_save_fig = 'datasets_output/figs3/'
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

    img1_list = []
    img2_list = []
    img3_list = []
    ssim_truth = []
    ssim_pred = []
    error_mean = np.ndarray((len(folders_1), 1))
    for i in [43, 11, 73]:
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

        """黑白转为照度值"""
        """映射为伪彩色"""
        # LPIPS
        lpips_img0_o = lpips.im2tensor(lpips.load_image(path_truth + file_truth))  # RGB image from [-1,1]
        lpips_img0 = torch.split(lpips_img0_o, 256, dim=3)[1]
        lpips_img1 = lpips.im2tensor(lpips.load_image(path_test + file_test))
        loss_fn_alex = lpips.LPIPS(net='alex')
        dist01 = loss_fn_alex(lpips_img0, lpips_img1)
        print(dist01)

        img1 = Image.open(path_truth + file_truth)
        """格式观察，需pip install scipy==1.2.1"""
        # a = scipy.misc.imread(img1)
        img_truth_color = np.array(img1)[:, 256:]
        img_truth_grey = np.array(img1)[:, :256]
        img2 = Image.open(path_test + file_test)
        # img_test = np.array(img2)
        img_test = np.array(img2.convert('L'))

        # ssim_truth.append(img_truth_grey)
        # ssim_pred.append(img_test)

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
        # plt.subplot(1, 3, 1)
        # plt.title('ground_truth')
        # plt.imshow(img_origin_color)

        img1_list.append(img_origin_color)

        img_predict = Image.fromarray(np.uint8(predict_color))
        # plt.subplot(1, 3, 2)
        # plt.title('predict(grey)')
        # plt.imshow(img_predict)
        img2_list.append(img_predict)

        img_diff = Image.fromarray(np.uint8(diff_color))
        # plt.subplot(1, 3, 3)
        # plt.title('error(grey)')
        # plt.imshow(img_diff)
        img3_list.append(img_diff)

        ssim_truth.append(np.uint8(img_truth_color))
        ssim_pred.append(np.uint8(predict_color))

        # plt.show()
        # plt.savefig(path_save_fig + file_truth + '.png')
        # np.savetxt(path_save_lux + file_lux + '.txt', value_lux_predict, fmt='%.4e')

    """计算SSIM值"""
    for i in range(3):
        ssim_score = ssim(ssim_truth[i], ssim_pred[i], data_range=255, multichannel=True)

        print(ssim_score)

    """画图"""
    # imshow直接处理矩阵颜色有偏差，奇怪
    for i in range(3):
        plt.subplot(3, 3, 1 + i * 3)
        if i == 0:
            plt.title('ground_truth')
        plt.imshow(img1_list[i])
        plt.axis('off')

        plt.subplot(3, 3, 2 + i * 3)
        if i == 0:
            plt.title('predict')
        plt.imshow(img2_list[i])
        plt.axis('off')

        plt.subplot(3, 3, 3 + i * 3)
        if i == 0:
            plt.title('error')
        plt.imshow(img3_list[i])
        plt.axis('off')

        # cmap = mpl.cm.cool
        # norm = mpl.colors.Normalize(vmin=0, vmax=2000)
        #
        # fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
        #              cax=ax, orientation='horizontal', label='Some Units')

    # plt.show()
    plt.savefig(path_save_fig + '9.png')
    # np.savetxt(path_save_lux + file_lux + '.txt', value_lux_predict, fmt='%.4e')