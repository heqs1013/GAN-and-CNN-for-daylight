import os
from PIL import Image
import numpy as np
# from skimage.metrics import structural_similarity as ssim
# import lpips
# import torch
from sklearn.metrics import mean_squared_error

if __name__ == "__main__":
    path_test = 'test_200/'
    path_test_list = ['test_10/', 'test_20/', 'test_30/', 'test_60/', 'test_80/', 'test_100/', 'test_130/', 'test_200/']

    # path_truth = 'datasets/facades_rgb/test/'
    path_truth = 'datasets/facades/test _all/'
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
    ssim_mean = []
    lpips_mean = []
    mse_mean = []

    for path_test in path_test_list:
        ssim_result = []
        lpips_result = []
        mse_result = []
        # for i in [43]:
        for i in range(75):
            # for i in range(len(folders_1)):
            file_test = folders_1[i]
            # facades_rgb/test, SSIM
            # if i < 9:
            #     file_truth = '0' + file_test[7:9] + '.jpg'
            # else:
            #     file_truth = '00' + file_test[7:9] + '.jpg'

            # facades/test_all, MSE
            if i < 9:
                file_truth = '00' + str(int(file_test[8]) - 1) + '.jpg'
            else:
                file_truth = '00' + str(int(file_test[7:9]) - 1) + '.jpg'

            file_lux = folders_4[i]
            print(file_truth, file_test)

            """黑白转为照度值"""
            """映射为伪彩色"""
            img1 = Image.open(path_truth + file_truth)
            img_truth_color = np.array(img1)[:, 256:]
            img2 = Image.open(path_test + file_test)
            img_test = np.array(img2.convert('L'))
            value_lux = np.ndarray((256, 256))
            value_lux[:, :] = -1
            value_lux_predict = np.ndarray((256, 256))
            value_lux_predict[:, :] = -1
            lux_array = np.loadtxt(path_lux + file_lux)
            value_lux[:, 32:224] = lux_array
            predict_color = np.ndarray((img_shape[0], img_shape[1], 3))

            for x in range(img_shape[1]):
                for y in range(img_shape[0]):
                    if value_lux[y, x] == -1:
                        predict_color[y, x, :] = (0, 0, 0)
                    else:
                        predict_color[y, x, :] = color_map[color_num - int(img_test[y, x] * color_num / 255)]

            """计算SSIM值"""
            # ssim_img0 = np.uint8(img_truth_color)
            # ssim_img1 = np.uint8(predict_color)
            # ssim_score = ssim(ssim_img0, ssim_img1, data_range=255, multichannel=True)
            # ssim_result.append(ssim_score)
            #
            # """计算LPIPS"""
            # lpips_img0_o = lpips.im2tensor(lpips.load_image(path_truth + file_truth))  # RGB image from [-1,1]
            # lpips_img0 = torch.split(lpips_img0_o, 256, dim=3)[1]
            # lpips_img1 = lpips.im2tensor(lpips.load_image(path_test + file_test))
            # loss_fn_alex = lpips.LPIPS(net='alex')
            # dist = loss_fn_alex(lpips_img0, lpips_img1)
            # lpips_result.append(dist.item())

            """计算MSE"""
            mse = mean_squared_error(img_truth_color, img_test)
            mse_result.append(mse)

        # ssim_m = np.mean(np.array(ssim_result))
        # lpips_m = np.mean(np.array(lpips_result))
        # ssim_mean.append(ssim_m)
        # lpips_mean.append(lpips_m)
        mse_m = np.mean(np.array(mse_result))
        mse_mean.append(mse_m)

    # print(ssim_mean)
    # print(lpips_mean)
    print(mse_mean)
    print(lpips_mean)