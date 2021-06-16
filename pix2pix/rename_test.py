from PIL import Image
import os

if __name__ == "__main__":
    test_path = 'datasets/facades/test/'
    output_path = 'datasets/facades/test_rename'
    folder = os.listdir(test_path)
    for cnt in range(len(folder)):
        file = Image.open(test_path + folder[cnt])
        file.save(output_path + '/00' + str(cnt) + '.jpg')
