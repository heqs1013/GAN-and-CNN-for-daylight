import tensorflow as tf
import config
import pathlib
from config import image_height, image_width, channels
import numpy as np


def load_and_preprocess_image(img_path):
    # read pictures
    img_raw = tf.io.read_file(img_path)
    # decode pictures
    img_tensor = tf.image.decode_jpeg(img_raw, channels=channels)
    # resize
    img_tensor = tf.image.resize(img_tensor, [image_height, image_width])
    img_tensor = tf.cast(img_tensor, tf.float32)
    # normalization
    img = img_tensor / 255.0
    return img

def get_images_and_labels(data_root_dir):
    """
    :param data_root_dir:
    :return:
    """
    # get all images' paths (format: string)
    data_root = pathlib.Path(data_root_dir)
    all_image_path = [str(path) for path in list(data_root.glob('*/*.jpg'))]
    # # get labels' names
    # label_names = sorted(item.name for item in data_root.glob('*/'))
    # # dict: {label : index}
    # label_to_index = dict((label, index) for index, label in enumerate(label_names))
    # # get all images' labels
    # all_image_label = [label_to_index[pathlib.Path(single_image_path).parent.name] for single_image_path in all_image_path]

    label_min, label_max = label_min_max(data_root)

    # float_to_label
    all_image_label = [get_float_label(single_image_path, label_min, label_max) for single_image_path in all_image_path]

    return all_image_path, all_image_label


def label_min_max(data_root):

    """
    :param data_root:
    :return:
    """
    all_label_path = [str(path) for path in list(data_root.glob('*/*.npy'))]
    
	"""static"""
    # min_arr = [1, 5000, 1]
    # max_arr = [0, 0, 0]
    
	"""annual"""
	min_arr = [100, 100]
    max_arr = [0, 0]
    for npy_path in all_label_path:
        arr = np.load(npy_path, allow_pickle=True)
        for i in range(len(min_arr)):
            if arr[i] < min_arr[i]:
                min_arr[i] = arr[i]
            if arr[i] > max_arr[i]:
                max_arr[i] = arr[i]

    return min_arr, max_arr


def get_float_label(single_image_path, min_arr, max_arr):
    """
    :param single_image_path: */*.jpg
    :return: np.array[float, float, float]
    """
    # parent_path = pathlib.Path(single_image_path).parent.parent
    # img_name = pathlib.Path(single_image_path).name
    npy_path = pathlib.Path(single_image_path).with_suffix('.npy')
    arr = np.load(npy_path, allow_pickle=True)
    for i in range(len(min_arr)):
        arr[i] = (arr[i] - min_arr[i]) / (max_arr[i] - min_arr[i]) * 2 - 1

    return arr


def get_dataset(dataset_root_dir):
    all_image_path, all_image_label = get_images_and_labels(data_root_dir=dataset_root_dir)

    # load the dataset and preprocess images
    image_dataset = tf.data.Dataset.from_tensor_slices(all_image_path).map(load_and_preprocess_image)
    label_dataset = tf.data.Dataset.from_tensor_slices(all_image_label)
    dataset = tf.data.Dataset.zip((image_dataset, label_dataset))
    image_count = len(all_image_path)

    return dataset, image_count


def generate_datasets(type):
    test_dataset, test_count = get_dataset(dataset_root_dir=config.test_dir)
    test_dataset = test_dataset.batch(batch_size=config.BATCH_SIZE)

    if type == 'test':
        return test_dataset,test_count

    train_dataset, train_count = get_dataset(dataset_root_dir=config.train_dir)
    valid_dataset, valid_count = get_dataset(dataset_root_dir=config.valid_dir)

    # read the original_dataset in the form of batch
    train_dataset = train_dataset.shuffle(buffer_size=train_count).batch(batch_size=config.BATCH_SIZE)
    valid_dataset = valid_dataset.batch(batch_size=config.BATCH_SIZE)

    return train_dataset, valid_dataset, test_dataset, train_count, valid_count, test_count
