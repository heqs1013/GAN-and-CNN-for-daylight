import tensorflow as tf
import config
from prepare_data import generate_datasets
from train import get_model
import numpy as np
import time

if __name__ == '__main__':

    # GPU settings
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    # get the original_dataset
    # train_dataset, valid_dataset, test_dataset, train_count, valid_count, test_count = generate_datasets()
    test_dataset, test_count = generate_datasets(type='test')
    
    # load the model
    start_time = time.time()
    model = get_model()
    model.load_weights(filepath=config.save_model_dir)
    end_time = time.time()
    print('load time: {}s'.format(end_time - start_time))

    # Get the accuracy on the test set
    test_loss = tf.keras.metrics.Mean()
    test_accuracy = tf.keras.metrics.MeanAbsoluteError()

    @tf.function
    def my_test_step(images, labels):
        predictions = model(images, training=False)
        t_loss = tf.keras.losses.MSE(labels, predictions)

        test_loss(t_loss)
        test_accuracy(labels, predictions)

    # 修改
    predictions_arr = np.ndarray((test_count, 2))
    labels_arr = np.ndarray((test_count, 2))
    cnt = 0
    start_time = time.time()
    for test_images, test_labels in test_dataset:
        predictions = model(test_images, training=False)
        predictions_arr[config.BATCH_SIZE * cnt: config.BATCH_SIZE * cnt + predictions.shape[0], :] = predictions.numpy()
        cnt += 1

        my_test_step(test_images, test_labels)
        print("loss: {:.5f}, test accuracy: {:.5f}".format(test_loss.result(),
                                                           test_accuracy.result()))

    end_time = time.time()
    print('prediction time: {}s'.format(end_time - start_time))