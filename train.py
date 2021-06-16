from __future__ import absolute_import, division, print_function
import tensorflow as tf
from models.resnet import resnet_18, resnet_34, resnet_50, resnet_101, resnet_152
import config
from prepare_data import generate_datasets
import math
import time
import numpy as np
import os


def get_model():
    model = resnet_50()
    if config.model == "resnet18":
        model = resnet_18()
    if config.model == "resnet34":
        model = resnet_34()
    if config.model == "resnet101":
        model = resnet_101()
    if config.model == "resnet152":
        model = resnet_152()
    model.build(input_shape=(None, config.image_height, config.image_width, config.channels))
    model.summary()
    return model


if __name__ == '__main__':
    # GPU settings
    print(tf.__version__)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    # get the original_dataset
    train_dataset, valid_dataset, test_dataset, train_count, valid_count, test_count = generate_datasets()

    # create model
    model = get_model()

    # define loss and optimizer
    optimizer = tf.keras.optimizers.Adadelta()

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.MeanSquaredError(name='train_accuracy')

    valid_loss = tf.keras.metrics.Mean(name='valid_loss')
    valid_accuracy = tf.keras.metrics.MeanSquaredError(name='valid_accuracy')

    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            predictions = model(images, training=True)
            loss = tf.keras.losses.MSE(y_true=labels, y_pred=predictions)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(grads_and_vars=zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_accuracy(labels, predictions)

    @tf.function
    def valid_step(images, labels):
        predictions = model(images, training=False)
        v_loss = tf.keras.losses.MSE(y_true=labels, y_pred=predictions)

        valid_loss(v_loss)
        valid_accuracy(labels, predictions)

    # start training
    a = time.time()
    log_list = []
    log_val = []
    for epoch in range(config.EPOCHS):
        train_loss.reset_states()
        train_accuracy.reset_states()
        valid_loss.reset_states()
        valid_accuracy.reset_states()
        step = 0
        for images, labels in train_dataset:
            step += 1
            train_step(images, labels)
            print("Epoch: {}/{}, step: {}/{}, loss: {:.5f}, accuracy: {:.5f}".format(epoch + 1,
                                                                                     config.EPOCHS,
                                                                                     step,
                                                                                     math.ceil(train_count / config.BATCH_SIZE),
                                                                                     train_loss.result(),
                                                                                     train_accuracy.result()))

            if np.mod(epoch, 10) == 0:
                log_list.append([epoch, step, train_loss.result(), train_accuracy.result()])

        for valid_images, valid_labels in valid_dataset:
            valid_step(valid_images, valid_labels)

        print("Epoch: {}/{}, train loss: {:.5f}, train accuracy: {:.5f}, "
              "valid loss: {:.5f}, valid accuracy: {:.5f}".format(epoch + 1,
                                                                  config.EPOCHS,
                                                                  train_loss.result(),
                                                                  train_accuracy.result(),
                                                                  valid_loss.result(),
                                                                  valid_accuracy.result()))
        log_val.append([epoch, train_loss.result(), train_accuracy.result(), valid_loss.result(), valid_accuracy.result()])
        if np.mod(epoch, 50) == 0:
            save_model = config.save_model_dir + str(epoch) + '/'
            if not os.path.isdir(save_model):
                os.makedirs(save_model)
            model.save_weights(filepath=save_model, save_format='tf')

    b = time.time()
    print('time cost: {}h'.format((b-a) / 3600))

    model.save_weights(filepath=config.save_model_dir, save_format='tf')

    # 学习曲线
    arr_train = np.array(log_list)
    np.save('arr_train_0604.npy', arr_train)

    arr_val = np.array(log_val)
    np.save('arr_val_0604.npy', arr_val)
    print('done')
