import numpy as np
import keras
import tensorflow as tf
from tensorflow import keras
from keras import layers



def auto_model(input_data, layer_types, layer_sizes):
    physical_devices = tf.config.list_physical_devices("GPU")
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    if 'Conv2D' in layer_types:
        input_shape = [input_data.shape[1], 1, 1]
    elif 'Conv1D' in layer_types:
        input_shape = [input_data.shape[1], 1]
    input_layer = keras.Input(shape=input_shape)
    x = input_layer
    for layer_type, layer_size in zip(layer_types, layer_sizes):
        if layer_type == 'Dense':
            x = layers.Dense(layer_size, activation="relu")(x)
        elif layer_type == 'Conv1D':
            # counter += 2
            x = layers.Conv1D(layer_size, kernel_size=int(np.floor((input_data.shape[1])/2)), activation="relu")(x)
        elif layer_type == 'LSTM':
            x = layers.LSTM(layer_size)(x)
    x = layers.Flatten()(x)
    output_layer1 = layers.Dense(1)(x)

    y = input_layer
    # y = layers.Conv1D(20, kernel_size=3, activation="relu")(y)
    # y = layers.Conv1D(20, kernel_size=1, activation="relu")(y)
    y = layers.Dense(150, activation="relu")(y)
    y = layers.Dense(100, activation="relu")(y)
    y = layers.Dense(50, activation="relu")(y)
    y = layers.Dense(25, activation="relu")(y)
    y = layers.Dense(10, activation="relu")(y)
    y = layers.Dense(5, activation="relu")(y)
    y = layers.Flatten()(y)
    output_layer2 = layers.Dense(1)(y)


    # y = layers.Conv1D(20, kernel_size=3, activation="relu")(y)
    # y = layers.Conv1D(20, kernel_size=1, activation="relu")(y)
    # z = layers.LSTM(50, activation="relu")(input_layer)
    # z = layers.Dense(20, activation="relu")(z)
    # z = layers.Dense(5, activation="relu")(z)
    # z = layers.Flatten()(z)
    # output_layer3 = layers.Dense(1)(z)


    output = layers.Concatenate()([output_layer1, output_layer2])

    x1 = layers.Dense(150, activation="relu")(output)
    # x1 = layers.Dense(80, activation="relu")(x1)
    output_layer = layers.Dense(1)(x1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    print(model.summary())

    return model


def manual_model(input_data):
    input_shape = [input_data.shape[1], 1]
    input_layer = keras.Input(shape=input_shape)

    x = input_layer
    x1 = layers.Conv1D(20, kernel_size=20, activation="relu")(x)
    # x1 = layers.BatchNormalization()(x1)
    x1 = layers.Dropout(0.25)(x1)
    x2 = layers.Conv1D(20, kernel_size=10, activation="relu")(x1)
    # x2 = layers.Dropout(0.25)(x2)
    # x = layers.Conv1D(10, kernel_size=10, activation="relu")(x)
    # x = layers.Dense(100, activation="relu")(x)
    # x = layers.Dense(50, activation="relu")(x)
    x3 = layers.Dense(5, activation="relu")(x2)
    x4 = layers.Dense(5, activation="relu")(x3)
    x5 = layers.Dense(5, activation="relu")(x4)
    x6 = layers.Flatten()(x5)
    output_layer1 = layers.Dense(1)(x6)

    y = input_layer
    y = layers.Dense(10, activation="relu")(y)
    y = layers.Dense(10, activation="relu")(y)
    y = layers.Dense(5, activation="relu")(y)
    y = layers.Flatten()(y)
    output_layer2 = layers.Dense(1)(y)

    z = x1
    z = layers.Dense(10, activation="relu")(z)
    z = layers.Dense(5, activation="relu")(z)
    z = layers.Dense(2, activation="relu")(z)
    z = layers.Flatten()(z)
    output_layer3 = layers.Dense(1)(z)

    z1 = x2
    z1 = layers.Dense(10, activation="relu")(z1)
    z1 = layers.Dense(5, activation="relu")(z1)
    z1 = layers.Dense(2, activation="relu")(z1)
    z1 = layers.Flatten()(z1)
    output_layer4 = layers.Dense(1)(z1)


    # y = layers.Conv1D(20, kernel_size=3, activation="relu")(y)
    # y = layers.Conv1D(20, kernel_size=1, activation="relu")(y)
    # z = layers.LSTM(50, activation="relu")(input_layer)
    # z = layers.Dense(20, activation="relu")(z)
    # z = layers.Dense(5, activation="relu")(z)
    # z = layers.Flatten()(z)
    # output_layer3 = layers.Dense(1)(z)


    output = layers.Concatenate()([output_layer1, output_layer2, output_layer3, output_layer4])

    x1 = layers.Dense(100, activation="relu")(output)
    # x1 = layers.Dense(80, activation="relu")(x1)
    output_layer = layers.Dense(1)(x1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    print(model.summary())

    return model