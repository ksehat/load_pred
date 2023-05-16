import copy

import tensorflow as tf, keras
from keras import layers


def create_model(input_data, layer_types, layer_sizes):
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
            x = layers.Conv1D(layer_size, kernel_size=2, activation="relu")(x)
        elif layer_type == 'LSTM':
            x = layers.LSTM(layer_size)(x)
    x = layers.Flatten()(x)
    output_layer1 = layers.Dense(1)(x)

    y = input_layer
    # y = layers.Conv1D(20, kernel_size=3, activation="relu")(y)
    # y = layers.Conv1D(20, kernel_size=1, activation="relu")(y)
    y = layers.Dense(20, activation="relu")(y)
    y = layers.Dense(10, activation="relu")(y)
    y = layers.Dense(5, activation="relu")(y)
    y = layers.Flatten()(y)
    output_layer2 = layers.Dense(1)(y)


    # y = layers.Conv1D(20, kernel_size=3, activation="relu")(y)
    # y = layers.Conv1D(20, kernel_size=1, activation="relu")(y)
    z = layers.LSTM(20, activation="relu")(input_layer)
    z = layers.Dense(10, activation="relu")(z)
    z = layers.Dense(5, activation="relu")(z)
    z = layers.Flatten()(z)
    output_layer3 = layers.Dense(1)(z)


    output = layers.Concatenate()([output_layer1, output_layer2, output_layer3])

    x1 = layers.Dense(100, activation="relu")(output)
    # x1 = layers.Dense(80, activation="relu")(x1)
    output_layer = layers.Dense(1)(x1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    print(model.summary())

    return model
