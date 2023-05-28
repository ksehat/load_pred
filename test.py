from keras.models import Model
from keras.layers import Input, Dense, Dropout, Concatenate
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.ensemble import AdaBoostRegressor


def simple_model(input_shape):
    # create model
    inputs = Input(shape=(input_shape[1],))
    x = Dense(25, kernel_initializer='normal', activation='relu')(inputs)
    x = Dropout(0.2)(x)
    x = Dense(10, kernel_initializer='normal', activation='relu')(x)

    # define additional branches
    y = Dense(20, kernel_initializer='normal', activation='relu')(inputs)
    y = Dropout(0.2)(y)
    y = Dense(10, kernel_initializer='normal', activation='relu')(y)

    z = Dense(15, kernel_initializer='normal', activation='relu')(inputs)
    z = Dropout(0.2)(z)
    z = Dense(5, kernel_initializer='normal', activation='relu')(z)

    # concatenate the outputs of the branches
    concatenated = Concatenate()([x, y, z])

    outputs = Dense(1, kernel_initializer='normal')(concatenated)

    model = Model(inputs=inputs, outputs=outputs)
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

ann_estimator = KerasRegressor(build_fn=lambda: simple_model(x_train.shape), epochs=100, batch_size=10, verbose=0)
boosted_ann = AdaBoostRegressor(base_estimator=ann_estimator)
# boosted_ann.fit(rescaledX, y_train.values.ravel())  # scale your training data
# boosted_ann.predict(rescaledX_Test)
