import tensorflow as tf
from tensorflow.keras import layers, models 

def build_model(input_shape):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (4,8), activation=tf.nn.relu, input_shape=input_shape, padding='same'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(32, (2,4), activation=tf.nn.relu, padding='same'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(32, (2,4), activation=tf.nn.relu, padding='same'))
    model.add(layers.Flatten())
    model.add(layers.Dense(50, activation=tf.nn.relu))
    model.add(layers.Dense(1, activation='linear'))
    return model

def init_model(weights_path, lat=24, lon=72, var_num=4, lr=0.0001):
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss = tf.keras.losses.MeanSquaredError()
    metrics = tf.keras.metrics.MeanSquaredError()
    model = build_model((lat, lon, var_num))
    model.compile(optimizer=optimizer, loss=loss, metris=[metrics])
    model.load_weights(weights_path)
    return model

def prediction(model, x_val):
    pred = model.predict(x_val)[:,0]
    return pred
