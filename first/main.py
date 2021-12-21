import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.layers import Dense


def get_two_sets():
    """
    :return: two numpy array
    """
    c = np.array([-40,-10,0,8,15,22,38])
    f = np.array([-40, 14, 32, 46, 59, 72, 100])
    return c, f


def get_model():
    """
    :return: model of neyron
    """
    return keras.Sequential()


def make_plot(log):
    """
    Making plot loss function
    :param log:
    :return:
    """
    try:
        plt.plot(log.history['loss'])
        plt.grid(True)
        plt.show()
    except Exception as err:
        print(err)

def main():
    c, f = get_two_sets()
    model = get_model()
    model.add(Dense(units=1, input_shape=(1,), activation='linear'))
    """
    units = 1 - один нейрон
    input_shape=(1,) - один вход
    Объект Dense - полносвязный слой
    """
    step_convergence = 0.1
    model.compile(loss='mean_squared_error', optimizer=keras.optimizer.Adam(step_convergence))
    """
    start
    """
    count_of_epochs = 500
    log = model.fit(c,f, epochs=count_of_epochs, verbose=False)
    make_plot(log)
    print(model.predict([100]))
    print(model.get_weights())
    

if __name__ == '__main__':
    main()