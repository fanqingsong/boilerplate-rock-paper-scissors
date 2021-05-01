# The example function below keeps track of the opponent's history and plays whatever the opponent played two plays ago. It is not a very good player so you will need to change the code to pass the challenge.

import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense, Input, LSTM
from keras.layers.core import Dense, Activation, Dropout
from keras.utils import np_utils
import keras as K

look_back = 4
win_dict = {"R": "P", "S": "R", "P": "S"}


def create_nn_model():
    init = K.initializers.glorot_uniform(seed=1)
    simple_adam = K.optimizers.Adam()

    model = Sequential()
    # model.add(Input(shape=(look_back,)))
    model.add(LSTM(10, input_shape=(1,look_back)))
    # model.add(Dense(20, activation='relu'))
    model.add(Dense(10, activation='relu'))
    # model.add(Dropout(0.3))
    # model.add(Dense(5, activation='relu'))
    # model.add(Dropout(0.3))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=simple_adam, metrics=['accuracy'])

    return model


def player(prev_play, opponent_history, model, batch_x, batch_y, review_epochs=10):
    # print(f"now player1 is in turn, opponent play is {prev_play}")

    plays = ["R","P","S"]
    play_dict = {"R":0,"P":1,"S":2}
    plays_categorial = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    opponent_history_len = len(opponent_history)
    # print(f"opponent_history_len = {opponent_history_len}")

    if opponent_history_len < look_back:
        if prev_play:
            opponent_history.append(prev_play)
        guess = random.randint(0,2)
        return plays[guess]

    one_x = [play_dict[move] for move in opponent_history[-look_back:]]

    one_y = play_dict[prev_play]
    one_y = plays_categorial[one_y]

    batch_x.append(one_x)
    batch_y.append(one_y)

    for i in range(0, review_epochs):
        # print(f"now train by epoch {i}")
        batch_x = np.array(batch_x)
        # print(batch_x.shape)
        batch_x_final = np.reshape(batch_x, (batch_x.shape[0], 1, batch_x.shape[1]))
        # print(batch_x.shape)

        batch_y = np.array(batch_y)
        # print(batch_y.shape)

        # print(batch_x_final.shape)
        # print(batch_y.shape)
        model.train_on_batch(batch_x_final, batch_y)

    opponent_history.append(prev_play)

    current_x = [play_dict[move] for move in opponent_history[-look_back:]]
    current_x = np.array([current_x])
    current_x = np.reshape(current_x, (current_x.shape[0], 1, current_x.shape[1]))
    predict_y = model.predict_on_batch(current_x)
    predict_y = predict_y.tolist()
    # print(predict_y)
    predict_y = predict_y[0]
    guess = np.argmax(predict_y)
    # print(guess)

    opponent_play = plays[guess]
    me_play = random.choice(['R', 'P', 'S'])
    me_play = win_dict.get(opponent_play, me_play)

    return me_play



