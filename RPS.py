# The example function below keeps track of the opponent's history and plays whatever the opponent played two plays ago. It is not a very good player so you will need to change the code to pass the challenge.

import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.layers.core import Dense, Activation, Dropout
from keras.utils import np_utils

epochs = 1
look_back = 20


def create_nn_model():
    model = Sequential()
    model.add(Input(shape=(look_back,)))
    model.add(Dense(5, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

    return model


def player(prev_play, opponent_history, model, batch_x, batch_y):
    # print("now player1 give result")

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

    for i in range(0, epochs):
        # print(f"now train by epoch {i}")
        # print(batch_x)
        # print(batch_y)
        batch_x = np.array(batch_x)
        batch_y = np.array(batch_y)
        # print(batch_x.shape)
        # print(batch_y.shape)
        model.train_on_batch(batch_x, batch_y)

    opponent_history.append(prev_play)

    current_x = [play_dict[move] for move in opponent_history[-look_back:]]
    current_x = np.array([current_x])
    predict_y = model.predict_on_batch(current_x)
    predict_y = predict_y.tolist()
    # print(predict_y)
    predict_y = predict_y[0]
    guess = np.argmax(predict_y)
    # print(guess)

    opponent_play = plays[guess]
    me_play = random.choice(['R', 'P', 'S'])
    if opponent_play == "R":
        me_play = "P"
    elif opponent_play == "S":
        me_play = "R"
    elif opponent_play == "P":
        me_play = "S"

    return me_play



