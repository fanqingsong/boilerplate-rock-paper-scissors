

import mchmm as mc
import numpy as np

winDict = {"R": "P", "S": "R", "P": "S"}


def player(prev_play, opponent_history=[]):
    # firstCall
    if len(opponent_history) <= 0:
        opponent_history.append("R")
        opponent_history.append("S")

    if len(prev_play) <= 0:
        prev_play = "P"
    # /firstCall

    opponent_history.append(prev_play)

    memory = 800
    guess = predict(prev_play, opponent_history, memory)

    return guess


def predict(prev_play, oppnent_history, memoryLength):
    if len(oppnent_history) > memoryLength:
        oppnent_history.pop(0)

    chain = mc.MarkovChain().from_data(oppnent_history)
    # print(chain.states)

    predictionNextItem = giveMostProbableNextItem(chain, prev_play)

    winningMove = winDict[predictionNextItem]

    return winningMove


def giveIndexOfState(chain, item):
    positions = np.where(chain.states == item)
    # print(positions)

    return positions[0][0]


def giveMostProbableNextItem(chain, lastItem):
    state_index = giveIndexOfState(chain, lastItem)

    # print(chain.observed_p_matrix)

    observed_probability = chain.observed_p_matrix[state_index]

    observed_index = np.argmax(observed_probability)

    retval = chain.states[observed_index]

    return retval