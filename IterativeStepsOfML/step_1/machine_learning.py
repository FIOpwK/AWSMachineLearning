from ast import parse
from operator import indexOf
import numpy as np
import xgboost as xgb
import sklearn as sk
import random


# supervised learning : every sample has value/lable/output
supervised_training_sample = {
    'sample_1': 'label',
    'sample_199': 'label',
    'sample_999999999': 'label',
}


# unsupervised learning : no labels for training_data 
unsupervised_training_sample = {}


# reinforcement learning : actions trigger situations 
def reinforced_training(sample_size):
    # this represents if cat is present in image
    return whichAction(sample_size) == 0
    

# which actions to take to maximize a reward on the way to reaching a specific goal
def whichAction(x):
    actions = ["left", "right", "forward", "backward"]

    if x not in range(1, x*x):
        return 0

    if len(str(x)) >= 2:
        return [actions[0], actions[3]]

    elif len(str(x)) <= 1:
        return [actions[0], actions[0]]
    
    elif len(str(x)) == 0:
        return [actions[3], actions[1]]

    return [
        actions[random.randrange(actions[0],actions[len(actions)])], 
        actions[actions[random.randrange(actions[0],actions[len(actions)])], actions[0]]
    ]