import os

import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt

from skipgram import SkipGramModel
from bayesian_skipgram import BayesianSkipgram

if __name__=="__main__":

    model_objects = {
        SkipGramModel: "skipgram",
        BayesianSkipgram: "bayesian_skipgram"
    }
    model_root = "./models"
    figs = {}
    axs = {}
    for model, model_name in model_objects.items():
        print("Loading: ", model_name)
        model_ready, loss, params = model.load(model_root, model_name)
        print("Loaded model -", model_name)
        plt.plot(loss)
        plt.xlabel("Number of Epochs")
        plt.ylabel("Mean Loss")
        plt.title("{} Loss".format(model_name))
        plt.show()
