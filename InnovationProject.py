import math

from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import dateutil
from tensorflow.python.data import Dataset

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

sampleLoanData = pd.read_csv("C:\PersonalProject\innovationRepo\LoanData.csv", sep=",")

def preprocessData(sampleLoanData):
    selected_features = sampleLoanData[
        ["loan_id",
         "activity_period",
         "unpaid_principal_balance"]]
    return selected_features


feature = preprocessData(sampleLoanData).groupby('loan_id')

for key, item in feature:

    print(feature.get_group(key), "\n\n")
    print(key)

def buildGraph():
    plt.figure(figsize=(13, 8))

    ax = plt.subplot(1, 2, 1)
    ax.set_title("Validation Data")

    ax.set_autoscaley_on(False)
    ax.set_ylim([32, 43])
    ax.set_autoscalex_on(False)
    ax.set_xlim([-126, -112])
    plt.scatter(feature.get_group[1000000]["unpaid_principal_balance"],
                feature.get_group[1000000]["activity_period"],
                cmap="coolwarm",
                c=feature.get_group[1000000]["activity_period"] / feature.get_group[1000000]["activity_period"].max())

    # ax = plt.subplot(1,2,2)
    # ax.set_title("Training Data")
    #
    # ax.set_autoscaley_on(False)
    # ax.set_ylim([32, 43])
    # ax.set_autoscalex_on(False)
    # ax.set_xlim([-126, -112])
    # plt.scatter(training_examples["longitude"],
    #             training_examples["latitude"],
    #             cmap="coolwarm",
    #             c=training_targets["median_house_value"] / training_targets["median_house_value"].max())
    _ = plt.plot()