import math
import os

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
directory = os.path.dirname(__file__)

def preprocessData(sampleLoanData):
    selected_features = sampleLoanData[
        ["loan_id",
         "activity_period",
         "unpaid_principal_balance"]]
    return selected_features


feature = preprocessData(sampleLoanData).groupby('loan_id')

def buildGraph(key):
    plt.figure(figsize=(13, 8))
    ax = plt.subplot(1, 1, 1)
    ax.set_title("Loan ID " + str(key))
    plt.xlabel("Activity Period")
    plt.ylabel("Unpaid Principle Balance")
    plt.plot(feature.get_group(key)["activity_period"], feature.get_group(key)["unpaid_principal_balance"])
    filepath : str = 'graphs/' + str(key) + '/'
    filename : str = str(key) + 'DataGraph.png'
    folder_path = os.path.join(directory, filepath)

    if not os.path.isdir(folder_path):
        os.makedirs(folder_path)

    plt.savefig(folder_path + filename)

def buildGraphs(feature):
    for key, item in feature:
        print(feature.get_group(key), "\n\n")
        print(key)
        buildGraph(key)