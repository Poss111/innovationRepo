import math
import os
from matplotlib import cm
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.data import Dataset

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

sampleLoanData = pd.read_csv("C:\PersonalProject\innovationRepo\Data\SampleLoanData - SampleDisclosureFile.csv", sep=",")

print(sampleLoanData)

print(sampleLoanData.dtypes)

dataFrame = pd.DataFrame(sampleLoanData)
loanIdGroupingFrame = dataFrame.groupby('Loan ID').sum()

print(loanIdGroupingFrame.index)
