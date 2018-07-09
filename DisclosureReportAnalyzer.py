import pandas as pd
import tensorflow as tf
import numpy as np

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 20
pd.options.display.float_format = '{:.1f}'.format

sampleLoanData = pd.read_csv("C:\PersonalProject\innovationRepo\Data\SampleLoanData - SampleDisclosureFile.csv", sep=",")

# print(sampleLoanData)

# print(sampleLoanData.dtypes)

dataFrame = pd.DataFrame(sampleLoanData)
loanIdGroupingFrame = dataFrame.groupby('Loan ID')

sellervocab = {'Bank of America': np.array([1, 0, 0, 0]), 'J.P. Morgan': np.array([0, 1, 0, 0]), 'Wells Fargo': np.array([0, 0, 1, 0]), 'Other': np.array([0, 0, 0, 1])}
servicervocab = {'Ditech Financial LLC': np.array([1, 0, 0, 0]), 'N.A.': np.array([0, 1, 0, 0]), 'Ocwen Loan Servicing, LLC': np.array([0, 0, 1, 0]), 'Other': np.array([0, 0, 0, 1])}

def preprocess_data(sampleLoanData):
    processedSampleLoanData = sampleLoanData.copy(True)
    vector_sellername = []
    for seller in processedSampleLoanData['Seller Name']:
        if sellervocab[seller] is not None:
            vector_sellername.append(sellervocab[seller])
        else:
            print("Other hit for Seller :: " + seller)
            vector_sellername.append(sellervocab['Other'])

    vector_servicername = []
    for servicer in processedSampleLoanData['Servicer Name']:
        if servicervocab[servicer] is not None:
            vector_servicername.append(servicervocab[servicer])
        else:
            print("Other hit for Servicer :: " + servicer)
            vector_servicername.append(servicervocab['Other'])

    processedSampleLoanData['Seller Name'] = vector_sellername
    processedSampleLoanData['Servicer Name'] = vector_servicername

    return processedSampleLoanData

def iterate_through_group_by(loanIdGroupingFrame):
    for key in loanIdGroupingFrame.groups.keys():
        proccess_data(loanIdGroupingFrame.get_group(key))


def proccess_data(groupByFrame):
    firstTime = False;
    for key in groupByFrame["UPB"].keys():
        if firstTime:
            difference = groupByFrame["UPB"][key] - groupByFrame["UPB"][key - 1]
            print("Diff :: " + str(difference) + " = " + str(groupByFrame["UPB"][key]) + " - " + str(groupByFrame["UPB"][key - 1]))
        else:
            firstTime = True



# iterate_through_group_by(loanIdGroupingFrame)

print(sellervocab)
print(servicervocab)

frame = preprocess_data(sampleLoanData)
servicer_name = frame['Servicer Name']
seller_name = frame['Seller Name']

seller_x_servicer = tf.feature_column.crossed_column(set([servicer_name, seller_name]), hash_bucket_size=1000)

print(seller_x_servicer)




