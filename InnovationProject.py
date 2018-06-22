import math
import os

from matplotlib import cm
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import sklearn.metrics
from tensorflow.python.data import Dataset

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

sampleLoanData = pd.read_csv("C:\PersonalProject\innovationRepo\LoanData.csv", sep=",")
directory = os.path.dirname(__file__)
sampleLoanDataForMachineLearning = pd.read_csv(("C:\\PersonalProject\\innovationRepo\\" + "/SampleLoanData-PercentageOfChangeAndLikelinessToDelin.csv"), sep=",")

def preprocessData(sampleLoanDataForMachineLearning):
    selected_features = sampleLoanDataForMachineLearning[
        ["loan_id",
         "percentage_of_change",
         "likeliness_to_go_into_deliquency"]]

    return selected_features


feature = preprocessData(sampleLoanDataForMachineLearning)
feature.describe()
target = sampleLoanDataForMachineLearning["likeliness_to_go_into_deliquency"]

def build_graph_for_percent_over_delinquency(feature):
    plt.figure(figsize=(13, 8))
    ax = plt.subplot(1, 1, 1)
    ax.set_title("Percentage of Change over Delinquency Likeliness ")
    plt.xlabel("Percentage of Change")
    plt.ylabel("Likeliness to go into Delinquency")
    plt.scatter(feature["percentage_of_change"], feature["likeliness_to_go_into_deliquency"])


def build_graph(key,data):
    plt.figure(figsize=(13, 8))
    ax = plt.subplot(1, 1, 1)
    ax.set_title("Loan ID " + str(key))
    plt.xlabel("Activity Period")
    plt.ylabel("Unpaid Principle Balance")
    ax.set_ylim(data["current_unpaid_principal_balance"].min().min(), data["current_unpaid_principal_balance"].max().max())
    plt.plot(data.get_group(key)["activity_period"], data.get_group(key)["current_unpaid_principal_balance"])
    filepath : str = 'graphs/' + str(key) + '/'
    filename : str = str(key) + 'DataGraph.png'
    folder_path = os.path.join(directory, filepath)

    if not os.path.isdir(folder_path):
        os.makedirs(folder_path)

    plt.savefig(folder_path + filename)


def build_graphs(feature):
    for key, item in feature:
        print(feature.get_group(key), "\n\n")
        print(key)
        build_graph(key,feature)


def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    features = {key:np.array(value) for key, value in dict(features).items()}

    ds = Dataset.from_tensor_slices((features, targets))
    ds = ds.batch(batch_size).repeat(num_epochs)

    if shuffle:
        ds = ds.shuffle(10000)

    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels


def train_model(learning_rate, steps, batch_size, input_feature):
    """Trains a linear regression model.

    Args:
      learning_rate: A `float`, the learning rate.
      steps: A non-zero `int`, the total number of training steps. A training step
        consists of a forward and backward pass using a single batch.
      batch_size: A non-zero `int`, the batch size.
      input_feature: A `string` specifying a column from `sampleLoanDataForMachineLearning`
        to use as input feature.

    Returns:
      A Pandas `DataFrame` containing targets and the corresponding predictions done
      after training the model.
    """

    periods = 40
    steps_per_period = steps / periods

    my_feature = input_feature
    my_feature_data = sampleLoanDataForMachineLearning[[my_feature]].astype('float32')
    my_label = "likeliness_to_go_into_deliquency"
    targets = sampleLoanDataForMachineLearning[my_label].astype('float32')

    # Create input functions.
    training_input_fn = lambda: my_input_fn(my_feature_data, targets, batch_size=batch_size)
    predict_training_input_fn = lambda: my_input_fn(my_feature_data, targets, num_epochs=1, shuffle=False)

    # Create feature columns.
    feature_columns = [tf.feature_column.numeric_column(my_feature)]

    # Create a linear regressor object.
    my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
    linear_regressor = tf.estimator.LinearRegressor(
        feature_columns=feature_columns,
        optimizer=my_optimizer
    )

    # Set up to plot the state of our model's line each period.
    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)
    plt.title("Learned Line by Period")
    plt.ylabel(my_label)
    plt.xlabel(my_feature)
    sample = sampleLoanDataForMachineLearning.sample(n=40)
    plt.scatter(sample[my_feature], sample[my_label])
    colors = [cm.coolwarm(x) for x in np.linspace(-1, 1, periods)]

    # Train the model, but do so inside a loop so that we can periodically assess
    # loss metrics.
    print("Training model...")
    print("RMSE (on training data):")
    root_mean_squared_errors = []
    for period in range (0, periods):
        # Train the model, starting from the prior state.
        linear_regressor.train(
            input_fn=training_input_fn,
            steps=steps_per_period,
        )
        # Take a break and compute predictions.
        predictions = linear_regressor.predict(input_fn=predict_training_input_fn)
        predictions = np.array([item['predictions'][0] for item in predictions])

        # Compute loss.
        root_mean_squared_error = math.sqrt(sklearn.metrics.mean_squared_error(predictions, targets))
        # Occasionally print the current loss.
        print("  period %02d : %0.2f" % (period, root_mean_squared_error))
        # Add the loss metrics from this period to our list.
        root_mean_squared_errors.append(root_mean_squared_error)
        # Finally, track the weights and biases over time.
        # Apply some math to ensure that the data and line are plotted neatly.
        y_extents = np.array([0, sample[my_label].max()])

        weight = linear_regressor.get_variable_value('linear/linear_model/%s/weights' % input_feature)[0]
        bias = linear_regressor.get_variable_value('linear/linear_model/bias_weights')

        x_extents = (y_extents - bias) / weight
        x_extents = np.maximum(np.minimum(x_extents,
                                          sample[my_feature].max()),
                               sample[my_feature].min())
        y_extents = weight * x_extents + bias
        plt.plot(x_extents, y_extents, color=colors[period])
        filepath : str = 'graphs/'
        filename : str = 'Model.png'
        folder_path = os.path.join(directory, filepath)

        if not os.path.isdir(folder_path):
            os.makedirs(folder_path)

        plt.savefig(folder_path + filename)
    print("Model training finished.")

    # Output a graph of loss metrics over periods.
    plt.subplot(1, 2, 2)
    plt.ylabel('RMSE')
    plt.xlabel('Periods')
    plt.title("Root Mean Squared Error vs. Periods")
    plt.tight_layout()
    plt.plot(root_mean_squared_errors)
    filepath : str = 'graphs/'
    filename : str = str(learning_rate) + '_' + str(period) + '_' + str(batch_size) + '_' + str(steps) + 'RSME.png'
    folder_path = os.path.join(directory, filepath)

    if not os.path.isdir(folder_path):
        os.makedirs(folder_path)

    plt.savefig(folder_path + filename)

    # Create a table with calibration data.
    calibration_data = pd.DataFrame()
    calibration_data["predictions"] = pd.Series(predictions)
    calibration_data["targets"] = pd.Series(targets)

    print("Final RMSE (on training data): %0.2f" % root_mean_squared_error)

    return calibration_data


# build_graphs(sampleLoanData.groupby("loan_id"))

calibration_data = train_model(
    learning_rate=0.021275,
    steps=1000,
    batch_size=40,
    input_feature="percentage_of_change")


print(calibration_data["predictions"])