from matplotlib import pyplot as plt
import pandas as pd


def plot_training_loss(loss : pd.Series):
  """Plot the loss/epoch function"""

  plt.figure()
  plt.xlabel("Epoch")
  plt.ylabel(str(loss.name))

  plt.plot(loss.index, loss.values, label="Loss")
  plt.legend()
  plt.show() 

def plot_model_accuracy(
    df : pd.DataFrame,
    label_key : str,
    label : str = "Label",
    prediction_key : str = "prediction",
    prediction : str = "Prediction"):
    """Plots the predicted values against the actual values using scatter plot for the predicted values.

    Args:
        df (pd.DataFrame): the dataframe that contains the data
        label_key (str): the expected output/label column's key in the dataframe
        label (str, optional): the expected output's user friendly name. Defaults to "Label".
        prediction_key (str, optional): the prediction column's key in the dataframe. Defaults to "prediction".
        prediction (str, optional): the prediction's user friendly name. Defaults to "Prediction".
    """

    plt.plot(df[label_key], df[label_key], label=label)
    plt.scatter(df[label_key],df[prediction_key] , c='r', label=prediction)
    
    plt.legend()

    plt.show()