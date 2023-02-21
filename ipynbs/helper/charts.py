from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

def plot_training_loss(loss: pd.Series):
  """Plot the loss/epoch function"""

  plt.figure()
  plt.xlabel("Epoch")
  plt.ylabel(str(loss.name))

  plt.plot(loss.index, loss.values, label="Loss")
  plt.legend()
  plt.show()

def plot_training_losses(losses : list[str], df : pd.DataFrame, ignore_first : bool = True):
    """Plot the different loss functions per epoch"""
    if(ignore_first):
        #removing first row
        df = df.tail(-1)

    plt.figure()

    plt.xlabel("Epoch")
    plt.ylabel("Errors")

    for (i, loss) in enumerate(losses):
        plt.plot(df.index, df[loss], label=loss)

    plt.legend()
    plt.show()

def plot_heatmap_from_df(df: pd.DataFrame, x=6, y=6):
    plt.figure(figsize=(x, y))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="mako")
    plt.show()

def plot_model_accuracy(
        df: pd.DataFrame,
        label_key: str,
        label: str = "Label",
        prediction_key: str = "prediction",
        prediction: str = "Prediction"):
    """Plots the predicted values against the actual values using scatter plot for the predicted values.

    Args:
        df (pd.DataFrame): the dataframe that contains the data
        label_key (str): the expected output/label column's key in the dataframe
        label (str, optional): the expected output's user friendly name. Defaults to "Label".
        prediction_key (str, optional): the prediction column's key in the dataframe. Defaults to "prediction".
        prediction (str, optional): the prediction's user friendly name. Defaults to "Prediction".
    """

    plt.scatter(df[label_key], df[label_key], label=label)
    plt.scatter(df[label_key], df[prediction_key], c='r', label=prediction)

    plt.legend()

    plt.show()

def plot_training_test_validat_accuracy(
        dfs: list[pd.DataFrame],
        label_key: str,
        label: str = "Label",
        prediction_key: str = "prediction",
        prediction: str = "Prediction",
        titles = ["Training data", "Validation data", "Test data"],
        columns = 3):
    """Plots the accuracy of the three different data sets.

    Args:
        df (pd.DataFrame): the dataframe that contains the data, order should be [training, validation, test]
        label_key (str): the expected output/label column's key in the dataframe
        label (str, optional): the expected output's user friendly name. Defaults to "Label".
        prediction_key (str, optional): the prediction column's key in the dataframe. Defaults to "prediction".
        prediction (str, optional): the prediction's user friendly name. Defaults to "Prediction".
    """
    fig, ax = plt.subplots(1,columns, figsize=(15, 5))
    for i, df in enumerate(dfs):
        ax[i].scatter(df[label_key], df[label_key], label=label)
        ax[i].scatter(df[label_key], df[prediction_key], c='r', label=prediction)
        ax[i].set_title(titles[i])
        