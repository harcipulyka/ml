import pandas as pd
from heatmaps import *
from scatter_plot import *

#TODO probably should modify the original, dunno tho
def sample_and_predict(df : pd.DataFrame, features, model, n = 200) -> pd.DataFrame:
    """Samples the training data and uses the model to infer values to features

    Args:
        df (pd.DataFrame): the original datafram
        features (_type_): the (list of) column name(s) that define the feature(s) 
        model (_type_): the tensorflow model
        n (int, optional): the sample size. Defaults to 200.

    Returns:
        pd.DataFrame: the modified dataframe with the predicted values in column "prediction"
    """

    sample = df.sample(n=n, ignore_index=True)
    features = df[features]
    predictions = model.predict(features).ravel()
    sample["prediction"] = predictions

    return sample