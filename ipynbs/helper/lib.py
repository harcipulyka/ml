import pandas as pd
import numpy as np

def create_inference_sample(df : pd.DataFrame, features, model, n = 200):
    """Creates a sample dataframe that contains predicted values

    Args:
        df (pd.DataFrame): the dataframe that contains the features and labels
        features (_type_): the features to use with the model
        model (_type_): the model to use
        n (_type_): number of samples. Defaults to 200.

    Returns:
        _type_: _description_
    """
    sample_df = df.sample(n=n, ignore_index=True)

    features = sample_df[features]
    predictions = model.predict(features).ravel()
    sample_df["prediction"] = predictions

    return sample_df


def shuffle_dataframe(df : pd.DataFrame):
    #shuffle database (in place, no new dataframe)
    return df.sample(frac=1).reset_index(drop=True)


def split_dataframe_train_validate_test(df : pd.DataFrame, train = 0.7, validate = 0.2, test = 0.1, seed = None):
    """Split a dataframe into train, validate and test sets. Proportion can be defined, but has to add together to one.
    The method also serves as a shuffle, because the fractions are handled randomly with df.sample()

    Args:
        df (DataFrame): The dataframe to split
        train (float, optional): Training set ratio. Defaults to 0.7.
        validate (float, optional): Validation set ratio. Defaults to 0.2.
        test (float, optional): Test set ratio. Defaults to 0.1.
    """

    if(abs((train + validate + test) - 1.0) > 0.00001):
        raise Exception("The ratios have to add up to 1, but they add up to ({})", (train+validate+test))

    original_length = len(df.index)

    training_df = df.sample(frac=train, random_state=seed)
    rest = df.drop(training_df.index)

    #determining fraction for the rest (validate / test)
    validate = 1 / (validate + test) * validate

    validate_df = rest.sample(frac=validate, random_state=seed)
    test_df = rest.drop(validate_df.index)

    new_length = len(training_df.index) + len(validate_df.index) + len(test_df.index)

    if(new_length != original_length):
        raise Exception("New dataframes length doesn't add up to the original. Original: {}, new: {}", original_length, new_length)
    elif(len(training_df.index) < 3 or len(validate_df.index) < 3 or len(test_df.index) < 3):
        raise Exception("One of the new dataframes has less than 3 entries")
    
    return training_df, validate_df, test_df
