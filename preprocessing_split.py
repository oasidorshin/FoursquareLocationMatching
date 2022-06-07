import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import OrdinalEncoder


from utils import *
from feature_utils import clean_string, get_shingles


def group_split(df):
    gkf = GroupKFold(n_splits=2)
    splits = list(gkf.split(
        df, groups=df["point_of_interest"]))

    return df.iloc[splits[0][1]], df.iloc[splits[1][1]]


def preprocessing(df, encoder=None):
    df = df.set_index("id", drop=False)

    # Name cleaning
    df = clean_string(df, "name", "name_cleaned")

    # Name shingles
    df = get_shingles(df, "name_cleaned", (2, 3))

    # Categories to frozenset
    df["categories"] = df["categories"].fillna("None")
    df["categories"] = df["categories"].apply(lambda x: x.split(", "))
    df["categories"] = df["categories"].apply(frozenset)

    # Fill missing country
    df["country"] = df["country"].fillna("None")

    # Encode categorical columns
    if encoder is None:
        # No encoders provided, create and save
        encoder_params = {"dtype": np.int32,
                          "handle_unknown": "use_encoded_value",
                          "unknown_value": -1}

        ordinal_encoder = OrdinalEncoder(**encoder_params)

        ordinal_encoder = ordinal_encoder.fit(df[["country", "categories"]])
        pickle_save(ordinal_encoder, "saved/ordinal_encoder.pkl")
        encoder = ordinal_encoder

    df[["country", "categories"]] = encoder.transform(
        df[["country", "categories"]])

    return df


if __name__ == "__main__":
    train_with_target = pd.read_pickle("saved/train_with_target.pkl")
    train_with_target = preprocessing(train_with_target)

    fold0_df, fold1_df = group_split(train_with_target)

    pickle_save(fold0_df, "saved/fold0_df.pkl")
    pickle_save(fold1_df, "saved/fold1_df.pkl")
