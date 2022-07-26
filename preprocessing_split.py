import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import OrdinalEncoder


from utils import *
from feature_utils import clean_string, get_shingles, apply_notnull, get_numbers_from_name


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

    # Full address
    df["full_address"] = df["address"].fillna("") +\
        " " + df["city"].fillna("") +\
        " " + df["state"].fillna("")

    df.loc[df["full_address"] == "  ", "full_address"] = np.NaN
    df = clean_string(df, "full_address", "full_address_cleaned")
    df = get_shingles(df, "full_address_cleaned", (3,))

    # Numbers in name/address
    df = apply_notnull(df, "name_cleaned", "numbers_in_name", get_numbers_from_name)
    df.loc[df["numbers_in_name"] == "", "numbers_in_name"] = np.NaN

    df = apply_notnull(df, "full_address_cleaned", "numbers_in_full_address", get_numbers_from_name)
    df.loc[df["numbers_in_full_address"] == "", "numbers_in_full_address"] = np.NaN

    df = get_shingles(df, "numbers_in_name", (1, 2))
    df = get_shingles(df, "numbers_in_full_address", (1, 2))

    # Catogories shingles
    df = get_shingles(df, "categories", (3,))

    # Categories to frozenset
    df["categories"] = df["categories"].fillna("None")
    df["categories"] = df["categories"].apply(lambda x: x.split(", "))
    df["categories"] = df["categories"].apply(frozenset)

    # Fill missing country
    df["country"] = df["country"].astype(str)

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

    df[["country_enc", "categories_enc"]] = encoder.transform(df[["country", "categories"]])

    # Count features
    df["lat_round1"] = df["latitude"].round(1).astype(str)
    df["lon_round1"] = df["longitude"].round(1).astype(str)
    df["lon_lat_round1"] = df["lat_round1"] + " " + df["lon_round1"]

    df["lat_round0"] = df["latitude"].round(0).astype(str)
    df["lon_round0"] = df["longitude"].round(0).astype(str)
    df["lon_lat_round0"] = df["lat_round0"] + " " + df["lon_round0"]

    df["lon_lat_round1_perc"] = df.groupby("lon_lat_round1")["id"].transform("count") / len(df) * 100
    df["lon_lat_round0_perc"] = df.groupby("lon_lat_round0")["id"].transform("count") / len(df) * 100

    df["name_cleaned_perc"] = df.groupby("name_cleaned")["id"].transform("count") / len(df) * 100

    return df


if __name__ == "__main__":
    train_with_target = pd.read_pickle("saved/train_with_target.pkl")
    train_with_target = preprocessing(train_with_target)

    fold0_df, fold1_df = group_split(train_with_target)

    pickle_save(fold0_df, "saved/fold0_df.pkl")
    pickle_save(fold1_df, "saved/fold1_df.pkl")
