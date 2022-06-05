import numpy as np
import pandas as pd

import sys

from utils import *
from feature_utils import haversine_vec, get_shingle_similarity

from tqdm import tqdm


def feature_engineering(train_df, pairs, add_target=True):

    # Candidates count
    if "count1" not in pairs.columns:
        pairs["count1"] = pairs.groupby("p1")["p1"].transform("count")
        pairs["count1"] = pairs["count1"].astype(np.int32)

    if "count2" not in pairs.columns:
        pairs["count2"] = pairs.groupby("p2")["p2"].transform("count")
        pairs["count2"] = pairs["count2"].astype(np.int32)

    # Haversine
    if "haversine" not in pairs.columns:
        pairs["haversine"] = haversine_vec(
            train_df.loc[pairs["p1"], "latitude"],
            train_df.loc[pairs["p1"], "longitude"],
            train_df.loc[pairs["p2"], "latitude"],
            train_df.loc[pairs["p2"], "longitude"])
        pairs["haversine"] = pairs["haversine"].astype(np.float16)

    # Name similarity
    for name in ["jaccard", "overlap", "cosine"]:
        for k in tqdm([2, 3]):
            feature_name = f"name_cleaned_{name}_{k}"
            if feature_name not in pairs.columns:
                similarity = get_shingle_similarity(name)
                pairs[feature_name] = similarity(train_df.loc[pairs["p1"], f"name_cleaned_shingles_{k}"],
                                                 train_df.loc[pairs["p2"], f"name_cleaned_shingles_{k}"])
                pairs[feature_name] = pairs[feature_name].astype(np.float16)

    # Category
    if "categories1" not in pairs.columns:
        pairs["categories1"] = train_df.loc[pairs["p1"],
                                            "categories"].reset_index(drop=True)
        pairs["categories1"] = pairs["categories1"].astype(np.int32)

    if "categories2" not in pairs.columns:
        pairs["categories2"] = train_df.loc[pairs["p2"],
                                            "categories"].reset_index(drop=True)
        pairs["categories2"] = pairs["categories2"].astype(np.int32)

    # Country (same for every pair)
    if "country" not in pairs.columns:
        pairs["country"] = train_df.loc[pairs["p1"],
                                        "country"].reset_index(drop=True)
        pairs["country"] = pairs["country"].astype(np.int32)

    # Target
    if add_target:
        if "match" not in pairs.columns:
            matches = []
            for row in tqdm(pairs.itertuples()):
                if row.p1 in train_df.loc[row.p2, "id_target"]:
                    matches.append(1)
                else:
                    matches.append(0)

            pairs["match"] = matches
            pairs["match"] = pairs["match"].astype(np.int32)

    return pairs


if __name__ == "__main__":
    for fold in [0, 1]:
        fold_df = pickle_load(f"saved/fold{fold}_df.pkl")
        fold_pairs = pickle_load(f"saved/pairs_fold{fold}.pkl")

        fold_pairs_features = feature_engineering(fold_df, fold_pairs)

        print(fold_pairs_features)

        print(
            f"Total size of fold_pairs_features: {(sys.getsizeof(fold_pairs_features) / 1024**2):.2f} MB")

        pickle_save(fold_pairs_features,
                    f"saved/pairs_features_fold{fold}.pkl")
