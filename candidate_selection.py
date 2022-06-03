import numpy as np
import pandas as pd

from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from utils import *
from feature_utils import overlap


def country_closest_k(train_df, country, candidate_k):
    country_df = train_df[train_df["country"] == country]

    # Coordinates
    country_np = np.deg2rad(country_df[["latitude", "longitude"]].to_numpy())

    # To 3d
    country_np = np.vstack([(np.cos(country_np[:, 0]) * np.cos(country_np[:, 1])),
                            (np.cos(country_np[:, 0]) *
                             np.sin(country_np[:, 1])),
                            (np.sin(country_np[:, 0]))]).T

    neigh = NearestNeighbors(n_jobs=-1).fit(country_np)
    try:
        distances, neighbors_indices = neigh.kneighbors(
            country_np, n_neighbors=candidate_k, return_distance=True)
    except:
        # Handle Expected n_neighbors <= n_samples error
        # Add all but exclude itself
        neighbors_indices = [
            [i for i in range(len(country_df)) if i != j] for j in range(len(country_df))]
        neighbors_indices = np.array(neighbors_indices, dtype=int)

    # Convert indices to id
    ids = country_df["id"].to_numpy()
    neighbors_ids = pd.Series(list(neighbors_indices), index=country_df.index).apply(
        lambda candidate_indices: ids[candidate_indices])

    return neighbors_ids


def candidate_selection(train_df, candidate_k):
    train_df["k_candidates"] = pd.Series(dtype='object')
    uq_countries = train_df["country"].value_counts().index

    for country in tqdm(uq_countries):
        train_df.loc[train_df["country"] == country, "k_candidates"] = \
            country_closest_k(train_df, country, candidate_k)

    # Empty candidates
    for row in train_df.loc[train_df["k_candidates"].isnull(), "k_candidates"].index:
        train_df.at[row, "k_candidates"] = []

    return train_df


def forming_pairs_filtering(train_df, th):
    pairs = []
    dict_ = train_df["name_cleaned_shingles_3"].to_dict()

    for p1_idx in tqdm(train_df.index):
        for p2_idx in train_df.loc[p1_idx, "k_candidates"]:
            if p1_idx == p2_idx:  # Skip
                continue

            try:
                sim = overlap(dict_[p1_idx], dict_[p2_idx])
                if sim >= th:
                    pairs.append([p1_idx, p2_idx])
            except:
                pass

    return pd.DataFrame(pairs, columns=["p1", "p2"])


if __name__ == "__main__":

    for fold in [0, 1]:
        fold_df = pickle_load(f"saved/fold{fold}_df.pkl")
        fold_df = candidate_selection(fold_df, 160)

        pairs = forming_pairs_filtering(fold_df, 0.1)

        pickle_save(pairs, f"saved/pairs_fold{fold}.pkl")
