import numpy as np
import pandas as pd

import sys

from utils import *
from feature_utils import haversine_vec, get_shingle_similarity

from sklearn.feature_extraction.text import TfidfVectorizer

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
        pairs["haversine"] = pairs["haversine"].astype(np.float32)

    # Name similarity
    for name in ["jaccard", "overlap", "cosine"]:
        for k in tqdm([2, 3]):
            feature_name = f"name_cleaned_{name}_{k}"
            if feature_name not in pairs.columns:
                similarity = get_shingle_similarity(name)
                pairs[feature_name] = similarity(train_df.loc[pairs["p1"], f"name_cleaned_shingles_{k}"],
                                                 train_df.loc[pairs["p2"], f"name_cleaned_shingles_{k}"])
                pairs[feature_name] = pairs[feature_name].astype(np.float16)

    # Full address similarity
    for name in ["jaccard", "overlap"]:
        for k in tqdm([3]):
            feature_name = f"full_address_{name}_{k}"
            if feature_name not in pairs.columns:
                similarity = get_shingle_similarity(name)
                pairs[feature_name] = similarity(train_df.loc[pairs["p1"], f"full_address_cleaned_shingles_{k}"],
                                                 train_df.loc[pairs["p2"], f"full_address_cleaned_shingles_{k}"])
                pairs[feature_name] = pairs[feature_name].astype(np.float16)

    # Name-address similarity
    for name in ["overlap"]:
        for k in tqdm([3]):
            feature_name = f"name_address_{name}_{k}"
            if feature_name not in pairs.columns:
                similarity = get_shingle_similarity(name)
                pairs[feature_name] = similarity(train_df.loc[pairs["p1"], f"name_cleaned_shingles_{k}"],
                                                 train_df.loc[pairs["p2"], f"full_address_cleaned_shingles_{k}"])

                pairs[feature_name] += similarity(train_df.loc[pairs["p1"], f"full_address_cleaned_shingles_{k}"],
                                                  train_df.loc[pairs["p2"], f"name_cleaned_shingles_{k}"])
                pairs[feature_name] = pairs[feature_name] / 2

                pairs[feature_name] = pairs[feature_name].astype(np.float16)

    # Numbers in name similarity
    for name in ["overlap"]:
        feature_name = f"numbers_in_name_{name}"
        if feature_name not in pairs.columns:
            similarity = get_shingle_similarity(name)
            pairs[feature_name] = similarity(train_df.loc[pairs["p1"], f"numbers_in_name_shingles_1"],
                                             train_df.loc[pairs["p2"], f"numbers_in_name_shingles_1"])
            pairs[feature_name] += similarity(train_df.loc[pairs["p1"], f"numbers_in_name_shingles_2"],
                                              train_df.loc[pairs["p2"], f"numbers_in_name_shingles_2"])
            pairs[feature_name] = pairs[feature_name] / 2

            pairs[feature_name] = pairs[feature_name].astype(np.float16)

    # Numbers in address similarity
    for name in ["overlap"]:
        feature_name = f"numbers_in_address_{name}"
        if feature_name not in pairs.columns:
            similarity = get_shingle_similarity(name)
            pairs[feature_name] = similarity(train_df.loc[pairs["p1"], f"numbers_in_full_address_shingles_1"],
                                             train_df.loc[pairs["p2"], f"numbers_in_full_address_shingles_1"])
            pairs[feature_name] += similarity(train_df.loc[pairs["p1"], f"numbers_in_full_address_shingles_2"],
                                              train_df.loc[pairs["p2"], f"numbers_in_full_address_shingles_2"])
            pairs[feature_name] = pairs[feature_name] / 2

            pairs[feature_name] = pairs[feature_name].astype(np.float16)

    # Numbers in name-address similarity
    for name in ["overlap"]:
        feature_name = f"numbers_in_name_address_{name}"
        if feature_name not in pairs.columns:
            similarity = get_shingle_similarity(name)
            pairs[feature_name] = similarity(train_df.loc[pairs["p1"], f"numbers_in_name_shingles_1"],
                                             train_df.loc[pairs["p2"], f"numbers_in_full_address_shingles_1"])
            pairs[feature_name] += similarity(train_df.loc[pairs["p1"], f"numbers_in_full_address_shingles_1"],
                                              train_df.loc[pairs["p2"], f"numbers_in_name_shingles_1"])
            pairs[feature_name] += similarity(train_df.loc[pairs["p1"], f"numbers_in_name_shingles_2"],
                                              train_df.loc[pairs["p2"], f"numbers_in_full_address_shingles_2"])
            pairs[feature_name] += similarity(train_df.loc[pairs["p1"], f"numbers_in_full_address_shingles_2"],
                                              train_df.loc[pairs["p2"], f"numbers_in_name_shingles_2"])
            pairs[feature_name] = pairs[feature_name] / 4

            pairs[feature_name] = pairs[feature_name].astype(np.float16)

    # Category
    if "categories1" not in pairs.columns:
        pairs["categories1"] = train_df.loc[pairs["p1"], "categories_enc"].reset_index(drop=True)
        pairs["categories1"] = pairs["categories1"].astype(np.int32)

    if "categories2" not in pairs.columns:
        pairs["categories2"] = train_df.loc[pairs["p2"], "categories_enc"].reset_index(drop=True)
        pairs["categories2"] = pairs["categories2"].astype(np.int32)

    # Categories text similarity
    for name in ["overlap", "jaccard"]:
        feature_name = f"categories_{name}"
        if feature_name not in pairs.columns:
            similarity = get_shingle_similarity(name)
            pairs[feature_name] = similarity(train_df.loc[pairs["p1"], f"categories_shingles_3"],
                                             train_df.loc[pairs["p2"], f"categories_shingles_3"])

            pairs[feature_name] = pairs[feature_name].astype(np.float16)

    # Country (same for every pair)
    if "country" not in pairs.columns:
        pairs["country"] = train_df.loc[pairs["p1"], "country_enc"].reset_index(drop=True)
        pairs["country"] = pairs["country"].astype(np.int32)

    # Country-based TF-IDF features
    for country in tqdm(train_df["country_enc"].unique()):
        country_df = train_df[train_df["country_enc"] == country]
        country_df["country_index"] = np.arange(len(country_df)).astype(int)

        country_pairs = pairs[pairs["country"] == country]
        country_pairs["country_index1"] = country_df.loc[country_pairs["p1"], "country_index"].to_numpy()
        country_pairs["country_index2"] = country_df.loc[country_pairs["p2"], "country_index"].to_numpy()

        index1 = country_pairs["country_index1"].to_numpy()
        index2 = country_pairs["country_index2"].to_numpy()

        for column in ["name_cleaned", "full_address_cleaned"]:
            try:
                vectorizer_words = TfidfVectorizer()
                vectorizer_trigrams = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3))
                words_matrix = vectorizer_words.fit_transform(country_df[column].fillna(""))
                trigrams_matrix = vectorizer_trigrams.fit_transform(country_df[column].fillna(""))
            except:
                continue

            pairs.loc[pairs["country"] == country, f"tfidf_trigram_{column}"] = \
                np.sum(trigrams_matrix[index1].multiply(trigrams_matrix[index2]), axis=1)

            pairs.loc[pairs["country"] == country, f"tfidf_words_{column}"] = \
                np.sum(words_matrix[index1].multiply(words_matrix[index2]), axis=1)

            pairs[f"tfidf_trigram_{column}"] = pairs[f"tfidf_trigram_{column}"].astype(np.float16)

            pairs[f"tfidf_words_{column}"] = pairs[f"tfidf_words_{column}"].astype(np.float16)

    # Group-by features
    # Haversine features
    groupby_p1 = pairs.groupby('p1')["haversine"]
    groupby_p2 = pairs.groupby('p2')["haversine"]
    pairs[f"p1_haversine_mean"] = groupby_p1.transform(np.mean).astype(np.float32)
    pairs[f"p2_haversine_mean"] = groupby_p2.transform(np.mean).astype(np.float32)
    pairs[f"p1_haversine_min"] = groupby_p1.transform(np.min).astype(np.float32)
    pairs[f"p2_haversine_min"] = groupby_p2.transform(np.min).astype(np.float32)
    pairs[f"p1_haversine_max"] = groupby_p1.transform(np.max).astype(np.float32)
    pairs[f"p2_haversine_max"] = groupby_p2.transform(np.max).astype(np.float32)

    pairs[f"p1_haversine_rank"] = ((groupby_p1.transform(
        "rank", method="min") - 1) / pairs["count1"]).astype(np.float16)
    pairs[f"p2_haversine_rank"] = ((groupby_p2.transform(
        "rank", method="min") - 1) / pairs["count2"]).astype(np.float16)

    # Name features
    for feature in ["name_cleaned_overlap_3", "tfidf_trigram_name_cleaned"]:
        groupby_p1 = pairs.groupby('p1')[feature]
        groupby_p2 = pairs.groupby('p2')[feature]
        pairs[f"p1_{feature}_mean"] = groupby_p1.transform(np.mean).astype(np.float16)
        pairs[f"p2_{feature}_mean"] = groupby_p2.transform(np.mean).astype(np.float16)
        pairs[f"p1_{feature}_max"] = groupby_p1.transform(np.max).astype(np.float16)
        pairs[f"p2_{feature}_max"] = groupby_p2.transform(np.max).astype(np.float16)

        pairs[f"p1_{feature}_rank"] = ((groupby_p1.transform(
            "rank", method="min") - 1) / pairs["count1"]).astype(np.float16)
        pairs[f"p2_{feature}_rank"] = ((groupby_p2.transform(
            "rank", method="min") - 1) / pairs["count2"]).astype(np.float16)

    # Address/numbers features: only mean
    for feature in ["full_address_overlap_3",
                    "numbers_in_name_overlap",
                    "numbers_in_address_overlap",
                    "numbers_in_name_address_overlap",
                    "categories_overlap",
                    "categories_jaccard"]:

        groupby_p1 = pairs.groupby('p1')[feature]
        groupby_p2 = pairs.groupby('p2')[feature]
        pairs[f"p1_{feature}_mean"] = groupby_p1.transform(np.mean).astype(np.float16)
        pairs[f"p2_{feature}_mean"] = groupby_p2.transform(np.mean).astype(np.float16)

    # Count features
    if "lon_lat_round1_perc1" not in pairs.columns:
        pairs["lon_lat_round1_perc1"] = train_df.loc[pairs["p1"], "lon_lat_round1_perc"].to_numpy()
        pairs["lon_lat_round1_perc1"] = pairs["lon_lat_round1_perc1"].astype(np.float16)

    if "lon_lat_round1_perc2" not in pairs.columns:
        pairs["lon_lat_round1_perc2"] = train_df.loc[pairs["p2"], "lon_lat_round1_perc"].to_numpy()
        pairs["lon_lat_round1_perc2"] = pairs["lon_lat_round1_perc2"].astype(np.float16)

    if "lon_lat_round0_perc1" not in pairs.columns:
        pairs["lon_lat_round0_perc1"] = train_df.loc[pairs["p1"], "lon_lat_round0_perc"].to_numpy()
        pairs["lon_lat_round0_perc1"] = pairs["lon_lat_round0_perc1"].astype(np.float16)

    if "lon_lat_round0_perc2" not in pairs.columns:
        pairs["lon_lat_round0_perc2"] = train_df.loc[pairs["p2"], "lon_lat_round0_perc"].to_numpy()
        pairs["lon_lat_round0_perc2"] = pairs["lon_lat_round0_perc2"].astype(np.float16)

    if "name_cleaned_perc1" not in pairs.columns:
        pairs["name_cleaned_perc1"] = train_df.loc[pairs["p1"], "name_cleaned_perc"].to_numpy()
        pairs["name_cleaned_perc1"] = pairs["name_cleaned_perc1"].astype(np.float16)

    if "name_cleaned_perc2" not in pairs.columns:
        pairs["name_cleaned_perc2"] = train_df.loc[pairs["p2"], "name_cleaned_perc"].to_numpy()
        pairs["name_cleaned_perc2"] = pairs["name_cleaned_perc2"].astype(np.float16)

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

        print(f"Total size of fold_pairs_features: {(sys.getsizeof(fold_pairs_features) / 1024**2):.2f} MB")

        pickle_save(fold_pairs_features, f"saved/pairs_features_fold{fold}.pkl")
