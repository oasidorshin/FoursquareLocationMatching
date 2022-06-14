import numpy as np
import pandas as pd

import lightgbm
from lightgbm import LGBMClassifier, LGBMRanker
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold

from utils import *

if __name__ == "__main__":
    for (fold_train, fold_val) in [(0, 1), (1, 0)]:
        train_df = pickle_load(f"saved/pairs_features_fold{fold_train}.pkl")
        val_df = pickle_load(f"saved/pairs_features_fold{fold_val}.pkl")

        cat_features = ["country", "categories1", "categories2"]
        num_features = [x for x in train_df.columns
                        if x not in ['p1', 'p2', 'match'] + cat_features]
        target = "match"

        main_params = {'reg_alpha': 0,
                       'reg_lambda': 5,
                       'max_depth': 64,
                       'num_leaves': 256,
                       'learning_rate': 0.05,
                       'n_estimators': 1000,
                       'min_child_samples': 20,
                       'subsample': 0.5,
                       'colsample_bytree': 0.5,
                       'colsample_bynode': 1,
                       'early_stopping_round': 50
                       }

        aux_params = {'importance_type': 'gain',
                      'random_state': 42,
                      'subsample_freq': 1,
                      'is_unbalance': False
                      }

        lgbmc = LGBMClassifier(
            **main_params, **aux_params)

        # Train
        lgbmc.fit(train_df[num_features + cat_features],
                  train_df[target], verbose=10,
                  categorical_feature=cat_features,
                  eval_set=(
                      val_df[num_features + cat_features], val_df[target]),
                  eval_metric="average_precision")

        # Validate
        val_df["predict_proba"] = lgbmc.predict_proba(
            val_df[num_features + cat_features])[:, 1]
        print("MAP:",
              average_precision_score(val_df["match"], val_df["predict_proba"]))

        full_val_df = pickle_load(f"saved/fold{fold_val}_df.pkl")
        th = 0.5
        prediction = val_df[
            val_df["predict_proba"] > th][["p1", "p2"]].groupby("p1").agg(set)
        full_val_df["prediction"] = prediction

        # Fill empty
        for row in full_val_df.loc[full_val_df["prediction"].isnull(), "prediction"].index:
            full_val_df.at[row, "prediction"] = set()

        # Add itself
        full_val_df.apply(lambda x: x["prediction"].add(x["id"]), axis=1)

        # TODO better inference

        print("Jaccard:", jaccard_score(
            full_val_df["id_target"], full_val_df["prediction"]))

        # Save model
        pickle_save(lgbmc, f"saved/lgbmc_fold{fold_train}.pkl")
