from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score

from utils import *

if __name__ == "__main__":
    for (fold_train, fold_val) in [(0, 1), (1, 0)]:
        train_df = pickle_load(f"saved/pairs_features_fold{fold_train}.pkl")
        val_df = pickle_load(f"saved/pairs_features_fold{fold_val}.pkl")

        cat_features = ["country", "categories1", "categories2"]
        num_features = [x for x in train_df.columns if x not in ['p1', 'p2', 'match'] + cat_features]
        target = "match"

        cbc = CatBoostClassifier(iterations=500,
                                 l2_leaf_reg=10,
                                 model_size_reg=1,  # Without those params memory explodes when saving
                                 max_ctr_complexity=2,
                                 eval_metric="PRAUC",
                                 task_type="GPU",
                                 learning_rate=0.2,
                                 depth=10,
                                 verbose=10,
                                 thread_count=-1,
                                 metric_period=10)
        # Train
        cbc.fit(train_df[num_features + cat_features],
                y=train_df[target],
                eval_set=(val_df[num_features + cat_features], val_df[target]),
                cat_features=cat_features)

        # Validate
        val_df["predict_proba"] = cbc.predict_proba(val_df[num_features + cat_features])[:, 1]

        pickle_save(val_df["predict_proba"].to_numpy(), f"saved/cb_outoffold{fold_val}.pkl")

        print("MAP:", average_precision_score(val_df["match"], val_df["predict_proba"]))

        full_val_df = pickle_load(f"saved/fold{fold_val}_df.pkl")
        th = 0.5
        prediction = val_df[val_df["predict_proba"] > th][["p1", "p2"]].groupby("p1").agg(set)
        full_val_df["prediction"] = prediction

        # Fill empty
        for row in full_val_df.loc[full_val_df["prediction"].isnull(), "prediction"].index:
            full_val_df.at[row, "prediction"] = set()

        # Add itself
        full_val_df.apply(lambda x: x["prediction"].add(x["id"]), axis=1)

        print("Jaccard:", jaccard_score(full_val_df["id_target"], full_val_df["prediction"]))

        # Save model
        pickle_save(cbc, f"saved/cbc_fold{fold_train}.pkl")
