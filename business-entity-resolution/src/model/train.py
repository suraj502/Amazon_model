"""
Define model training, entity-level cross-validation,
feature ablation and hard negative mining.
"""

from typing import Any

import os
import joblib
import numpy as np
import pandas as pd

import lightgbm as lgb

from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    fbeta_score,
    precision_score,
    recall_score
)

from src.utils.config_loader import CONFIG, get_config_value


MODEL_TYPE = get_config_value(
    CONFIG,
    "model",
    "type"
)



def evaluate_model(
    X,
    y,
    config=CONFIG
):
    """
    Train and evaluate model on given features.
    """


    params = get_config_value(
        config,
        "model",
        "params"
    )


    model = lgb.LGBMClassifier(
        **params
    )


    model.fit(
        X,
        y
    )


    probabilities = model.predict_proba(
        X
    )[:, 1]


    predictions = (
        probabilities >= 0.5
    ).astype(int)



    return {

        "f0.5":
            fbeta_score(
                y,
                predictions,
                beta=0.5
            ),


        "precision":
            precision_score(
                y,
                predictions,
                zero_division=0
            ),


        "recall":
            recall_score(
                y,
                predictions,
                zero_division=0
            )

    }





def train_model(
    features: pd.DataFrame,
    labels: pd.Series,
    config: dict[str, Any] = CONFIG
):
    """
    Train LightGBM model using entity-level CV.
    """


    entity_column = None


    for col in [

        "source1_entity_id",

        "entity_id"

    ]:

        if col in features.columns:

            entity_column = col
            break



    if entity_column is None:

        raise ValueError(
            "Entity id column missing"
        )



    X = features.drop(

        columns=[

            entity_column,

            "label"

        ],

        errors="ignore"

    )


    groups = features[entity_column]



    params = get_config_value(
        config,
        "model",
        "params"
    )


    folds = get_config_value(
        config,
        "model",
        "cv",
        "n_folds"
    )



    cv = GroupKFold(
        n_splits=folds
    )


    scores = []



    for train_idx, val_idx in cv.split(

        X,

        labels,

        groups

    ):


        train_entities = set(
            groups.iloc[train_idx]
        )


        val_entities = set(
            groups.iloc[val_idx]
        )


        assert train_entities.isdisjoint(
            val_entities
        )



        model = lgb.LGBMClassifier(
            **params
        )


        model.fit(

            X.iloc[train_idx],

            labels.iloc[train_idx]

        )


        val_prob = model.predict_proba(

            X.iloc[val_idx]

        )[:,1]


        val_pred = (

            val_prob >= 0.5

        ).astype(int)



        scores.append({

            "f0.5":
                fbeta_score(
                    labels.iloc[val_idx],
                    val_pred,
                    beta=0.5
                ),

            "precision":
                precision_score(
                    labels.iloc[val_idx],
                    val_pred,
                    zero_division=0
                ),

            "recall":
                recall_score(
                    labels.iloc[val_idx],
                    val_pred,
                    zero_division=0
                )

        })



    os.makedirs(
        "reports",
        exist_ok=True
    )


    pd.DataFrame(scores).to_csv(

        "reports/model_cv_report.csv",

        index=False

    )



    final_model = lgb.LGBMClassifier(
        **params
    )


    final_model.fit(

        X,

        labels

    )



    os.makedirs(
        "artifacts",
        exist_ok=True
    )


    joblib.dump(

        final_model,

        "artifacts/m3_model.pkl"

    )


    return final_model





def run_feature_ablation(
    features,
    labels
):
    """
    Compare feature groups.
    """


    experiments = {


        "string_only":[

            "name_jaccard",

            "name_levenshtein",

            "name_jaro_winkler",

            "address_jaccard",

            "address_levenshtein",

            "address_jaro_winkler",

            "country_match",

            "phonetic_match"

        ],



        "string_tfidf":[

            "name_jaccard",

            "name_levenshtein",

            "name_jaro_winkler",

            "address_jaccard",

            "address_levenshtein",

            "address_jaro_winkler",

            "country_match",

            "phonetic_match",

            "name_tfidf_cosine",

            "address_tfidf_cosine"

        ],



        "string_tfidf_embedding":[

            "name_tfidf_cosine",

            "address_tfidf_cosine",

            "name_embedding_cosine",

            "address_embedding_cosine",

            "combined_embedding_cosine"

        ],



        "all_features":

            list(
                features.select_dtypes(
                    include=np.number
                ).columns
            )

    }



    results = []



    for name, columns in experiments.items():


        available = [

            c for c in columns

            if c in features.columns

        ]


        result = evaluate_model(

            features[available],

            labels

        )


        result["experiment"] = name


        results.append(
            result
        )



    report = pd.DataFrame(
        results
    )


    os.makedirs(
        "reports",
        exist_ok=True
    )


    report.to_csv(

        "reports/feature_ablation.csv",

        index=False

    )


    return report





def find_hard_negatives(

    model,

    features,

    labels,

    threshold=0.7

):
    """
    Find high confidence false positives.
    """


    probabilities = model.predict_proba(

        features

    )[:,1]



    mask = (

        (probabilities > threshold)

        &

        (labels == 0)

    )



    hard_negatives = features[
        mask
    ].copy()



    hard_negatives["label"] = 0



    return hard_negatives





def save_hard_negative_report(
    hard_negatives
):

    os.makedirs(

        "reports",

        exist_ok=True

    )


    with open(

        "reports/hard_negative_report.md",

        "w"

    ) as f:


        f.write(

f"""
# Hard Negative Mining Report

Number of hard negatives:

{len(hard_negatives)}

"""

        )