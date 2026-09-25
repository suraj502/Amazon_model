"""Define prediction and match selection contracts for trained models."""

from typing import Any

import os
import joblib
import pandas as pd
import numpy as np

from src.utils.config_loader import CONFIG, get_config_value


ID_COLUMN = get_config_value(
    CONFIG,
    "schema",
    "id_column"
)



def load_model(
    path: str = "artifacts/m3_model.pkl"
):
    """
    Load trained M3 model.
    """

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"Model not found: {path}"
        )


    return joblib.load(
        path
    )





def predict_matches(
    model: Any,
    features: pd.DataFrame,
    config: dict[str, Any] = CONFIG
) -> pd.DataFrame:
    """
    Predict candidate match probabilities.

    Output:

    source1_entity_id
    candidate_entity_id
    match_probability
    """



    df = features.copy()



    # Keep identifiers

    id_columns = [

        col for col in [

            "source1_entity_id",

            "candidate_entity_id",

            "entity_id"

        ]

        if col in df.columns

    ]



    ids = df[id_columns].copy()



    X = df.drop(

        columns=id_columns,

        errors="ignore"

    )



    # Remove label during prediction

    X = X.drop(

        columns=["label"],

        errors="ignore"

    )



    probabilities = model.predict_proba(

        X

    )[:,1]



    ids["match_probability"] = (

        probabilities.astype(
            np.float32
        )

    )


    return ids