"""Define embedding feature extraction for candidate pairs."""

from typing import Any

import pandas as pd
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from src.utils.config_loader import CONFIG, get_config_value


EMBEDDING_MODEL = get_config_value(
    CONFIG,
    "features",
    "embedding",
    "model_name"
)


BATCH_SIZE = get_config_value(
    CONFIG,
    "features",
    "embedding",
    "batch_size"
)



_model = None



def load_embedding_model():

    global _model


    if _model is None:

        _model = SentenceTransformer(
            EMBEDDING_MODEL
        )


    return _model





def generate_embeddings(
    texts,
    model
):
    """
    Generate embeddings in batches.
    """


    embeddings = model.encode(

        list(texts),

        batch_size=BATCH_SIZE,

        show_progress_bar=True,

        convert_to_numpy=True

    )


    return embeddings





def build_embedding_features(
    pairs: pd.DataFrame,
    config: dict[str, Any] = CONFIG
) -> pd.DataFrame:
    """
    Build embedding cosine similarity features.
    """


    df = pairs.copy()


    model = load_embedding_model()



    name_s1 = (
        df.get(
            "name_basic_norm_s1",
            pd.Series("", index=df.index)
        )
        .fillna("")
        .astype(str)
    )


    name_s2 = (
        df.get(
            "name_basic_norm_s2",
            pd.Series("", index=df.index)
        )
        .fillna("")
        .astype(str)
    )


    address_s1 = (
        df.get(
            "address_basic_norm_s1",
            pd.Series("", index=df.index)
        )
        .fillna("")
        .astype(str)
    )


    address_s2 = (
        df.get(
            "address_basic_norm_s2",
            pd.Series("", index=df.index)
        )
        .fillna("")
        .astype(str)
    )



    # Name embeddings

    name_embeddings_s1 = generate_embeddings(
        name_s1,
        model
    )


    name_embeddings_s2 = generate_embeddings(
        name_s2,
        model
    )


    name_similarity = cosine_similarity(
        name_embeddings_s1,
        name_embeddings_s2
    ).diagonal()



    # Address embeddings

    address_embeddings_s1 = generate_embeddings(
        address_s1,
        model
    )


    address_embeddings_s2 = generate_embeddings(
        address_s2,
        model
    )


    address_similarity = cosine_similarity(
        address_embeddings_s1,
        address_embeddings_s2
    ).diagonal()



    # Combined text embedding

    combined_s1 = (
        name_s1
        + " "
        + address_s1
    )


    combined_s2 = (
        name_s2
        + " "
        + address_s2
    )


    combined_embeddings_s1 = generate_embeddings(
        combined_s1,
        model
    )


    combined_embeddings_s2 = generate_embeddings(
        combined_s2,
        model
    )


    combined_similarity = cosine_similarity(
        combined_embeddings_s1,
        combined_embeddings_s2
    ).diagonal()



    features = pd.DataFrame({

        "name_embedding_cosine":
            name_similarity,

        "address_embedding_cosine":
            address_similarity,

        "combined_embedding_cosine":
            combined_similarity

    })



    return pd.concat(

        [

            df[["entity_id"]]
            if "entity_id" in df.columns
            else pd.DataFrame(index=df.index),


            features.astype(
                np.float32
            )

        ],

        axis=1

    )