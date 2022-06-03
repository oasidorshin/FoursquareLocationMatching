import numpy as np
import pandas as pd

import string

from unidecode import unidecode
from strsimpy.shingle_based import ShingleBased


def apply_notnull(df, column, target_column, function):
    df.loc[df[column].notnull(), target_column] = \
        df.loc[df[column].notnull(), column].apply(function)

    return df


def clean_string(df, column, target_column):
    # Unidecode
    df = apply_notnull(df, column, target_column, lambda x: unidecode(x))

    # Replace AND, AT
    df = apply_notnull(df, target_column, target_column, lambda x: x.translate(
        str.maketrans({"@": "at", "&": "and"})))

    # Strip punctuation
    df = apply_notnull(df, target_column, target_column, lambda x: x.translate(
        str.maketrans('', '', string.punctuation)))

    # To lowercase
    df = apply_notnull(df, target_column, target_column, lambda x: x.lower())

    # Remove leading spaces
    df = apply_notnull(df, target_column, target_column, lambda x: x.strip())

    return df


def get_shingles(df, column, shingle_k):
    for k in shingle_k:
        sh = ShingleBased(k=k)
        df = apply_notnull(df, column,
                           f"{column}_shingles_{k}", lambda x: sh.get_profile(x))

    return df
