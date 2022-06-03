import numpy as np
import pandas as pd

import string
import math

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


def overlap(profile0, profile1):
    union = set()
    for k in profile0.keys():
        union.add(k)
    for k in profile1.keys():
        union.add(k)
    inter = int(len(profile0.keys()) + len(profile1.keys()) - len(union))
    return inter / min(len(profile0), len(profile1))


def jaccard(profile0, profile1):
    union = set()
    for ite in profile0.keys():
        union.add(ite)
    for ite in profile1.keys():
        union.add(ite)
    inter = int(len(profile0.keys()) + len(profile1.keys()) - len(union))
    return 1.0 * inter / len(union)


def cosine(profile0, profile1):
    small = profile1
    large = profile0
    if len(profile0) < len(profile1):
        small = profile0
        large = profile1
    agg = 0.0
    for k, v in small.items():
        i = large.get(k)
        if not i:
            continue
        agg += 1.0 * v * i
    dot_product = agg

    agg = 0.0
    for k, v in profile0.items():
        agg += 1.0 * v * v
    profile0_norm = math.sqrt(agg)

    agg = 0.0
    for k, v in profile1.items():
        agg += 1.0 * v * v
    profile1_norm = math.sqrt(agg)

    return dot_product / (profile0_norm * profile1_norm)
