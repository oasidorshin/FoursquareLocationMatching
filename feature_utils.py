import numpy as np
import pandas as pd

import string
import math
import re

from unidecode import unidecode
from strsimpy.shingle_based import ShingleBased
from haversine import haversine


def apply_notnull(df, column, target_column, function):
    """Utility to apply function only on not-null values of the particular column.

    Args:
        df (pd.DataFrame): DataFrame to apply function on.
        column (str): Column to apply function on.
        target_column (str): Column to assign function output.
        function (func): Function to apply.

    Returns:
        pd.DataFrame: Resulting DataFrame.
    """
    df.loc[df[column].notnull(), target_column] = \
        df.loc[df[column].notnull(), column].apply(function)

    return df


def pair_func(func, x1, x2):
    """Wrapper for func; if both arguments are NaN, -1 is returned,
    if only one then -0.5."""
    if type(x1) == float and type(x2) == float:
        return -1
    elif type(x1) == float or type(x2) == float:
        return -0.5
    try:
        return func(x1, x2)
    except:
        return -1


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


def get_shingle_similarity(name):
    if name == "cosine":
        func = cosine
    elif name == "jaccard":
        func = jaccard
    elif name == "overlap":
        func = overlap

    func_ = np.vectorize(
        lambda x1, x2: pair_func(func, x1, x2))
    return func_


def haversine_vec(lat1, lon1, lat2, lon2):
    def h(la1, lo1, la2, lo2):
        return haversine((la1, lo1), (la2, lo2), unit='m')
    return np.vectorize(h)(lat1, lon1, lat2, lon2)


def get_numbers_from_name(name):
    return "".join(re.findall("[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", name))
