import os
import pytest
import matchers
import json

from numpy import array_equal
from pandas import DataFrame, read_csv
from typing import Dict, List

monthly_matcher = matchers.MonthlyClimateIndexMatcher()
path_001 = 'dataframes/test_matchers/get_shifted_df/'
path_002 = 'dataframes/test_matchers/merge_with_classes/'
path_003 = 'dataframes/test_matchers/get_classes_rows/'


def json_load(path):
    with open(path, 'r', encoding='utf8') as json_file:
        result = json.load(json_file)
    result = {int(k): v if k.isnumeric() else k for k, v in result.items()}
    return result


def compare_exact(first, second):
    """Return whether two dicts of arrays are exactly equal"""
    if first.keys() != second.keys():
        return False
    return all(array_equal(first[key], second[key]) for key in first)


@pytest.mark.parametrize(
    ('input_df', 'expected_df'),
    [
        (read_csv(f'{path_001}/input_df/{i}'), read_csv(f'{path_001}/expected_df/{i}'))
        for i in os.listdir(f'{path_001}/input_df')
    ]
)
def test_get_shifted_df(input_df: DataFrame, expected_df: DataFrame):
    assert monthly_matcher.__get_shifted_df__(input_df).equals(expected_df)


@pytest.mark.parametrize(
    ('input_df', 'classes_df', 'expected_df'),
    [
        (read_csv(f'{path_002}/input_df/{i}'), read_csv(f'{path_002}/classes_df/{i}'), read_csv(f'{path_002}/expected_df/{i}'))
        for i in os.listdir(f'{path_002}/input_df')
    ]
)
def test_merge_with_classes(input_df: DataFrame, classes_df: DataFrame, expected_df: DataFrame):
    assert monthly_matcher.__merge_with_classes__(input_df, classes_df).equals(expected_df)


@pytest.mark.parametrize(
    ('input_df', 'expected'),
    [
        (read_csv(f'{path_003}/input_df/{i}.csv'), json_load(f'{path_003}/expected_json/{i}.json'))
        for i, _ in enumerate(os.listdir(f'{path_003}/input_df'), 1)
    ]
)
def test_get_classes_rows(input_df: DataFrame, expected: Dict[int, List]):
    assert compare_exact(monthly_matcher.__get_classes_rows__(input_df), expected)
