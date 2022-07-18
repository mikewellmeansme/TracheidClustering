import os
import pytest
import json

import climate_indexes
import matchers

from collections import defaultdict
from numpy import array_equal, nan
from pandas import DataFrame, read_csv
from typing import Dict, List

path_001 = 'dataframes/test_matchers/get_shifted_df/'
path_002 = 'dataframes/test_matchers/merge_with_classes/'
path_003 = 'dataframes/test_matchers/get_classes_rows/'
path_004 = 'dataframes/test_matchers/get_values_per_class/'

monthly_matcher = matchers.MonthlyClimateIndexMatcher()
climate_index = climate_indexes.ClimateIndex('Prec', f'{path_004}/prec_minusinsk.csv')


def json_load(path: str) -> Dict:
    with open(path, 'r', encoding='utf8') as json_file:
        result = json.load(json_file)
    result = {int(k): v if k.isnumeric() else k for k, v in result.items()}
    return result


def fix_nan(d: Dict[int, List]) -> Dict[int, List]:
    result = defaultdict(list)
    for key in d:
        for el in d[key]:
            val = nan if el == 'NaN' else float(el)
            result[key].append(val)
    return dict(result)


def compare_exact(first: Dict, second: Dict) -> bool:
    """Return whether two dicts of arrays are exactly equal"""
    if first.keys() != second.keys():
        return False
    return all(array_equal(first[key], second[key], equal_nan=True) for key in first)


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
        (
                read_csv(f'{path_002}/input_df/{i}'),
                read_csv(f'{path_002}/classes_df/{i}'),
                read_csv(f'{path_002}/expected_df/{i}')
        )
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


get_values_per_class_args = json_load(f'{path_004}/arguments.json')


@pytest.mark.parametrize(
    ('input_df', 'expected', 'prev', 'month', 'classes'),
    [
        (
                read_csv(f'{path_004}/input_df.csv'),
                fix_nan(json_load(f'{path_004}/expected/{i}.json')),
                *get_values_per_class_args[i]
        )
        for i, _ in enumerate(os.listdir(f'{path_004}/expected'), 1)
    ]
)
def test_get_values_per_class(
        input_df: DataFrame,
        expected: Dict[int, List],
        prev: bool,
        month: str,
        classes: List
):
    assert compare_exact(monthly_matcher.get_values_per_class(climate_index, input_df, prev, month, classes), expected)
