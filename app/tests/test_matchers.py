import os
import pytest
import matchers

from pandas import DataFrame, read_csv

monthly_matcher = matchers.MonthlyClimateIndexMatcher()
path_001 = 'dataframes/test_matchers/get_shifted_df/'
path_002 = 'dataframes/test_matchers/merge_with_classes/'


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
def test_merge_with_classes(input_df, classes_df, expected_df):
    assert monthly_matcher.__merge_with_classes__(input_df, classes_df).equals(expected_df)
