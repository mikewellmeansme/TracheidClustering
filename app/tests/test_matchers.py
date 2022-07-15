import pytest
import matchers
from pandas import DataFrame, read_csv

monthly_matcher = matchers.MonthlyClimateIndexMatcher()


@pytest.mark.parametrize(
    ('input_df', 'expected_df'),
    [
        (read_csv('test_dataframes/df1.csv'), read_csv('test_dataframes/df2.csv'))
    ]
)
def test_get_shifted_df(input_df: DataFrame, expected_df: DataFrame):
    assert monthly_matcher.__get_shifted_df__(input_df).equals(expected_df)


@pytest.mark.parametrize(
    ('input_df', 'classes_df', 'expected_df'),
    [
        (read_csv('test_dataframes/df3.csv'), read_csv('test_dataframes/df1.csv'), read_csv('test_dataframes/df4.csv'))
    ]
)
def test_merge_with_classes(input_df, classes_df, expected_df):
    assert monthly_matcher.__merge_with_classes__(input_df, classes_df).equals(expected_df)
