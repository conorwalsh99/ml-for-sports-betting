import pytest
import pandas as pd 
import numpy as np

from src.utils import (find_preliminary_games,
                       remove_preliminary_games,
                       convert_moneyline_to_decimal, 
                       convert_MMDD_to_str,
                       match_odds_to_games,
                       sort_labels_by_values,
                       validate_preprocessed_dataframe,
                       validate_odds_dataframe,
                       validate_modelling_ready_dataframe,
                       )


@pytest.fixture
def remove_preliminary_games_input():
    return pd.DataFrame(
        {
            'Date': ['Tue, Oct 16, 2018', 'Tue, Oct 16, 2018', 'Wed Oct 19 2018', 'Thu Oct 20 2018', 'Fri Oct 21 2018', 'Sat Oct 22 2018'],
            'Visitor/Neutral': ['Los Angeles Lakers', 'Los Angeles Lakers', 'Chicago Bulls', 'Boston Celtics', 'Miami Heat', 'Los Angeles Lakers'],
            'Home/Neutral': ['Boston Celtics', 'Boston Celtics', 'Boston Celtics', 'Los Angeles Lakers', 'Los Angeles Lakers', 'Boston Celtics'],
        }
    )

@pytest.fixture
def find_preliminary_games_expected():
    return [0, 1, 2, 4]

@pytest.fixture
def remove_preliminary_games_expected():
    return pd.DataFrame(
        {
            'Date': ['Thu Oct 20 2018', 'Sat Oct 22 2018'],
            'Visitor/Neutral': ['Boston Celtics', 'Los Angeles Lakers'],
            'Home/Neutral': ['Los Angeles Lakers', 'Boston Celtics'],
        }
    )


@pytest.fixture
def negative_moneyline_odds_input():
    return -100

@pytest.fixture
def positive_moneyline_odds_input():
    return 400

@pytest.fixture
def negative_moneyline_to_decimal_odds_expected():
    return 2.0

@pytest.fixture
def positive_moneyline_to_decimal_odds_expected():
    return 5.0

@pytest.fixture
def convert_date_double_digits_input():
    return 1016

@pytest.fixture
def convert_date_single_digit_input():
    return 601

@pytest.fixture
def convert_date_double_digits_expected():
    return 'Oct 16,'

@pytest.fixture
def convert_date_single_digit_expected():
    return 'Jun 1,'

@pytest.fixture
def odds_df():
    return pd.DataFrame(
        {
            'Date': [1016.0, 1016.0, 1016.0, 1016.0],
            'Rot': [501.0, 502.0, 503.0, 504.0],
            'VH': ['V', 'H', 'V', 'H'],
            'Team': ['Philadelphia', 'Boston', 'OklahomaCity', 'GoldenState'],
            'ML': [170.0, -200.0, 711.0, -1100.0]

        }
    )

@pytest.fixture
def match_odds_to_games_input():
    return pd.DataFrame(
        {
            'Date': ['Tue, Oct 16, 2018', 'Tue, Oct 16, 2018'],
            'Home/Neutral': ['Boston Celtics', 'Golden State Warriors'],
            'Visitor/Neutral': ['Philadelphia 76ers', 'Oklahoma City Thunder'],
        }
    )

@pytest.fixture
def match_odds_to_games_expected():
    clean_odds_df = pd.DataFrame(
        {
            'home_odds': [1.5, 1.0909090909],
            'away_odds': [2.7, 8.11]
        }
    )
    box_score_df = pd.DataFrame(
        {
            'Date': ['Tue, Oct 16, 2018', 'Tue, Oct 16, 2018'],
            'Home/Neutral': ['Boston Celtics', 'Golden State Warriors'],
            'Visitor/Neutral': ['Philadelphia 76ers', 'Oklahoma City Thunder'],
        }
    )
    return box_score_df, clean_odds_df


@pytest.fixture
def validate_preprocessed_dataframe_good_input():
    return pd.DataFrame(
        {
            'Date': ['12/03/2018', '23/09/2017'],
            'Visitor/Neutral': ['Boston Celtics', 'Miami Heat'],
            'PTS':[108, 119],
            'Home/Neutral': ['Los Angeles Lakers', 'Chicago Bulls'],
            'PTS.1': [99, 88],
            'home_victory': [0, 1],
            'previous_win_pct': [0.25, 0.9],
            'FG': [0, 1],
            'FG%': [2.5, 9],
            '3P' : [6, 9.8],
            '3P%': [0.7, -2.5], 
            'FT': [0.7, -2.5], 
            'FT%': [0.7, -2.5], 
            'ORB': [0.7, -2.5], 
            'DRB': [0.7, -2.5], 
            'AST': [0.7, -2.5], 
            'STL': [0.7, -2.5], 
            'BLK': [0.7, -2.5], 
            'TOV': [0.7, -2.5], 
            'PF': [0.7, -2.5], 
            'TS%': [0.7, -2.5], 
            'eFG%': [0.7, -2.5], 
            '3PAr': [0.7, -2.5], 
            'FTr': [0.7, -2.5], 
            'ORB%': [0.7, -2.5], 
            'DRB%': [0.7, -2.5], 
            'TRB%': [0.7, -2.5], 
            'AST%': [0.7, -2.5], 
            'STL%': [0.7, -2.5], 
            'BLK%': [0.7, -2.5], 
            'TOV%': [0.7, -2.5], 
            'ORtg': [0.7, -2.5], 
            'DRtg': [0.7, -2.5], 
        }
    )

@pytest.fixture
def validate_preprocessed_dataframe_target_not_binary():
    return pd.DataFrame(
        {
            'Date': ['12/03/2018', '23/09/2017'],
            'Visitor/Neutral': ['Boston Celtics', 'Miami Heat'],
            'PTS':[108, 119],
            'Home/Neutral': ['Los Angeles Lakers', 'Chicago Bulls'],
            'PTS.1': [99, 88],
            'home_victory': [0, 4.5],
            'previous_win_pct': [0.25, 0.9],
            'FG': [0, 1],
            'FG%': [2.5, 9],
            '3P' : [6, 9.8],
            '3P%': [0.7, -2.5], 
            'FT': [0.7, -2.5], 
            'FT%': [0.7, -2.5], 
            'ORB': [0.7, -2.5], 
            'DRB': [0.7, -2.5], 
            'AST': [0.7, -2.5], 
            'STL': [0.7, -2.5], 
            'BLK': [0.7, -2.5], 
            'TOV': [0.7, -2.5], 
            'PF': [0.7, -2.5], 
            'TS%': [0.7, -2.5], 
            'eFG%': [0.7, -2.5], 
            '3PAr': [0.7, -2.5], 
            'FTr': [0.7, -2.5], 
            'ORB%': [0.7, -2.5], 
            'DRB%': [0.7, -2.5], 
            'TRB%': [0.7, -2.5], 
            'AST%': [0.7, -2.5], 
            'STL%': [0.7, -2.5], 
            'BLK%': [0.7, -2.5], 
            'TOV%': [0.7, -2.5], 
            'ORtg': [0.7, -2.5], 
            'DRtg': [0.7, -2.5], 
        }
    )

@pytest.fixture
def validate_preprocessed_dataframe_extra_col():
    return pd.DataFrame(
        {
            'Date': ['12/03/2018', '23/09/2017'],
            'Visitor/Neutral': ['Boston Celtics', 'Miami Heat'],
            'PTS':[108, 119],
            'Home/Neutral': ['Los Angeles Lakers', 'Chicago Bulls'],
            'PTS.1': [99, 88],
            'USG%': [0.9, 6.7],
            'home_victory': [0, 1],
            'previous_win_pct': [0.25, 0.9],
            'FG': [0, 1],
            'FG%': [2.5, 9],
            '3P' : [6, 9.8],
            '3P%': [0.7, -2.5], 
            'FT': [0.7, -2.5], 
            'FT%': [0.7, -2.5], 
            'ORB': [0.7, -2.5], 
            'DRB': [0.7, -2.5], 
            'AST': [0.7, -2.5], 
            'STL': [0.7, -2.5], 
            'BLK': [0.7, -2.5], 
            'TOV': [0.7, -2.5], 
            'PF': [0.7, -2.5], 
            'TS%': [0.7, -2.5], 
            'eFG%': [0.7, -2.5], 
            '3PAr': [0.7, -2.5], 
            'FTr': [0.7, -2.5], 
            'ORB%': [0.7, -2.5], 
            'DRB%': [0.7, -2.5], 
            'TRB%': [0.7, -2.5], 
            'AST%': [0.7, -2.5], 
            'STL%': [0.7, -2.5], 
            'BLK%': [0.7, -2.5], 
            'TOV%': [0.7, -2.5], 
            'ORtg': [0.7, -2.5], 
            'DRtg': [0.7, -2.5], 
        }
    )

@pytest.fixture
def validate_preprocessed_dataframe_missing_cols():
    return pd.DataFrame(
        {
            'Date': ['12/03/2018', '23/09/2017'],
            'Visitor/Neutral': ['Boston Celtics', 'Miami Heat'],
            'PTS':[108, 119],
            'Home/Neutral': ['Los Angeles Lakers', 'Chicago Bulls'],
            'PTS.1': [99, 88],
            'home_victory': [0, 1],
            'FG%': [2.5, 9],
            '3P' : [6, 9.8],
            '3P%': [0.7, -2.5], 
            'FT': [0.7, -2.5], 
            'FT%': [0.7, -2.5], 
            'ORB': [0.7, -2.5], 
            'DRB': [0.7, -2.5], 
            'AST': [0.7, -2.5], 
            'STL': [0.7, -2.5], 
            'BLK': [0.7, -2.5], 
            'TOV': [0.7, -2.5], 
            'PF': [0.7, -2.5], 
            'TS%': [0.7, -2.5], 
            'eFG%': [0.7, -2.5], 
            '3PAr': [0.7, -2.5], 
            'FTr': [0.7, -2.5], 
            'ORB%': [0.7, -2.5], 
            'DRB%': [0.7, -2.5], 
            'TRB%': [0.7, -2.5], 
            'AST%': [0.7, -2.5], 
            'STL%': [0.7, -2.5], 
            'BLK%': [0.7, -2.5], 
            'TOV%': [0.7, -2.5], 
            'ORtg': [0.7, -2.5], 
            'DRtg': [0.7, -2.5], 
        }
    )

@pytest.fixture
def validate_preprocessed_dataframe_features_not_numeric():
    return pd.DataFrame(
        {
            'Date': ['12/03/2018', '23/09/2017'],
            'Visitor/Neutral': ['Boston Celtics', 'Miami Heat'],
            'PTS':[108, 119],
            'Home/Neutral': ['Los Angeles Lakers', 'Chicago Bulls'],
            'PTS.1': [99, 88],
            'home_victory': [0, 1],
            'previous_win_pct': [0.25, 0.9],
            'FG': [0, 1],
            'FG%': [2.5, 9],
            '3P' : [6, 'No Previous Games.'],
            '3P%': [0.7, -2.5], 
            'FT': [0.7, -2.5], 
            'FT%': [0.7, -2.5], 
            'ORB': [0.7, -2.5], 
            'DRB': [0.7, -2.5], 
            'AST': [0.7, -2.5], 
            'STL': [0.7, -2.5], 
            'BLK': [0.7, -2.5], 
            'TOV': [0.7, -2.5], 
            'PF': [0.7, -2.5], 
            'TS%': [0.7, -2.5], 
            'eFG%': [0.7, -2.5], 
            '3PAr': [0.7, -2.5], 
            'FTr': [0.7, -2.5], 
            'ORB%': [0.7, -2.5], 
            'DRB%': [0.7, -2.5], 
            'TRB%': [0.7, -2.5], 
            'AST%': [0.7, -2.5], 
            'STL%': [0.7, -2.5], 
            'BLK%': [0.7, -2.5], 
            'TOV%': [0.7, -2.5], 
            'ORtg': [0.7, -2.5], 
            'DRtg': [0.7, -2.5], 
        }
    )

@pytest.fixture
def validate_odds_dataframe_good_input():
    odds_df = pd.DataFrame(
        {
            'home_odds': [1.2, 4.5, 6.0],
            'away_odds': [2.3, 1.5, 1.2]
        }
    )
    size_betting_simulation = 3
    return odds_df, size_betting_simulation


@pytest.fixture
def validate_odds_dataframe_extra_cols():
    odds_df = pd.DataFrame(
        {
            'home_odds': [1.2, 4.5, 6.0],
            'away_odds': [2.3, 1.5, 1.2],
            'some_other_column': [1.2, 4.6, 3.0],
        }
    )
    size_betting_simulation = 3
    return odds_df, size_betting_simulation

@pytest.fixture
def validate_odds_dataframe_missing_cols():
    odds_df = pd.DataFrame(
        {
            'away_odds': [2.3, 1.5, 1.2]
        }
    )
    size_betting_simulation = 3
    return odds_df, size_betting_simulation

@pytest.fixture
def validate_odds_dataframe_non_float_input():
    odds_df = pd.DataFrame(
        {
            'home_odds': [1.2, 'str', 6.0],
            'away_odds': [2.3, 1.5, 1.2]
        }
    )
    size_betting_simulation = 3
    return odds_df, size_betting_simulation

@pytest.fixture
def validate_odds_dataframe_less_than_1():
    odds_df = pd.DataFrame(
        {
            'home_odds': [1.2, 4.5, 0.8],
            'away_odds': [2.3, 1.5, 1.2]
        }
    )
    size_betting_simulation = 3
    return odds_df, size_betting_simulation

@pytest.fixture
def validate_odds_dataframe_length_mismatch():
    odds_df = pd.DataFrame(
        {
            'home_odds': [1.2, 4.5],
            'away_odds': [2.3, 1.5]
        }
    )
    size_betting_simulation = 3
    return odds_df, size_betting_simulation


@pytest.fixture
def validate_modelling_ready_dataframe_good_input():
    df = pd.DataFrame(
        {
            'previous_win_pct': [0.25, 0.9],
            '3P' : [6, 9.8],
            '3P%': [0.7, -2.5], 
            'FT': [0.7, -2.5], 
            'FT%': [0.7, -2.5], 
            'ORB': [0.7, -2.5], 
            'DRB': [0.7, -2.5], 
            'AST': [0.7, -2.5], 
            'STL': [0.7, -2.5], 
            'BLK': [0.7, -2.5], 
            'TOV': [0.7, -2.5], 
            'PF': [0.7, -2.5], 
            'TS%': [0.7, -2.5], 
            'eFG%': [0.7, -2.5], 
            '3PAr': [0.7, -2.5], 
            'FTr': [0.7, -2.5], 
            'ORB%': [0.7, -2.5], 
            'DRB%': [0.7, -2.5], 
            'TRB%': [0.7, -2.5], 
            'AST%': [0.7, -2.5], 
            'STL%': [0.7, -2.5], 
            'BLK%': [0.7, -2.5], 
            'TOV%': [0.7, -2.5], 
            'ORtg': [0.7, -2.5], 
            'DRtg': [0.7, -2.5], 
        }
    )
    dropped_features = ['FG', 'FG%']

    return df, dropped_features

@pytest.fixture
def validate_modelling_ready_dataframe_extra_cols_input():
    df = pd.DataFrame(
        {
            'Date': ['12/03/2018', '23/09/2017'],
            'Visitor/Neutral': ['Boston Celtics', 'Miami Heat'],
            'PTS':[108, 119],
            'Home/Neutral': ['Los Angeles Lakers', 'Chicago Bulls'],
            'PTS.1': [99, 88],
            'home_victory': [0, 1],
            'previous_win_pct': [0.25, 0.9],
            '3P' : [6, 9.8],
            '3P%': [0.7, -2.5], 
            'FT': [0.7, -2.5], 
            'FT%': [0.7, -2.5], 
            'ORB': [0.7, -2.5], 
            'DRB': [0.7, -2.5], 
            'AST': [0.7, -2.5], 
            'STL': [0.7, -2.5], 
            'BLK': [0.7, -2.5], 
            'TOV': [0.7, -2.5], 
            'PF': [0.7, -2.5], 
            'TS%': [0.7, -2.5], 
            'eFG%': [0.7, -2.5], 
            '3PAr': [0.7, -2.5], 
            'FTr': [0.7, -2.5], 
            'ORB%': [0.7, -2.5], 
            'DRB%': [0.7, -2.5], 
            'TRB%': [0.7, -2.5], 
            'AST%': [0.7, -2.5], 
            'STL%': [0.7, -2.5], 
            'BLK%': [0.7, -2.5], 
            'TOV%': [0.7, -2.5], 
            'ORtg': [0.7, -2.5], 
            'DRtg': [0.7, -2.5], 
        }
    )
    dropped_features = ['FG', 'FG%']

    return df, dropped_features    

@pytest.fixture
def validate_modelling_ready_dataframe_missing_cols_input():
    df = pd.DataFrame(
        {
            '3P' : [6, 9.8],
            '3P%': [0.7, -2.5], 
            'FT': [0.7, -2.5], 
            'FT%': [0.7, -2.5], 
            'ORB': [0.7, -2.5], 
            'DRB': [0.7, -2.5], 
            'AST': [0.7, -2.5], 
            'STL': [0.7, -2.5], 
            'BLK': [0.7, -2.5], 
            'TOV': [0.7, -2.5], 
            'PF': [0.7, -2.5], 
            'TS%': [0.7, -2.5], 
            'eFG%': [0.7, -2.5], 
            '3PAr': [0.7, -2.5], 
            'FTr': [0.7, -2.5], 
            'ORB%': [0.7, -2.5], 
            'DRB%': [0.7, -2.5], 
            'TRB%': [0.7, -2.5], 
            'AST%': [0.7, -2.5], 
            'STL%': [0.7, -2.5], 
            'BLK%': [0.7, -2.5], 
            'TOV%': [0.7, -2.5], 
            'ORtg': [0.7, -2.5], 
            'DRtg': [0.7, -2.5], 
        }
    )
    dropped_features = ['FG', 'FG%']

    return df, dropped_features


@pytest.fixture
def validate_modelling_ready_dataframe_non_float_values():
    df = pd.DataFrame(
        {
            'previous_win_pct': [0.25, 0.9],
            '3P' : [6, 9.8],
            '3P%': [0.7, -2.5], 
            'FT': ['str', -2.5], 
            'FT%': [0.7, -2.5], 
            'ORB': [0.7, -2.5], 
            'DRB': [0.7, -2.5], 
            'AST': [0.7, -2.5], 
            'STL': [0.7, -2.5], 
            'BLK': [0.7, -2.5], 
            'TOV': [0.7, -2.5], 
            'PF': [0.7, -2.5], 
            'TS%': [0.7, -2.5], 
            'eFG%': [0.7, -2.5], 
            '3PAr': [0.7, -2.5], 
            'FTr': [0.7, -2.5], 
            'ORB%': [0.7, -2.5], 
            'DRB%': [0.7, -2.5], 
            'TRB%': [0.7, -2.5], 
            'AST%': [0.7, -2.5], 
            'STL%': [0.7, -2.5], 
            'BLK%': [0.7, -2.5], 
            'TOV%': [0.7, -2.5], 
            'ORtg': [0.7, -2.5], 
            'DRtg': [0.7, -2.5], 
        }
    )
    dropped_features = ['FG', 'FG%']

    return df, dropped_features

@pytest.fixture
def validate_modelling_ready_dataframe_dropped_features_present():
    df = pd.DataFrame(
        {
            'previous_win_pct': [0.25, 0.9],
            'FG': [0, 1],
            'FG%': [2.5, 9],
            '3P' : [6, 9.8],
            '3P%': [0.7, -2.5], 
            'FT': [0.7, -2.5], 
            'FT%': [0.7, -2.5], 
            'ORB': [0.7, -2.5], 
            'DRB': [0.7, -2.5], 
            'AST': [0.7, -2.5], 
            'STL': [0.7, -2.5], 
            'BLK': [0.7, -2.5], 
            'TOV': [0.7, -2.5], 
            'PF': [0.7, -2.5], 
            'TS%': [0.7, -2.5], 
            'eFG%': [0.7, -2.5], 
            '3PAr': [0.7, -2.5], 
            'FTr': [0.7, -2.5], 
            'ORB%': [0.7, -2.5], 
            'DRB%': [0.7, -2.5], 
            'TRB%': [0.7, -2.5], 
            'AST%': [0.7, -2.5], 
            'STL%': [0.7, -2.5], 
            'BLK%': [0.7, -2.5], 
            'TOV%': [0.7, -2.5], 
            'ORtg': [0.7, -2.5], 
            'DRtg': [0.7, -2.5], 
        }
    )
    dropped_features = ['FG', 'FG%']

    return df, dropped_features


@pytest.fixture
def get_lists_to_sort():
    values_list = [1, 3, 2, 4]
    labels_list = ['first', 'second', 'third', 'fourth']
    return values_list, labels_list

@pytest.fixture
def sorted_list_descending():
    values_list_descending = [4, 3, 2, 1]
    sorted_labels_list = ['fourth', 'second', 'third', 'first']
    return sorted_labels_list, values_list_descending

@pytest.fixture
def sorted_list_ascending():
    values_list_ascending = [1, 2, 3, 4]
    sorted_labels_list = [ 'first', 'third',  'second', 'fourth']
    return sorted_labels_list, values_list_ascending


def test_find_preliminary_games(remove_preliminary_games_input, find_preliminary_games_expected):
    box_score_data = remove_preliminary_games_input
    result = find_preliminary_games(box_score_data, num_games=2)
    expected = find_preliminary_games_expected
    assert result == expected, f'remove_preliminary_games returned {result} instead of {expected}.'

def test_remove_preliminary_games(remove_preliminary_games_input, remove_preliminary_games_expected):
    box_score_data = remove_preliminary_games_input
    result = remove_preliminary_games(box_score_data, num_games=2)
    expected = remove_preliminary_games_expected
    pd.testing.assert_frame_equal(result, expected), f'remove_preliminary_games returned {result} instead of {expected}.'

def test_convert_negative_moneyline_to_decimal_odds(negative_moneyline_odds_input, negative_moneyline_to_decimal_odds_expected):
    result = convert_moneyline_to_decimal(negative_moneyline_odds_input)
    expected = negative_moneyline_to_decimal_odds_expected
    assert result == expected, f'convert_moneyline_to_decimal returned {result} instead of {expected}'

def test_convert_positive_moneyline_to_decimal_odds(positive_moneyline_odds_input, positive_moneyline_to_decimal_odds_expected):
    result = convert_moneyline_to_decimal(positive_moneyline_odds_input)
    expected = positive_moneyline_to_decimal_odds_expected
    assert result == expected, f'convert_moneyline_to_decimal returned {result} instead of {expected}'

def test_convert_date_single_digits(convert_date_single_digit_input, convert_date_single_digit_expected):
    result = convert_MMDD_to_str(convert_date_single_digit_input)
    expected = convert_date_single_digit_expected
    assert result == expected, f'convert_MMDD_to_str returned {result} instead of {expected}.'

def test_convert_date_double_digits(convert_date_double_digits_input, convert_date_double_digits_expected):
    result = convert_MMDD_to_str(convert_date_double_digits_input)
    expected = convert_date_double_digits_expected
    assert result == expected, f'convert_MMDD_to_str returned {result} instead of {expected}.'

def test_get_corresponding_odds(odds_df, match_odds_to_games_input, match_odds_to_games_expected):
    box_score_data_result, odds_data_result = match_odds_to_games(match_odds_to_games_input, odds_df)
    box_score_data_expected, odds_data_expected = match_odds_to_games_expected
    pd.testing.assert_frame_equal(box_score_data_result, box_score_data_expected), f'match_odds_to_games returned {box_score_data_result} instead of {box_score_data_expected}.'    
    pd.testing.assert_frame_equal(odds_data_result, odds_data_expected), f'match_odds_to_games returned {odds_data_result} instead of {odds_data_expected}.'

def test_sort_labels_by_values_ascending(get_lists_to_sort, sorted_list_ascending):
    values_list, labels_list = get_lists_to_sort
    result = sort_labels_by_values(values_list, labels_list, return_values=True, descending=False)
    expected = sorted_list_ascending
    assert result == expected, f'sort_labels_by_values returned {result} instead of {expected}.'
    
def test_sort_labels_by_values_descending(get_lists_to_sort, sorted_list_descending):
    values_list, labels_list = get_lists_to_sort
    result = sort_labels_by_values(values_list, labels_list, return_values=True, descending=True)
    expected = sorted_list_descending
    assert result == expected, f'sort_labels_by_values returned {result} instead of {expected}.'

def test_validate_preprocessed_dataframe_good_input(validate_preprocessed_dataframe_good_input):
    df = validate_preprocessed_dataframe_good_input
    assert validate_preprocessed_dataframe(df), 'ValueError raised.'

def test_validate_preprocessed_dataframe_missing_cols(validate_preprocessed_dataframe_missing_cols):
    with pytest.raises(ValueError):
        validate_preprocessed_dataframe(validate_preprocessed_dataframe_missing_cols)

def test_validate_preprocessed_dataframe_extra_col(validate_preprocessed_dataframe_extra_col):
    with pytest.raises(ValueError):
        validate_preprocessed_dataframe(validate_preprocessed_dataframe_extra_col)

def test_validate_preprocessed_dataframe_features_not_numeric(validate_preprocessed_dataframe_features_not_numeric):
    with pytest.raises(ValueError):
        validate_preprocessed_dataframe(validate_preprocessed_dataframe_features_not_numeric)

def test_validate_preprocessed_dataframe_target_not_binary(validate_preprocessed_dataframe_target_not_binary):
    with pytest.raises(ValueError):
        validate_preprocessed_dataframe(validate_preprocessed_dataframe_target_not_binary)

def test_validate_odds_dataframe_good_input(validate_odds_dataframe_good_input):
    df, n = validate_odds_dataframe_good_input
    assert validate_odds_dataframe(df, n), 'validate_odds_dataframe raised a ValueError.'

def test_validate_odds_dataframe_extra_cols(validate_odds_dataframe_extra_cols):
    df, n = validate_odds_dataframe_extra_cols
    with pytest.raises(ValueError):
        validate_odds_dataframe(df, n), 'validate_odds_dataframe raised a ValueError.'

def test_validate_odds_dataframe_missing_cols(validate_odds_dataframe_missing_cols):
    df, n = validate_odds_dataframe_missing_cols
    with pytest.raises(ValueError):
        validate_odds_dataframe(df, n), 'validate_odds_dataframe raised a ValueError.'

def test_validate_odds_dataframe_non_float_input(validate_odds_dataframe_non_float_input):
    df, n = validate_odds_dataframe_non_float_input
    with pytest.raises(TypeError):
        validate_odds_dataframe(df, n), 'validate_odds_dataframe raised a TypeError.'

def test_validate_odds_dataframe_less_than_1(validate_odds_dataframe_less_than_1):
    df, n = validate_odds_dataframe_less_than_1
    with pytest.raises(ValueError):
        validate_odds_dataframe(df, n), 'validate_odds_dataframe raised a ValueError.'

def test_validate_modelling_ready_dataframe_good_input(validate_modelling_ready_dataframe_good_input):
    df, dropped_features = validate_modelling_ready_dataframe_good_input
    assert validate_modelling_ready_dataframe(df, dropped_features), f'validate_modelling_ready_dataframe_good_input failed to return True.'

def test_validate_modelling_ready_dataframe_extra_cols(validate_modelling_ready_dataframe_extra_cols_input):
    df, dropped_features = validate_modelling_ready_dataframe_extra_cols_input
    with pytest.raises(ValueError):
        validate_modelling_ready_dataframe(df, dropped_features), f'validate_modelling_ready_dataframe_good_input failed to return True.'    

def test_validate_modelling_ready_dataframe_missing_cols(validate_modelling_ready_dataframe_missing_cols_input):
    df, dropped_features = validate_modelling_ready_dataframe_missing_cols_input
    with pytest.raises(ValueError):
        validate_modelling_ready_dataframe(df, dropped_features), f'validate_modelling_ready_dataframe_good_input failed to return True.'            
        
def test_validate_modelling_ready_dataframe_dropped_features_present(validate_modelling_ready_dataframe_dropped_features_present):
    df, dropped_features = validate_modelling_ready_dataframe_dropped_features_present
    with pytest.raises(ValueError):
        validate_modelling_ready_dataframe(df, dropped_features), f'validate_modelling_ready_dataframe_good_input failed to return True.'            

def test_validate_modelling_ready_dataframe_non_float_values(validate_modelling_ready_dataframe_non_float_values):
    df, dropped_features = validate_modelling_ready_dataframe_non_float_values
    with pytest.raises(TypeError):
        validate_modelling_ready_dataframe(df, dropped_features), f'validate_modelling_ready_dataframe_good_input failed to return True.'
