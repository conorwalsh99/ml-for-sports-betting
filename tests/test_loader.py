import pytest 
import pandas as pd 
import numpy as np 
import json


from src.loader import Loader


@pytest.fixture
def get_box_scores_2015():
    box_scores_path = rf'./data/input/results_and_box_scores_2015.csv'
    box_scores_df = pd.read_csv(box_scores_path)
    return box_scores_df

@pytest.fixture
def get_previous_season_win_pct_dict():
    previous_season_won_pct_dict_path = rf'./data/input/win_pct_2014.json'
    with open(previous_season_won_pct_dict_path, 'r') as file:
        win_pct_dict = json.load(file)
    return win_pct_dict

@pytest.fixture
def get_odds():
    odds_path = r'./data/input/nba odds 2018-19.csv'
    return  pd.read_csv(odds_path)
    

class TestLoader:

    def test_load_box_scores(self, get_box_scores_2015):
        loader = Loader(year=2015)
        result = loader.load_box_scores()
        expected = get_box_scores_2015
        pd.testing.assert_frame_equal(result, expected), f'Loader.load_box_scores returned {result} instead of {expected}.'
    

    def test_load_previous_season_win_pct_dict(self, get_previous_season_win_pct_dict):
        loader = Loader(year=2015)
        result = loader.load_previous_season_win_pct_dict()
        expected = get_previous_season_win_pct_dict
        assert result == expected, f'Loader.load_previous_season_ win_pct returned {result} instead of {expected}.'

    def test_load_odds(self, get_odds):
        loader = Loader()
        result = loader.load_odds(path=r'./data/input/nba odds 2018-19.csv')
        expected = get_odds
        pd.testing.assert_frame_equal(result, expected), f'Loader.load_odds returned {result} instead of {expected}.'

