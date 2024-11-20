import pytest 
import pandas as pd 
import numpy as np 
import json


from src.feature_engineering import FeatureConstructor

@pytest.fixture
def add_target_input():
    return pd.DataFrame(
        {
            'Date': ['Tue, Oct 16, 2018', 'Tue, Oct 16, 2018'],
            'Visitor/Neutral': ['New York Knicks', 'Los Angeles Lakers'],
            'Home/Neutral': ['Boston Celtics', 'Chicago Bulls'],
            'PTS': [100, 110],
            'PTS.1': [99, 111],
        }
    )

@pytest.fixture
def add_target_expected():
    return pd.DataFrame(
        {
            'home_victory': [np.int32(0), np.int32(1)],
            'Date': ['Tue, Oct 16, 2018', 'Tue, Oct 16, 2018'],
            'Visitor/Neutral': ['New York Knicks', 'Los Angeles Lakers'],
            'Home/Neutral': ['Boston Celtics', 'Chicago Bulls'],
            'PTS': [100, 110],
            'PTS.1': [99, 111],
        }
    )

@pytest.fixture
def get_raw_differentials_input():
    return pd.DataFrame(
        {
            'Date': ['Tue, Oct 16, 2018', 'Tue, Oct 16, 2018'],
            'Visitor/Neutral': ['New York Knicks', 'Los Angeles Lakers'],
            'Home/Neutral': ['Boston Celtics', 'Chicago Bulls'],
            'PTS': [100, 110],
            'PTS.1': [99, 111],
            'home_STL': [4, 7],
            'home_FG': [50, 67],
            'away_STL': [3, 9],
            'away_FG': [46, 87],
        }
    )

@pytest.fixture
def get_features():
    return ['STL', 'FG']

@pytest.fixture
def get_raw_differentials_expected():
    return pd.DataFrame(
        {
            'Date': ['Tue, Oct 16, 2018', 'Tue, Oct 16, 2018'],
            'Visitor/Neutral': ['New York Knicks', 'Los Angeles Lakers'],
            'Home/Neutral': ['Boston Celtics', 'Chicago Bulls'],
            'PTS': [100, 110],
            'PTS.1': [99, 111],
            'home_STL': [1, -2],
            'home_FG': [4, -20],
            'away_STL': [-1, 2],
            'away_FG': [-4, 20],
        }
    )

@pytest.fixture
def get_average_box_scores_input():
    return pd.DataFrame(
        {
            'Date': ['Tue, Oct 16, 2018', 'Tue, Oct 16, 2018', 'Wed Oct 19 2018', 'Thu Oct 20 2018', 'Fri Oct 21 2018'],
            'Visitor/Neutral': ['New York Knicks', 'Los Angeles Lakers', 'Boston Celtics', 'Boston Celtics', 'Los Angeles Lakers'],
            'Home/Neutral': ['Boston Celtics', 'Chicago Bulls', 'Los Angeles Lakers', 'Miami Heat', 'Boston Celtics'],
            'PTS': [100, 110, 105, 88, 90],
            'PTS.1': [99, 111, 92, 99, 102],
            'home_STL': [4, 7, 9, 4, 11],
            'home_FG': [50, 67, 34, 68, 46],
            'away_STL': [3, 9, 5, 8, 10],
            'away_FG': [46, 87, 54, 66, 49],
        }
    )


@pytest.fixture
def get_average_box_scores_expected():
    return pd.DataFrame(
        {
            'Date': ['Tue, Oct 16, 2018', 'Tue, Oct 16, 2018', 'Wed Oct 19 2018', 'Thu Oct 20 2018', 'Fri Oct 21 2018'],
            'Visitor/Neutral': ['New York Knicks', 'Los Angeles Lakers', 'Boston Celtics', 'Boston Celtics', 'Los Angeles Lakers'],
            'Home/Neutral': ['Boston Celtics', 'Chicago Bulls', 'Los Angeles Lakers', 'Miami Heat', 'Boston Celtics'],
            'PTS': [100, 110, 105, 88, 90],
            'PTS.1': [99, 111, 92, 99, 102],
            'home_STL': ['No Previous Games', 'No Previous Games', 2, 'No Previous Games', 1/3],
            'home_FG': ['No Previous Games', 'No Previous Games', 20, 'No Previous Games', 22/3],
            'away_STL': ['No Previous Games', 'No Previous Games', 1, -3/2, 3],
            'away_FG': ['No Previous Games', 'No Previous Games', 4, 12, 0/2],
        }
    )


@pytest.fixture 
def add_previous_season_win_pct_input():
    return pd.DataFrame(
        {
            'home_victory': [np.int32(0), np.int32(1), np.int32(0), np.int32(1), np.int32(1)],
            'Date': ['Tue, Oct 16, 2018', 'Tue, Oct 16, 2018', 'Wed Oct 19 2018', 'Thu Oct 20 2018', 'Fri Oct 21 2018'],
            'Visitor/Neutral': ['New York Knicks', 'Los Angeles Lakers', 'Boston Celtics', 'Boston Celtics', 'Los Angeles Lakers'],
            'Home/Neutral': ['Boston Celtics', 'Chicago Bulls', 'Los Angeles Lakers', 'Miami Heat', 'Boston Celtics'],
            'PTS': [100, 110, 105, 88, 90],
            'PTS.1': [99, 111, 92, 99, 102],
        }
    )

@pytest.fixture 
def get_previous_season_records():
    with open(rf'.\data\input\win_pct_2018.json', 'r') as f:
        win_pct_dict = json.load(f)
    return win_pct_dict
    
@pytest.fixture 
def add_previous_season_win_pct_expected():
    return pd.DataFrame(
        {
            'home_victory': [np.int32(0), np.int32(1), np.int32(0), np.int32(1), np.int32(1)],
            'Date': ['Tue, Oct 16, 2018', 'Tue, Oct 16, 2018', 'Wed Oct 19 2018', 'Thu Oct 20 2018', 'Fri Oct 21 2018'],
            'Visitor/Neutral': ['New York Knicks', 'Los Angeles Lakers', 'Boston Celtics', 'Boston Celtics', 'Los Angeles Lakers'],
            'Home/Neutral': ['Boston Celtics', 'Chicago Bulls', 'Los Angeles Lakers', 'Miami Heat', 'Boston Celtics'],
            'PTS': [100, 110, 105, 88, 90],
            'PTS.1': [99, 111, 92, 99, 102],
            'home_previous_win_pct': [0.671, 0.329, 0.427, 0.537, 0.671],
            'away_previous_win_pct': [0.354, 0.427, 0.671, 0.671, 0.427],
        }
    )


@pytest.fixture
def get_averaged_differentials_input():
    return pd.DataFrame(
        {   
            'home_victory': [np.int32(0), np.int32(1), np.int32(0), np.int32(1), np.int32(1)],
            'Date': ['Tue, Oct 16, 2018', 'Tue, Oct 16, 2018', 'Wed Oct 19 2018', 'Thu Oct 20 2018', 'Fri Oct 21 2018'],
            'Visitor/Neutral': ['New York Knicks', 'Los Angeles Lakers', 'Boston Celtics', 'Boston Celtics', 'Los Angeles Lakers'],
            'PTS': [100, 110, 105, 88, 90],
            'Home/Neutral': ['Boston Celtics', 'Chicago Bulls', 'Los Angeles Lakers', 'Miami Heat', 'Boston Celtics'],            
            'PTS.1': [99, 111, 92, 99, 102],
            'home_STL': ['No Previous Games', 'No Previous Games', 2, 'No Previous Games', 1/3],
            'home_FG': ['No Previous Games', 'No Previous Games', 20, 'No Previous Games', 22/3],
            'away_STL': ['No Previous Games', 'No Previous Games', 1, -3/2, 3],
            'away_FG': ['No Previous Games', 'No Previous Games', 4, 12, 0/2],
            'home_previous_win_pct': [0.671, 0.329, 0.427, 0.537, 0.671],
            'away_previous_win_pct': [0.354, 0.427, 0.671, 0.671, 0.427],                        
        }
    )

@pytest.fixture
def get_averaged_differentials_expected():
    return pd.DataFrame(
        {
            'home_victory': [np.int32(0), np.int32(1), np.int32(0), np.int32(1), np.int32(1)],            
            'Date': ['Tue, Oct 16, 2018', 'Tue, Oct 16, 2018', 'Wed Oct 19 2018', 'Thu Oct 20 2018', 'Fri Oct 21 2018'],
            'Visitor/Neutral': ['New York Knicks', 'Los Angeles Lakers', 'Boston Celtics', 'Boston Celtics', 'Los Angeles Lakers'],
            'PTS': [100, 110, 105, 88, 90],
            'Home/Neutral': ['Boston Celtics', 'Chicago Bulls', 'Los Angeles Lakers', 'Miami Heat', 'Boston Celtics'],
            'PTS.1': [99, 111, 92, 99, 102],
            'STL': ['No Previous Games', 'No Previous Games', (2 - 1), 'No Previous Games', (1/3 - 3)],
            'FG': ['No Previous Games', 'No Previous Games', (20 - 4), 'No Previous Games', (22/3 - 0)],      
            'previous_win_pct': [(0.671 - 0.354), (0.329 - 0.427), (0.427 - 0.671), (0.537 - 0.671), (0.671 - 0.427)],
        }
    )


class TestFeatureConstructor:
    def test_add_target(self, add_target_input, add_target_expected):
        feature_constructor = FeatureConstructor(box_score_data=add_target_input)
        feature_constructor.add_target()
        result =  feature_constructor.box_score_data
        pd.testing.assert_frame_equal(result, add_target_expected), f'Did not add target correctly. Returned {result} instead of {add_target_expected}'

    def test_get_raw_differentials(self, get_raw_differentials_input, get_features, get_raw_differentials_expected):
        feature_constructor = FeatureConstructor(box_score_data=get_raw_differentials_input)
        feature_constructor.raw_differentials_df = feature_constructor.box_score_data.copy(deep=True)
        feature_constructor.average_box_scores_df = feature_constructor.box_score_data.copy(deep=True)
        feature_constructor.get_raw_differentials(features=get_features)
        result = feature_constructor.raw_differentials_df
        pd.testing.assert_frame_equal(result, get_raw_differentials_expected), f'Returned {result} instead of {get_raw_differentials_expected}'

    def test_get_average_box_scores_use_differences(self, get_average_box_scores_input, get_features, get_average_box_scores_expected):
        feature_constructor = FeatureConstructor(box_score_data=get_average_box_scores_input)
        feature_constructor.raw_differentials_df = feature_constructor.box_score_data.copy(deep=True)
        feature_constructor.average_box_scores_df = feature_constructor.box_score_data.copy(deep=True)
        feature_constructor.get_raw_differentials(features=get_features)
        feature_constructor.get_average_box_scores(features=get_features)
        result = feature_constructor.average_box_scores_df
        pd.testing.assert_frame_equal(result, get_average_box_scores_expected), f'Returned {result} instead of {get_average_box_scores_expected}'
    
    def test_add_previous_season_win_pct(self, add_previous_season_win_pct_input, get_previous_season_records, add_previous_season_win_pct_expected):
        feature_constructor = FeatureConstructor(add_previous_season_win_pct_input, get_previous_season_records)
        feature_constructor.raw_differentials_df = feature_constructor.box_score_data.copy(deep=True)
        feature_constructor.average_box_scores_df = feature_constructor.box_score_data.copy(deep=True)
        feature_constructor.add_previous_season_win_pct()
        result = feature_constructor.average_box_scores_df
        pd.testing.assert_frame_equal(result, add_previous_season_win_pct_expected), f'Returned {result} instead of {add_previous_season_win_pct_expected}'

    def test_get_averaged_differentials_differences_used(self, 
        get_averaged_differentials_input,  
        get_features,
        get_averaged_differentials_expected, 
        ):
        feature_constructor = FeatureConstructor(get_averaged_differentials_input)
        feature_constructor.raw_differentials_df = feature_constructor.box_score_data.copy(deep=True)
        feature_constructor.average_box_scores_df = feature_constructor.box_score_data.copy(deep=True)
        feature_constructor.get_averaged_differentials(get_features)
        result = feature_constructor.average_box_scores_df
        pd.testing.assert_frame_equal(result, get_averaged_differentials_expected), f'Returned {result} instead of {get_averaged_differentials_expected}'

    def test_construct_features(self, get_average_box_scores_input, get_features, get_previous_season_records, get_averaged_differentials_expected):
        feature_constructor = FeatureConstructor(box_score_data=get_average_box_scores_input,
                                                previous_season_win_pcts=get_previous_season_records)
        result = feature_constructor.construct_features(get_features)
        pd.testing.assert_frame_equal(result, get_averaged_differentials_expected), f'Returned {result} instead of {get_averaged_differentials_expected}'
