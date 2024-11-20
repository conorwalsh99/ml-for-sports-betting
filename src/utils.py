import os
import json
import pandas as pd 
import numpy as np
import toml
import numbers
from typing import List, Tuple, Union
from collections import namedtuple


config = toml.load(os.path.join('.', 'config.toml'))

TEAMS = config['NBA']['teams']
BOX_SCORE_STATISTICS = config['modelling']['BOX_SCORE_STATISTICS']
ADDITIONAL_FEATURES = config['modelling']['ADDITIONAL_FEATURES']
TARGET = config['modelling']['TARGET']
GAME_ID_COLS = config['modelling']['GAME_IDENTIFIER_COLS']

month_int_to_str_dict = {
    '10': 'Oct',
    '11': 'Nov',
    '12': 'Dec',
    '1': 'Jan',
    '2': 'Feb',
    '3': 'Mar',
    '4': 'Apr',
    '5': 'May',
    '6': 'Jun',
    '7': 'Jul', 
}
team_name_conversion_dict = {
    'Dallas Mavericks': 'Dallas', 
    'Boston Celtics': 'Boston', 
    'Phoenix Suns': 'Phoenix',
    'Detroit Pistons': 'Detroit',
    'Toronto Raptors': 'Toronto',
    'Memphis Grizzlies': 'Memphis',
    'Portland Trail Blazers': 'Portland',
    'Cleveland Cavaliers': 'Cleveland',
    'Denver Nuggets': 'Denver',
    'Houston Rockets': 'Houston',
    'San Antonio Spurs': 'SanAntonio',
    'Chicago Bulls': 'Chicago',
    'Golden State Warriors': 'GoldenState',
    'Philadelphia 76ers': 'Philadelphia',
    'Oklahoma City Thunder': 'OklahomaCity',
    'Utah Jazz': 'Utah',
    'Miami Heat': 'Miami',
    'Charlotte Hornets': 'Charlotte',
    'Minnesota Timberwolves': 'Minnesota',
    'Washington Wizards': 'Washington',
    'Sacramento Kings': 'Sacramento',
    'Atlanta Hawks': 'Atlanta',
    'Brooklyn Nets': 'Brooklyn',
    'New York Knicks': 'NewYork',
    'Milwaukee Bucks': 'Milwaukee',
    'Los Angeles Lakers': 'LALakers',
    'Los Angeles Clippers': 'LAClippers',
    'Indiana Pacers': 'Indiana',
    'New Orleans Pelicans': 'NewOrleans',
    'Orlando Magic': 'Orlando', 
}


def find_preliminary_games(box_score_data: pd.DataFrame, num_games: int) -> List[int]:
    """
    For a given season, find the indices of the first num_games games played by each team.
    
    Args:
        box_score_data (pd.DataFrame):
            DataFrame containing box score data for the given season.
        num_games (int):
            Number of initial games for each team in the season (not sufficiently well-informed to be considered for predictions).
        
    Returns:
        List[int]:
            Indices of preliminary games.
    """
    preliminary_games = []
    for team in TEAMS:
        home_games_mask = box_score_data['Home/Neutral'] == team
        away_games_mask = box_score_data['Visitor/Neutral'] == team
        games_involving_team = box_score_data[(home_games_mask | away_games_mask)]
        given_team_preliminary_games = games_involving_team.index[:num_games]
        preliminary_games.extend(given_team_preliminary_games)

    return list(set(preliminary_games))

def remove_preliminary_games(box_score_data: pd.DataFrame, num_games: int = 10) -> pd.DataFrame:
    """
    For a given season, remove the first num_games games played by each team from the data.

    These games are only used to calculate the box score features for other data points, 
    and are not included in the training or test sets, as they are not sufficiently well-informed for prediction. 
    
    Args:
        box_score_data (pd.DataFrame):
            DataFrame containing box score data for the given season.
        num_games (int):
            Number of initial games for each team in the season (not sufficiently well-informed to be considered for predictions).
    
    Returns:
        pd.DataFrame: 
            Box score data with preliminary games removed.
    """
    preliminary_games = find_preliminary_games(box_score_data, num_games)
    box_score_data_informed_games = box_score_data.drop(index=preliminary_games)
    box_score_data_informed_games = box_score_data_informed_games.reset_index(drop=True)
    return box_score_data_informed_games 

def convert_moneyline_to_decimal(odds: float) -> float:
    """
    Convert moneyline odds to decimal odds.

    Args:
        odds (float):
            odds in moneyline format.

    Returns:
        float: 
            odds in decimal format.
    """
    if odds == 0:
        raise ValueError('Moneyline odds must be non-zero.')
    
    if odds < 0:
        stake = abs(odds)
        winnings = 100
    else:
        stake = 100
        winnings = odds 

    return 1 + (winnings / stake)
     
def convert_MMDD_to_str(date: int) -> str:
    """
    Convert date from MMDD int format to Month (3-char str) D (str(int)) format.

    e.g. 1016 -> Oct 16

    Note: return date string with a comma at the end e.g. Oct 16,
    This is done because later we check if this date string is a substring of a larger date string.
    Without including the comma, we could return multiple unwanted results,
    e.g. if we want to isolate games played on May 1, (date_str = 'May 1') checking:
        is date_str in larger_date_str will return True for each of 
        [May 1, May 10, May 11, May 12, ..., May 16, ...] and we only want to isolate games from May 1.
        Checking this with a comma at the end (is 'May 1,' in larger_date_str) will only return True for May 1, YYYY. 

    Args:
        date (int):
            Date in integer format (to be converted)
    Returns:
        str:
            Date in str format
    """
    date_str = str(int(date))
    if len(date_str) == 4:
        month = date_str[:2]
    else:
        month = date_str[:1]    
    month_str = month_int_to_str_dict[month]

    day = date_str[-2:]        
    day_str = str(day)
    if day_str[0] == '0':
        day_str = day_str[1:]

    return f'{month_str} {day_str},' 

def match_odds_to_games(box_score_data: pd.DataFrame, odds_data: pd.DataFrame) -> Tuple[pd.DataFrame]:
    """
    Get the odds that correspond to each game and return in new odds dataframe.  

    Args:
        box_score_data (pd.DataFrame):
            box score data for given season.
        odds_data (pd.DataFrame):
            Dataframe containing odds for given season in moneyline (and otherwise messy) format.
    
    Returns:
        Tuple[pd.DataFrame]:
            Box score data and dataframe containing odds for corresponding games in box score dataframe.
    """
    games_without_matching_odds = []
    home_odds = []
    away_odds = []

    home_mask = odds_data['VH'] == 'H' 
    away_mask = odds_data['VH'] == 'V' 
    neutral_mask = odds_data['VH'] == 'N'

    for i in range(box_score_data.shape[0]):
        home_team = box_score_data.iloc[i]['Home/Neutral']
        away_team = box_score_data.iloc[i]['Visitor/Neutral']
        home_team_converted = team_name_conversion_dict[home_team]
        away_team_converted = team_name_conversion_dict[away_team]
        home_team_mask = odds_data['Team'] == home_team_converted
        away_team_mask = odds_data['Team'] == away_team_converted

        date = box_score_data.iloc[i]['Date']
        date_mask = odds_data['Date'].apply(lambda x: pd.notna(x) and convert_MMDD_to_str(x) in date)

        home_team_odds_data = odds_data[((neutral_mask | home_mask) & home_team_mask & date_mask)]
        away_team_odds_data = odds_data[((neutral_mask | away_mask) & away_team_mask & date_mask)]

        if home_team_odds_data.shape[0] == 1 and away_team_odds_data.shape[0] == 1:
            home_moneyline_odds = home_team_odds_data['ML'].values[0]
            away_moneyline_odds = away_team_odds_data['ML'].values[0]
            home_odds.append(convert_moneyline_to_decimal(home_moneyline_odds))
            away_odds.append(convert_moneyline_to_decimal(away_moneyline_odds))
        else:
            games_without_matching_odds.append(i)
            print(f'Date of game missing odds: {date}, home team: {home_team}, away team: {away_team}')

    box_score_data = box_score_data.drop(index=games_without_matching_odds)
    box_score_data = box_score_data.reset_index(drop=True)
    corresponding_odds = pd.DataFrame(
        {
            'home_odds': home_odds,
            'away_odds': away_odds
        }
    )
    return box_score_data, corresponding_odds

def sort_labels_by_values(values: List[float], labels: List[str], descending=False, return_values=False) -> Union[Tuple[List[Union[float, str]]], List[str]]:
    """
    Given a list of values and a list of corresponding labels, sort both lists based on the list of values. 

    Args:
        values (List[float]):
            List of values to sort in ascending or descending order.
        labels (List[str]):
            List of labels that correspond to the points in values.
        descending (bool):
            Flag to indicate whether to sort lists in descending order. Defaults to False. 
        return_values (bool):
            Flag to indicate whether to return list of values along with list of labels. Defaults to False.
    
    Returns:
        Union[Tuple[List[Union[float, str]]], List[str]]:
            Tuple of lists of sorted values and corresponding labels, or just list of sorted labels.
    """
    zipped_list = list(zip(values, labels))
    sorted_zipped_list = sorted(zipped_list, reverse=descending)
    values_sorted, labels_sorted = zip(*sorted_zipped_list)

    return (list(labels_sorted), list(values_sorted)) if return_values else labels_sorted

def validate_preprocessed_dataframe(df: pd.DataFrame) -> bool:
    """Validate that the columns and data types contained in the dataframe are as expected.

    Args:
        df (pd.DataFrame): DataFrame containing preprocessed dataset.

    Raises:
        ValueError:
            - If one or more null values are found in the dataframe.
            - If the set of columns are not as expected.
            - If the target is not a binary variable.
            - If one or more non-numeric features are found.

    Returns:
        bool: Flag to indicate whether validation test passed or not.  
    """
    df = df.copy(deep=True)

    expected_columns = [TARGET] + GAME_ID_COLS + BOX_SCORE_STATISTICS + ADDITIONAL_FEATURES
    actual_columns = df.columns

    nulls_found = df.isnull().any().sum() > 0
    if nulls_found:
        raise ValueError('Nulls found in DataFrame.')

    expected_columns_found = set(expected_columns) == set(actual_columns)
    if not expected_columns_found:
        raise ValueError('Columns not as expected.')

    target_is_binary = set(df['home_victory'].values).issubset({0, 1})
    if not target_is_binary:
        raise ValueError('Target is not binary.')

    features = BOX_SCORE_STATISTICS + ADDITIONAL_FEATURES
    for feature in features:
        for value in df[feature].values:
            if not isinstance(value, numbers.Number):
                print(value)
                raise ValueError('Non-numeric features found.')
    
    return True
        
def validate_odds_dataframe(df: pd.DataFrame, betting_simulation_length: int) -> bool:
    """Validate that the columns and data types contained in the dataframe are as expected.

    Args:
        df (pd.DataFrames): DataFrame containing preprocessed odds data.
        betting_simulation_length (int): size of the betting simulation data.

    Raises:
        ValueError:            
            - If the set of columns are not as expected.
            - If non-floating point odds values are found.
            - If odds values less than 1 are found.            
            - If there is a mismatch in the length of the odds and betting simulation dataframes.        

    Returns:
        bool: Flag to indicate whether validation test passed or not.  
    """    
    df = df.copy(deep=True)

    expected_columns = ['home_odds', 'away_odds']
    actual_columns = df.columns

    expected_columns_found = set(expected_columns) == set(actual_columns)
    if not expected_columns_found:
        raise ValueError('Columns not as expected.')
    
    values_floats = (df.dtypes == float)
    if not np.all(values_floats):
        raise TypeError('Not all odds are floats.')
    
    values_greater_than_1 = (df >= 1)
    if not np.all(values_greater_than_1):
        raise ValueError('Not all odds greater than or equal to 1.')

    if df.shape[0] != betting_simulation_length:
        raise ValueError('Length of odds dataframe does not match length of betting simulation data.')
    
    return True

def validate_modelling_ready_dataframe(df: pd.DataFrame, dropped_features: List[str]) -> bool:
    """Validate that the columns and data types contained in the dataframe are as expected.

    Args:
        df (pd.DataFrame): DataFrame containing modelling ready data.
        dropped_features (List[str]): Features dropped before modelling.

    Raises:
        ValueError:
            - If the set of columns are not as expected.
            - If non-floating point feature values are found.

    Returns:
        bool: Flag to indicate whether validation test passed or not.  
    """    
    df = df.copy(deep=True)

    expected_columns = [col for col in BOX_SCORE_STATISTICS + ADDITIONAL_FEATURES if col not in dropped_features]
    actual_columns = df.columns

    expected_columns_found = set(expected_columns) == set(actual_columns)
    if not expected_columns_found:
        raise ValueError('Columns not as expected.')
    
    values_floats = (df.dtypes == float)
    if not np.all(values_floats):
        raise TypeError('Not all feature values are floats.')

    return True

def validate_preprocessed_datasets(preprocessed_datasets: namedtuple, betting_simulation_length: int) -> bool:
    """Validate the preprocessed datasets to ensure their integrity.

    Args:
        preprocessed_datasets (namedtuple): namedtuple containing preprocessed datasets.
        betting_simulation_length (int): size of the betting simulation data.        

    Raises:
        ValueError:
            - If one or more null values are found in the dataframe.
            - If the set of columns are not as expected (for either odds or feature dataframes).
            - If the target is not a binary variable.
            - If one or more non-numeric features are found.
            - If non-floating point odds values are found.
            - If odds values less than 1 are found.            
            - If there is a mismatch in the length of the odds and betting simulation dataframes.                        
        
    Returns:
        bool: Flag to indicate whether the validation test has passed or not.
    """        
    for df_name in preprocessed_datasets._fields:
        df = getattr(preprocessed_datasets, df_name)
        if df_name == 'odds':
            validate_odds_dataframe(df, betting_simulation_length)
        else:
            validate_preprocessed_dataframe(df)

def validate_modelling_ready_datasets(modelling_ready_datasets: namedtuple) -> bool:
    """Validate the modelling ready datasets to ensure their integrity.

    Args:
        modelling_ready_datasets (namedtuple): namedtuple containing modelling ready datasets.

    Raises:
        ValueError:
            - If the set of columns are not as expected.
            - If non-floating point feature values are found.

    Returns:
        bool: Flag to indicate whether validation test passed or not.  
    """ 
    with open(os.path.join(".", "data", "output", "feature_selection", "drift_detection.json"), "r") as file:
        drift_detection_results = json.load(file)
        dropped_features = drift_detection_results['dropped']
    for df_name in modelling_ready_datasets._fields:
        df = getattr(modelling_ready_datasets, df_name)       
        validate_modelling_ready_dataframe(df, dropped_features)
