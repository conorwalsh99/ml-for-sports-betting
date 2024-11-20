import os 
import toml
import pandas as pd 
import numpy as np 
from tqdm import tqdm
from typing import List, Dict


config = toml.load(os.path.join('.', 'config.toml'))
GAME_ID_COLS = config['modelling']['GAME_IDENTIFIER_COLS']
ADDITIONAL_FEATURES = config['modelling']['ADDITIONAL_FEATURES']
TARGET = config['modelling']['TARGET']


class FeatureConstructor:
    """
    Class to perform feature engineering. 
    
    Given raw results and box score data, construct features to be used for predictive modelling.

    Args:
        box_score_data (pd.DataFrame):
            Pandas DataFrame containing Box score statistics for a given NBA season.
        previous_season_win_pcts (Dict[str, float]):
            Dictionary containing winning percentage of each team from the previous season.
            
    Attributes:
        box_score_data (pd.DataFrame):
            Pandas DataFrame containing Box score statistics for a given NBA season.
        previous_season_win_pcts (Dict[str, float]):
            Dictionary containing winning percentage of each team from the previous season.
        raw_differentials_df (pd.DataFrame):
            Dataframe containing raw differentials of box scores versus opponent for given game, for each game in the given season.
        average_box_scores_df (pd.DataFrame):
            DataFrame containing, for each game, the difference in box scores versus previous opponents averaged over the season to date.
            For each game, this includes values from all games leading up to, but not including, the game in question.
    """
    def __init__(self, 
                box_score_data: pd.DataFrame,                 
                previous_season_win_pcts: Dict[str, float] = None, 
        ):
        self.box_score_data = box_score_data
        self.previous_season_win_pcts = previous_season_win_pcts  

    def add_target(self):
        """
        Add target variable (home_victory) to dataframe.
        """
        home_victory = np.where(self.box_score_data['PTS.1'] > self.box_score_data['PTS'], 1, 0)
        self.box_score_data.insert(0, TARGET, home_victory)
        self.raw_differentials_df = self.box_score_data.copy(deep=True)    
        self.average_box_scores_df = self.box_score_data.copy(deep=True)        

    def get_raw_differentials(self, features: List[str] = None):
        """
        For each game in the given season, get the difference in each box score between the home and away teams.
        Replace the raw box score value with this 'difference versus opponent' value.        

        Args:
            features (List[str]):
                list of features for which we get difference in box scores between home and away teams.
        """
        for feature in features:
            self.raw_differentials_df[f'{feature}_difference'] = self.raw_differentials_df[f'home_{feature}'] - self.raw_differentials_df[f'away_{feature}']
            self.raw_differentials_df[f'home_{feature}'] = self.raw_differentials_df[f'{feature}_difference']
            self.raw_differentials_df[f'away_{feature}'] = self.raw_differentials_df[f'{feature}_difference'] * (-1)
            self.raw_differentials_df = self.raw_differentials_df.drop(columns=[f'{feature}_difference'])

    def get_average_box_scores(self, features: List[str] = None):
        """
        Average the box score data over the season to date for each game (up to but not including the given game).

        Args:
            features (List[str]):
                Features for which we wish to average box scores over the season to date.
        """
        home_team_average_box_scores_to_date = {
            feature: []             
            for feature in features
        }
        
        away_team_average_box_scores_to_date = {
            feature: []             
            for feature in features
        }
        
        box_score_df = self.raw_differentials_df.copy(deep=True)

        for i in tqdm(range(box_score_df.shape[0])):
            home_team = box_score_df.iloc[i]['Home/Neutral']
            away_team = box_score_df.iloc[i]['Visitor/Neutral']

            games_to_date_mask = box_score_df.index < i
            home_team_home_games_mask = box_score_df['Home/Neutral'] == home_team
            home_team_away_games_mask = box_score_df['Visitor/Neutral'] == home_team

            away_team_home_games_mask = box_score_df['Home/Neutral'] == away_team
            away_team_away_games_mask = box_score_df['Visitor/Neutral'] == away_team

            home_team_home_games_to_date = box_score_df[(games_to_date_mask & home_team_home_games_mask)]
            home_team_away_games_to_date = box_score_df[(games_to_date_mask & home_team_away_games_mask)]

            away_team_home_games_to_date = box_score_df[(games_to_date_mask & away_team_home_games_mask)]
            away_team_away_games_to_date = box_score_df[(games_to_date_mask & away_team_away_games_mask)]

            home_team_num_games_to_date = len(home_team_home_games_to_date) + len(home_team_away_games_to_date)
            if home_team_num_games_to_date > 0:
                for feature in features:
                    home_team_cumulative_value_to_date = home_team_home_games_to_date[f'home_{feature}'].sum() + home_team_away_games_to_date[f'away_{feature}'].sum()
                    home_team_average_value_to_date = home_team_cumulative_value_to_date / home_team_num_games_to_date
                    home_team_average_box_scores_to_date[feature].append(home_team_average_value_to_date)
            else:
                for feature in features:
                    home_team_average_box_scores_to_date[feature].append('No Previous Games')

            away_team_num_games_to_date = len(away_team_home_games_to_date) + len(away_team_away_games_to_date)
            if away_team_num_games_to_date > 0:
                for feature in features:
                    away_team_cumulative_value_to_date = away_team_home_games_to_date[f'home_{feature}'].sum() + away_team_away_games_to_date[f'away_{feature}'].sum()
                    away_team_average_value_to_date = away_team_cumulative_value_to_date / away_team_num_games_to_date
                    away_team_average_box_scores_to_date[feature].append(away_team_average_value_to_date)
            else:
                for feature in features:
                    away_team_average_box_scores_to_date[feature].append('No Previous Games')

        for feature in features:
            self.average_box_scores_df[f'home_{feature}'] = home_team_average_box_scores_to_date[feature]
            self.average_box_scores_df[f'away_{feature}'] = away_team_average_box_scores_to_date[feature]

    def add_previous_season_win_pct(self):
        """
        Add the the home and away teams' winning percentage from the previous season of as columns in the dataframe.
        """
        home_teams = self.box_score_data['Home/Neutral'].to_list()
        away_teams = self.box_score_data['Visitor/Neutral'].to_list()

        home_previous_win_pct = [self.previous_season_win_pcts[team] for team in home_teams]
        away_previous_win_pct = [self.previous_season_win_pcts[team] for team in away_teams]

        self.average_box_scores_df['home_previous_win_pct'] = home_previous_win_pct
        self.average_box_scores_df['away_previous_win_pct'] = away_previous_win_pct

    def get_averaged_differentials(self, features: List[str] = None):
        """
        Perform feature extraction by taking the difference between home and away values for each feature of interest.
        This reduces dimensionality of the problem. 

        Args:
            features (List[str]):
                Features for which we wish to get differentials between home and away average values.
        """
        full_feature_set = features + ADDITIONAL_FEATURES

        for feature in full_feature_set:
            self.average_box_scores_df[feature] = list(
                map(
                lambda x, y: 'No Previous Games' if (x == 'No Previous Games' or y == 'No Previous Games') else  x - y,
                self.average_box_scores_df[f'home_{feature}'], self.average_box_scores_df[f'away_{feature}']
                )
            )
            self.average_box_scores_df = self.average_box_scores_df.drop(columns=[f'home_{feature}', f'away_{feature}'])

    def construct_features(self, features: List[str] = None) -> pd.DataFrame:
        """
        Run feature construction.

        Args:
            features (List[str]):
                List of box score features to be used.

        Returns:
            pd.DataFrame:
                Pandas dataframe containing features for predictive modelling.
        """
        self.add_target()        
        self.get_raw_differentials(features) # skip this step to compute average box scores to-date instead of average difference in box scores to-date 
        self.get_average_box_scores(features)
        self.add_previous_season_win_pct()
        self.get_averaged_differentials(features)
    
        cols_to_keep = [TARGET] + GAME_ID_COLS + features + ADDITIONAL_FEATURES
        return self.average_box_scores_df[cols_to_keep]
