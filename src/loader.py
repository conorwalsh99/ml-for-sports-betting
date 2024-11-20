import os
import pandas as pd  
import json
from typing import Optional, Dict


class Loader:
    """
    Class to load data.
    """
    def __init__(self, year: Optional[int] = None):
        self.year = year

    def load_box_scores(self) -> pd.DataFrame:
        """
        Load box score data.

        Returns:
            pd.DataFrame: box score data for given season.
        """
        return pd.read_csv(os.path.join('.', 'data', 'input', f'results_and_box_scores_{self.year}.csv'))

    def load_previous_season_win_pct_dict(self) -> Dict[str, float]:
        """
        Load dictionary containing the previous season winning percentage of each team.

        Returns:
            Dict[str, float]: Dictionary containing previous season winning percentage for each team.
        """
        with open(os.path.join('.', 'data', 'input', f'win_pct_{self.year - 1}.json'), 'rb') as file:
            return json.load(file)
            
    def load_odds(self, path: Optional[str] = None) -> pd.DataFrame:
        """Load odds data.

        Args:
            path (str, optional): Optional file path of odds data. Defaults to None.

        Returns:
            pd.DataFrame: Pandas dataframe containing odds data.
        """    
        return pd.read_csv(path) if path else pd.read_csv(os.path.join('.','data', 'input', 'nba odds 2018-19.csv'))
