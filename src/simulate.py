import os 
import pandas as pd 
import numpy as np 
import json
import logging

from typing import Union, List, Dict, Tuple, Union
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from calibra.errors import classwise_ece


logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)


class BettingSimulation:
    """Class to run betting simulations for a given NBA season.

    Args:
        optimal_model (str):
            Optimal model to be used to generate predictions for betting simulation
        optimal_features (List[str]):
            Optimal feature set.
        optimal_hyperparameters (Dict[str, Union[str, int, float, Tuple[int]]]):
            Dictionary containing optimal hyperparameter values for given model.
        X_train_final (pd.DataFrame):
            Feature data for final training dataset.
        y_train_final (List[int]):
            Target data for final training dataset.
        X_betting_simulation (pd.DataFrame):
            Feature data for betting simulation dataset.       
        y_betting_simulation (List[int]):
            Target data for betting simulation dataset.
        odds (pd.DataFrame):
            Published odds data for given NBA season.
        metric (str):
            Model evaluation metric for which predictive model was selected.
        num_runs (int):
            Number of times the model is fitted (under different random seeds) and used to generate predictions (which are then averaged)
            to obtain final predictions for each game in the given NBA season.                 

    Attributes:
        optimal_model (str):
            Optimal model to be used to generate predictions for betting simulation
        optimal_features (List[str]):
            Optimal feature set.
        optimal_hyperparameters (Dict[str, Union[str, int, float, Tuple[int]]]):
            Dictionary containing optimal hyperparameter values for given model.
        X_train_final (pd.DataFrame):
            Feature data for final training dataset.
        y_train_final (List[int]):
            Target data for final training dataset.
        X_betting_simulation (pd.DataFrame):
            Feature data for betting simulation dataset.       
        y_betting_simulation (List[int]):
            Target data for betting simulation dataset.
        odds (pd.DataFrame):
            Odds data for given NBA season.
        metric (str):
            Model evaluation metric for which predictive model was selected.
        num_runs (int):
            Number of times the model is fitted (under different random seeds) and used to generate predictions (which are then averaged)
            to obtain final predictions for each game in the given NBA season. 
        INITIAL_BANKROLL (int):
            Bettor's starting bankroll for a given betting simulation. 
        FIXED_BETTING_STAKE (int):
            Stake to wager on each value bet throughout a fixed betting rule betting simulation.   
    """
    INITIAL_BANKROLL = 10_000
    FIXED_BETTING_STAKE = 100

    def __init__(self, 
                optimal_model: str,
                optimal_features: List[str],
                optimal_hyperparameters: Dict[str, Union[str, int, float]],
                X_train_final: pd.DataFrame,
                y_train_final: List[int],
                X_betting_simulation: pd.DataFrame,                
                y_betting_simulation: List[int],
                odds: pd.DataFrame,
                metric: str,
                num_runs: int = 1,                
        ):
        self.model_name = optimal_model
        self.optimal_hyperparameters = optimal_hyperparameters        
        self.X_train_final = X_train_final[optimal_features]
        self.X_betting_simulation = X_betting_simulation[optimal_features]
        self.y_train_final = y_train_final.copy()
        self.y_betting_simulation = y_betting_simulation.copy()
        self.odds = odds
        self.metric = metric
        self.num_runs = num_runs        

    def train_model(self, random_state: int = 0) -> Union[SVC, RandomForestClassifier, LogisticRegression, MLPClassifier]:
        """Train the final model for the betting simulation.

        Args:
            random_state (int):
                Random seed under which the model is fitted, to ensure reproducibility of results. 

        Returns:
            Union[SVC, RandomForestClassifier, LogisticRegression, MLPClassifier]:
                The final trained model.
        """
        model_dict = {
            'lr': LogisticRegression,
            'rf': RandomForestClassifier,
            'svm': SVC,
            'mlp': MLPClassifier,
            }
        hyperparameters = self.optimal_hyperparameters[self.model_name]
        if self.model_name == 'svm':
            hyperparameters["probability"] = True
        model = model_dict[self.model_name](**hyperparameters, random_state=random_state)
        model.fit(self.X_train_final, self.y_train_final)
        return model
    
    def generate_predictions(self):
        """Generate predictions for each NBA game in the betting simulation.
        """
        self.prediction_array = np.zeros((self.X_betting_simulation.shape[0], 2))

        for r in range(self.num_runs):
            model = self.train_model(r)
            predictions = model.predict_proba(self.X_betting_simulation)
            self.prediction_array += predictions

        self.prediction_array /= self.num_runs

    def record_metrics(self):
        """Record metrics achieved by the model for the given season.
        """
        predicted_labels = (self.prediction_array[:, 1] >= 0.5).astype(int)
        metric_dict = {
            'calibration_score': classwise_ece(self.prediction_array, self.y_betting_simulation),
            'accuracy': accuracy_score(self.y_betting_simulation, predicted_labels), 
        }
        
        with open(os.path.join(".", "data", "output", "central_hypothesis", f"{self.metric}_driven_betting_simulation_metrics.json"), "w") as file:
            json.dump(metric_dict, file, indent=4)

    def get_odds_and_probs(self, game_index: int) -> Dict[str, Tuple[float, float, float]]:
        """Get predicted and implied probability, and odds, for each team in a given game.

        Args:
            game_index (int): index of game under consideration.

        Returns:
            Dict[str, Tuple[float, float, float]]: 
                Dictionary containing tuple of predicted probability, odds, and implied probability for home and away team, respectively.
        """
        home_team_pred_prob = self.prediction_array[game_index, 1]
        home_team_odds = self.odds['home_odds'].iloc[game_index]
        home_team_implied_prob = 1 / home_team_odds

        away_team_pred_prob = self.prediction_array[game_index, 0]
        away_team_odds = self.odds['away_odds'].iloc[game_index]
        away_team_implied_prob = 1 / away_team_odds

        return {
            'home': (
                home_team_pred_prob, home_team_odds, home_team_implied_prob
            ),

            'away': (
                away_team_pred_prob, away_team_odds, away_team_implied_prob
            )
        }

    @staticmethod
    def is_value_bet(predicted_probability: float, implied_probability: float) -> bool:
        """Check if a value bet is present.

        Args:
            predicted_probability (float): Model's estimate of chance of victory.
            implied_probability (float): Implied probability of victory according to bookmaker's odds.

        Returns:
            bool: Flag to indicate whether or not a value bet is present. True if predicted probability is greater than implied probability. False otherwise.
        """
        return True if predicted_probability > implied_probability else False

    @staticmethod
    def calculate_kelly_criterion(predicted_probability: float, odds: float) -> float:
        """Calculate the Kelly criterion given the predicted probability and the odds of the given outcome.

        Args:
            predicted_probability (float): Model's estimate of chance of victory.
            odds (float): Total odds (including the original stake).

        Returns:
            float: Proportion of bankroll to wager
        """
        p = predicted_probability
        q = 1 - p
        b = odds - 1

        return (b * p - q) / b

    @staticmethod
    def calculate_stake(rule: str, bankroll: float, k: float, f: float) -> float:
        """Calculate the stake for the given bet.

        Args:
            rule (str): Betting Rule in use. Can be either "fixed", or "kelly".
            bankroll (float): Bettor's bankroll at given point in time.
            k (float): Kelly Criterion calculated for given bet.
            f (float): Fractional Kelly in use.

        Returns:
            float: Stake to wager on given outcome.
        """
        if rule == 'fixed':
            return BettingSimulation.FIXED_BETTING_STAKE if bankroll >= BettingSimulation.FIXED_BETTING_STAKE else bankroll 
        else:
            return f * k * bankroll 
        
    def save_results(
            self, 
            rule: str, 
            betting_history: Dict[int, Dict[str, Union[float, bool]]], 
            final_balance: float,
            test: bool = False
            ) -> Tuple[Dict[str, Union[float, int]]]:
        """Save the results of the given betting simulation.

        Args:
            rule (str): 
                Betting rule in use. Can be either "fixed", or "kelly".
            betting_history (Dict[int, Dict[str, Union[float, bool]]]):  
                Dictionary tracking all betting history over the given season.
            final_balance (float):
                Final balance of the bettor at season's end.
            test (bool): 
                Flag to indicate if method is being run as unit test (in which case, do not save results to file). 
                Defaults to False.
        
        Returns:
            Tuple[Dict[str, Union[float, int]]]:
                Results of betting simulation and central hypothesis, returned for testing purposes.
        """
        roi = 100 * (final_balance - BettingSimulation.INITIAL_BANKROLL) / BettingSimulation.INITIAL_BANKROLL

        num_games = self.prediction_array.shape[0]
        num_bets_made = 0
        num_bets_won = 0

        for game in betting_history.values():
            if game['home']['bet']:
                num_bets_made += 1
                if game['home']['win']:
                    num_bets_won += 1
            if game['away']['bet']:
                num_bets_made += 1
                if game['away']['win']:
                    num_bets_won += 1                

        bet_pct = 100 * num_bets_made / num_games
        win_pct = 100 * num_bets_won / num_bets_made 

        betting_simulation_results = {
            'final_balance': final_balance,
            'ROI': round(roi, 2),
        }

        central_hypothesis_results = {
            'bets_made': num_bets_made,
            'bets_won': num_bets_won,
            'bet_percentage': round(bet_pct, 2),
            'win_percentage': round(win_pct, 2),
        }

        if not test:
            with open(os.path.join(".", "data", "output", f"{rule}_betting", f"{self.metric}_driven_results.json"), "w") as file:
                json.dump(betting_simulation_results, file, indent=4)

            with open(os.path.join(".", "data", "output", "central_hypothesis", f"{self.metric}_driven_results.json"), "w") as file:
                json.dump(central_hypothesis_results, file, indent=4)
        
        logging.info(f'{self.metric}-driven {rule} betting system results:')
        logging.info(f'Made {central_hypothesis_results["bets_made"]} bets ({central_hypothesis_results["bet_percentage"]}% of games.)')
        logging.info(f'Won {central_hypothesis_results["bets_won"]} bets ({central_hypothesis_results["win_percentage"]}% of bets.)')
        logging.info(f'Final balance: {betting_simulation_results["final_balance"]}$')
        logging.info(f'Return on investment of: {betting_simulation_results["ROI"]}%')

        return betting_simulation_results, central_hypothesis_results # returning for testing purposes only

    def run_betting_simulation(self, rule: str, kelly_fraction: float = 0.125, save_results: bool = True) -> Tuple[List[float], Dict[str, List[float]]]:
        """Run the betting simulation under the given rule.

        Args:
            rule (str): Betting rule in use. Can be either "fixed", or "kelly".
            kelly_fraction (float): Fractional Kelly in use. 
            save_results (bool): Flag to determine whether or not to save the results to file. Defaults to True.

        Raises:
            ValueError: If invalid betting rule is given.

        Returns:
            Tuple[List[float], Dict[str, List[float]]]:
                Tuple containing bankroll_tracker (list of bettor's bankroll after each game) and value_bets (dictionary containing predicted and implied probability for each value bet identified).
        """
        if rule not in ['fixed', 'kelly']:
            raise ValueError('Betting Rule must be one of "fixed" or "kelly".')
        
        bankroll = BettingSimulation.INITIAL_BANKROLL
        bankroll_tracker = [bankroll]
        betting_history = {}
        
        # loop through games
        for game in range(self.prediction_array.shape[0]):
            if bankroll <= 0:
                break

            betting_history[game] = {
                'home': {
                    'bet': False,
                    'win': False
                },
                'away': {
                    'bet': False,
                    'win': False
                }          
            }
            odds_and_probs = self.get_odds_and_probs(game)

            home_pred_prob, home_odds, home_implied_prob = odds_and_probs['home']
            betting_history[game]['home']['predicted_probability'] = home_pred_prob
            betting_history[game]['home']['implied_probability'] = home_implied_prob
            betting_history[game]['home']['odds'] = home_odds
            if self.is_value_bet(home_pred_prob, home_implied_prob):
                betting_history[game]['home']['bet'] = True
                kelly_criterion = self.calculate_kelly_criterion(home_pred_prob, home_odds)
                stake = BettingSimulation.calculate_stake(rule, bankroll, kelly_criterion, kelly_fraction)
                profit = stake * (home_odds - 1)
                bankroll -= stake
                home_victory = self.y_betting_simulation[game]
                if home_victory:
                    betting_history[game]['home']['win'] = True
                    bankroll += stake + profit

            away_pred_prob, away_odds, away_implied_prob = odds_and_probs['away']
            betting_history[game]['away']['predicted_probability'] = away_pred_prob
            betting_history[game]['away']['implied_probability'] = away_implied_prob
            betting_history[game]['away']['odds'] = away_odds
            if self.is_value_bet(away_pred_prob, away_implied_prob):                
                betting_history[game]['away']['bet'] = True
                kelly_criterion = self.calculate_kelly_criterion(away_pred_prob, away_odds)
                stake = BettingSimulation.calculate_stake(rule, bankroll, kelly_criterion, kelly_fraction)
                profit = stake * (away_odds - 1)
                bankroll -= stake
                home_victory = self.y_betting_simulation[game]
                if not home_victory:
                    betting_history[game]['away']['win'] = True
                    bankroll += stake + profit

            bankroll_tracker.append(bankroll)

        final_balance = bankroll_tracker[-1]
        if save_results:
            self.save_results(rule, betting_history, final_balance)

        value_bets = {
            'predicted_probability': [],
            'implied_probability': []
        }

        for game in betting_history.values():
            if game['home']['bet']:
                value_bets['predicted_probability'].append(game['home']['predicted_probability'])
                value_bets['implied_probability'].append(game['home']['implied_probability'])
            if game['away']['bet']:
                value_bets['predicted_probability'].append(game['away']['predicted_probability'])
                value_bets['implied_probability'].append(game['away']['implied_probability'])

        return bankroll_tracker, value_bets
