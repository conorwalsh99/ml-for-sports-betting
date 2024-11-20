import pandas as pd
import numpy as np 
import pytest 

from src.simulate import BettingSimulation


@pytest.fixture
def calculate_kelly_criterion_input():
    predicted_probability = 0.9
    odds = 2.0
    return predicted_probability, odds

@pytest.fixture
def calculate_kelly_criterion_expected():
    return 0.8 # calculated by hand

@pytest.fixture
def calculate_stake_fixed_normal_input():
    rule = 'fixed'
    bankroll = 9_000
    k = 0.8
    f = 0.125
    return rule, bankroll, k, f

@pytest.fixture
def calculate_stake_fixed_abnormal_input():
    rule = 'fixed'
    bankroll = 90
    k = 0.8
    f = 0.125
    return rule, bankroll, k, f

@pytest.fixture
def calculate_stake_fixed_normal_expected():
    return 100

@pytest.fixture
def calculate_stake_fixed_abnormal_expected():
    return 90

@pytest.fixture
def calculate_stake_kelly_input():
    rule = 'kelly'
    bankroll = 9_000
    k = 0.8
    f = 0.125
    return rule, bankroll, k, f

@pytest.fixture
def calculate_stake_kelly_expected():
    return 900 # calculated by hand

@pytest.fixture 
def save_results_input():  
    test = True  
    rule = 'fixed' 
    final_balance = 10_000
    betting_history = {
        0: {
            'home': {
                'predicted_probability': 0.9,
                'implied_probability': 3/4,
                'odds': 4/3,
                'bet': True,
                'win': False,

            },
            'away': {
                'predicted_probability': 0.1,
                'implied_probability': 1/3,
                'odds': 3,
                'bet': False,
                'win': False,
            },
        },

        1: {
            'home': {
                'predicted_probability': 0.3,
                'implied_probability': 0.5,
                'odds': 2,
                'bet': False,
                'win': False,

            },
            'away': {
                'predicted_probability': 0.7,
                'implied_probability': 1/3,
                'odds': 3,
                'bet': True,
                'win': True,
            },
        },
        2: {
            'home': {
                'predicted_probability': 0.55,
                'implied_probability': 3/4,
                'odds': 4/3,
                'bet': False,
                'win': False,
            },
            'away': {
                'predicted_probability': 0.45,
                'implied_probability': 0.5,
                'odds': 2,
                'bet': False,
                'win': False,
            },
        },        
    }

    return rule, betting_history, final_balance, test

@pytest.fixture
def save_results_expected():
    betting_simulation_results = {
        'final_balance': 10_000,
        'ROI': 0.00,
        }
    central_hypothesis_results = {
            'bets_made': 2,
            'bets_won': 1,
            'bet_percentage': 66.67,
            'win_percentage': 50.00
        }
    return betting_simulation_results, central_hypothesis_results

@pytest.fixture
def run_betting_simulation_normal_input():
    optimal_model = 'lr'
    optimal_features = ['feature_1', 'feature_2']
    optimal_hyperparameters = {            
        "solver": "liblinear",
        "C": 0.10231034090300602
    }        
    X_train_final = pd.DataFrame(
        {
            'feature_1': [0, 1, 2],
            'feature_2': [9, 6, 7]
        }
    )
    y_train_final = [1, 0, 1]
    X_betting_simulation = pd.DataFrame(
        {
            'feature_1': [0, 1, 2],
            'feature_2': [9, 6, 7]
        }
    )                 
    y_betting_simulation = [0, 0, 1]
    prediction_array = np.asarray(
        [
            [0.1, 0.9],
            [0.7, 0.3],
            [0.45, 0.55]
        ]
    )
    odds = pd.DataFrame(
        {
            'home_odds': [4/3, 2, 4/3],
            'away_odds': [3, 2, 2],
        }
    ) 
    metric = 'accuracy'
    num_runs = 1

    return (
        optimal_model,
        optimal_features,
        optimal_hyperparameters,
        X_train_final,
        y_train_final,        
        X_betting_simulation,                
        y_betting_simulation,
        odds,
        metric,
        num_runs,
        prediction_array
    )

@pytest.fixture
def run_betting_simulation_normal_expected():
    bankroll_tracker = [10_000, 9_900, 10_000, 10_000]
    value_bets = {
        'predicted_probability': [0.9, 0.7],
        'implied_probability': [3/4, 1/2]
    }
    return bankroll_tracker, value_bets

@pytest.fixture
def get_odds_and_probs_expected():
    return {
        'home': (
            0.55,  4/3, 3/4
        ),

        'away': (
            0.45, 2, 0.5
        )
    }


class TestBettingSimulation:
    def test_generate_predictions(self, run_betting_simulation_normal_input):
        (
        optimal_model,
        optimal_features,
        optimal_hyperparameters,
        X_train_final,
        y_train_final,        
        X_betting_simulation,                
        y_betting_simulation,
        odds,
        metric,
        num_runs,
        prediction_array
        ) = run_betting_simulation_normal_input
        
        optimal_hyperparameters = {
            'lr': optimal_hyperparameters
        }        

        simulation = BettingSimulation(optimal_model,
        optimal_features,
        optimal_hyperparameters,
        X_train_final,
        y_train_final,        
        X_betting_simulation,                
        y_betting_simulation,
        odds,
        metric,
        num_runs
        )
        simulation.generate_predictions()
        predictions = simulation.prediction_array
        assert isinstance(predictions, np.ndarray), f'predictions is not a numpy array.'
        assert np.all((predictions >= 0) & (predictions <= 1)), 'Found predicted probabilities outside of interval [0, 1].'

    def test_get_odds_and_probs(self, run_betting_simulation_normal_input, get_odds_and_probs_expected):
        (
        optimal_model,
        optimal_features,
        optimal_hyperparameters,
        X_train_final,
        y_train_final,        
        X_betting_simulation,                
        y_betting_simulation,
        odds,
        metric,
        num_runs,
        prediction_array
        ) = run_betting_simulation_normal_input
        
        optimal_hyperparameters = {
            'lr': optimal_hyperparameters
        }        

        simulation = BettingSimulation(optimal_model,
        optimal_features,
        optimal_hyperparameters,
        X_train_final,
        y_train_final,        
        X_betting_simulation,
        y_betting_simulation,
        odds,
        metric,
        num_runs
        )
        simulation.prediction_array = prediction_array

        game_index = 2
        result = simulation.get_odds_and_probs(game_index)
        expected = get_odds_and_probs_expected
        assert result == expected, f'get_odds_and_probs returned {result} instead of {expected}.'

    def test_is_value_bet_true(self):
        predicted_probability = 0.9
        implied_probability = 0.8
        message = 'Failed to identify value.'
        assert BettingSimulation.is_value_bet(predicted_probability=predicted_probability, implied_probability=implied_probability), message

    def test_is_value_bet_false(self):
        predicted_probability = 0.7
        implied_probability = 0.8
        message = 'Incorrectly to identify value.'
        assert not BettingSimulation.is_value_bet(predicted_probability=predicted_probability, implied_probability=implied_probability), message

    def test_calculate_kelly_criterion(self, calculate_kelly_criterion_input, calculate_kelly_criterion_expected):
        result = BettingSimulation.calculate_kelly_criterion(*calculate_kelly_criterion_input)
        expected = calculate_kelly_criterion_expected
        assert result == expected, f'Calculated kelly criterion incorrectly.'

    def test_calculate_stake_fixed_normal(self, calculate_stake_fixed_normal_input, calculate_stake_fixed_normal_expected):        
        result = BettingSimulation.calculate_stake(*calculate_stake_fixed_normal_input)
        expected = calculate_stake_fixed_normal_expected
        assert result == expected, f'BettingSimulation.calculate_stake returned {result} instead of {expected}.' 

    def test_calculate_stake_fixed_abnormal(self, calculate_stake_fixed_abnormal_input, calculate_stake_fixed_abnormal_expected):        
        result = BettingSimulation.calculate_stake(*calculate_stake_fixed_abnormal_input)
        expected = calculate_stake_fixed_abnormal_expected
        assert result == expected, f'BettingSimulation.calculate_stake returned {result} instead of {expected}.' 

    def test_calculate_stake_kelly(self, calculate_stake_kelly_input, calculate_stake_kelly_expected):        
        result = BettingSimulation.calculate_stake(*calculate_stake_kelly_input)
        expected = calculate_stake_kelly_expected
        assert result == expected, f'BettingSimulation.calculate_stake returned {result} instead of {expected}.'

    def test_save_results(self, run_betting_simulation_normal_input, save_results_input, save_results_expected):
        (
        optimal_model,
        optimal_features,
        optimal_hyperparameters,
        X_train_final,
        y_train_final,        
        X_betting_simulation,                
        y_betting_simulation,
        odds,
        metric,
        num_runs,
        prediction_array
        ) = run_betting_simulation_normal_input

        simulation = BettingSimulation(optimal_model,
        optimal_features,
        optimal_hyperparameters,
        X_train_final,
        y_train_final,        
        X_betting_simulation,                
        y_betting_simulation,
        odds,
        metric,
        num_runs
        )
        simulation.prediction_array = prediction_array
        betting_simulation_results_result, central_hypothesis_results_result = simulation.save_results(*save_results_input)
        betting_simulation_results_expected, central_hypothesis_results_expected = save_results_expected
        message = f'BettingSimulation.save_results returned {betting_simulation_results_result} instead of {betting_simulation_results_expected}.'
        assert betting_simulation_results_result == betting_simulation_results_expected, message
        message = f'BettingSimulation.save_results returned {central_hypothesis_results_result} instead of {central_hypothesis_results_expected}.'
        assert central_hypothesis_results_result == central_hypothesis_results_expected, message

    def test_run_betting_simulation_fixed_normal(self, run_betting_simulation_normal_input, run_betting_simulation_normal_expected):
        (
        optimal_model,
        optimal_features,
        optimal_hyperparameters,
        X_train_final,
        y_train_final,        
        X_betting_simulation,                
        y_betting_simulation,
        odds,
        metric,
        num_runs,
        prediction_array
        ) = run_betting_simulation_normal_input

        simulation = BettingSimulation(optimal_model,
        optimal_features,
        optimal_hyperparameters,
        X_train_final,
        y_train_final,        
        X_betting_simulation,                
        y_betting_simulation,
        odds,
        metric,
        num_runs
        )
        simulation.prediction_array = prediction_array

        bankroll_tracker_result, value_bets_result = simulation.run_betting_simulation(rule='fixed', save_results=False)
        bankroll_tracker_expected, value_bets_expected = run_betting_simulation_normal_expected
        assert bankroll_tracker_result == bankroll_tracker_expected, f'BettingSimulation.run_betting_simulation() returned wrong bankroll_tracker: {bankroll_tracker_result} instead of {bankroll_tracker_expected}'
        assert value_bets_result == value_bets_expected, f'BettingSimulation.run_betting_simulation() returned wrong value_bets: {value_bets_result} instead of {value_bets_expected}'
