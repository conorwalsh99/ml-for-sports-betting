import os
import toml
import json
import logging
import pandas as pd
import numpy as np 
from typing import List 
from collections import namedtuple

from loader import Loader
from feature_engineering import FeatureConstructor
from utils import (
    remove_preliminary_games, 
    match_odds_to_games, 
    validate_preprocessed_datasets, 
    validate_modelling_ready_datasets,
    )
from feature_extraction import FeatureExtractor
from feature_selection import FeatureSelector
from hpo import HPO
from model_selection import ModelSelector
from simulate import BettingSimulation
from plotting import BettingSimulationPlotter


config = toml.load(os.path.join('.', 'config.toml'))
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)


class NBABettingExperiment:
    """
    Class to run an NBA betting experiment.
     
    This experiment aims to compare the success of betting systems that select the predictive model based on accuracy, to those that select the predictive model based on calibration.

    Args:
        initial_training_years (List[int]):
            Seasons comprising the initial training data set (used to fit models during feature selection and hyperparameter optimisation).
        validation_years (List[int]):
            Seasons comprising the validation data set (used to evaluate models during feature selection and hyperparameter optimisation).
        test_years (List[int]):
            Season(s) comprising the test data set (used to evaluate models to select best model ahead of betting simulation).
        betting_simulation_years (List[int]):
            Season(s) comprising the betting simulation data set (used to run betting simulation and obtain experiment results).
        num_hpo_trials (int):
            Number of Bayesian Optimisation trials to run to find optimal hyperparameters for each model
        num_prediction_runs (int):
            Number of times to fit model and generate predictions before taking average prediction (to ensure stability of prediction).

    Attributes:
        initial_training_years (List[int]):
            Seasons comprising the initial training data set (used to fit models during feature selection and hyperparameter optimisation).
        validation_years (List[int]):
            Seasons comprising the validation data set (used to evaluate models during feature selection and hyperparameter optimisation).
        extended_training_years (List[int]):
            Season(s) comprising the extended training data set (used to fit models ahead of model selection).            
        test_years (List[int]):
            Season(s) comprising the test data set (used to evaluate models to select best model ahead of betting simulation).
        final_training_years (List[int]):
            Season(s) comprising the final training data set (used to fit models ahead of betting simulation).            
        betting_simulation_years (List[int]):
            Season(s) comprising the betting simulation data set (used to run betting simulation and obtain experiment results).
        num_hpo_trials (int):
            Number of Bayesian Optimisation trials to run to find optimal hyperparameters for each model
        num_prediction_runs (int):
            Number of times to fit model and generate predictions before taking average prediction (to ensure stability of prediction).
    """
    def __init__(
            self,
            initial_training_years: List[int] = [2015, 2016],
            validation_years: List[int] = [2017],
            test_years: List[int] = [2018],
            betting_simulation_years: List[int] = [2019],
            num_hpo_trials: int = 10,
            num_prediction_runs: int = 10,
    ):
        self.initial_training_years = initial_training_years
        self.validation_years = validation_years
        self.extended_training_years = self.initial_training_years + self.validation_years
        self.test_years = test_years
        self.final_training_years = self.extended_training_years + self.test_years
        self.betting_simulation_years = betting_simulation_years
        self.num_hpo_trials = num_hpo_trials
        self.num_prediction_runs = num_prediction_runs
        
        logging.info('Beginning experiment.')
        logging.info(f'Initial training set seasons: {self.initial_training_years}')
        logging.info(f'Validation set season(s): {self.validation_years}')
        logging.info(f'Extended training set seasons: {self.extended_training_years}')
        logging.info(f'Test set season(s): {self.test_years}')
        logging.info(f'Final training set seasons: {self.final_training_years}')
        logging.info(f'Betting simulation season(s): {self.betting_simulation_years}')
        

    # STEP 1. LOAD AND CLEAN DATA
    def load_and_preprocess(self) -> namedtuple:
        """
        Load the box score data, construct the features for predictive modelling, and remove the preliminary games from each season.
        Then, load the odds data and convert the format from moneyline to decimal odds.
        
        Returns:
            namedtuple:
                namedtuple containing pandas dataframes for initial training, validation, extended training, test, final training, betting simulation and odds data.
        """
        initial_training_data = pd.DataFrame()
        for year in self.initial_training_years:
            loader = Loader(year)
            box_score_data = loader.load_box_scores()
            previous_season_records = loader.load_previous_season_win_pct_dict()
            feature_constructor = FeatureConstructor(box_score_data=box_score_data,
                                                    previous_season_win_pcts=previous_season_records)
            feature_data = feature_constructor.construct_features(features=config['modelling']['BOX_SCORE_STATISTICS'])
            feature_data_informed_games = remove_preliminary_games(feature_data)
            logging.info(f'Dropped {feature_data.shape[0] - feature_data_informed_games.shape[0]} preliminary games from {year} season.')
            initial_training_data = pd.concat([initial_training_data, feature_data_informed_games], axis=0)
        logging.info(f'{initial_training_data.shape[0]} data points in initial training set.')
        
        validation_data = pd.DataFrame()
        for year in self.validation_years:
            loader = Loader(year)
            box_score_data = loader.load_box_scores()
            previous_season_records = loader.load_previous_season_win_pct_dict()
            feature_constructor = FeatureConstructor(box_score_data=box_score_data,
                                                    previous_season_win_pcts=previous_season_records)
            feature_data = feature_constructor.construct_features(features=config['modelling']['BOX_SCORE_STATISTICS'])
            feature_data_informed_games = remove_preliminary_games(feature_data)  
            logging.info(f'Dropped {feature_data.shape[0] - feature_data_informed_games.shape[0]} preliminary games from {year} season.')          
            validation_data = pd.concat([validation_data, feature_data_informed_games], axis=0)
        logging.info(f'{validation_data.shape[0]} data points in validation set.')

        extended_training_data = pd.concat([initial_training_data, validation_data], axis=0)
        logging.info(f'{extended_training_data.shape[0]} data points in extended training set.')
        
        test_data = pd.DataFrame()
        for year in self.test_years:
            loader = Loader(year)
            box_score_data = loader.load_box_scores()
            previous_season_records = loader.load_previous_season_win_pct_dict()
            feature_constructor = FeatureConstructor(box_score_data=box_score_data,
                                                    previous_season_win_pcts=previous_season_records)
            feature_data = feature_constructor.construct_features(features=config['modelling']['BOX_SCORE_STATISTICS'])
            feature_data_informed_games = remove_preliminary_games(feature_data) 
            logging.info(f'Dropped {feature_data.shape[0] - feature_data_informed_games.shape[0]} preliminary games from {year} season.')           
            test_data = pd.concat([test_data, feature_data_informed_games], axis=0)
        logging.info(f'{test_data.shape[0]} data points in test set.')

        final_training_data = pd.concat([extended_training_data, test_data], axis=0)
        logging.info(f'{final_training_data.shape[0]} data points in final training set.')        

        betting_simulation_data = pd.DataFrame()
        for year in self.betting_simulation_years:
            loader = Loader(year)
            box_score_data = loader.load_box_scores()
            previous_season_records = loader.load_previous_season_win_pct_dict()
            feature_constructor = FeatureConstructor(box_score_data=box_score_data,
                                                    previous_season_win_pcts=previous_season_records)
            feature_data = feature_constructor.construct_features(features=config['modelling']['BOX_SCORE_STATISTICS'])
            feature_data_informed_games = remove_preliminary_games(feature_data)  
            logging.info(f'Dropped {feature_data.shape[0] - feature_data_informed_games.shape[0]} preliminary games from {year} season.')          
            betting_simulation_data = pd.concat([betting_simulation_data, feature_data_informed_games], axis=0)
        logging.info(f'{betting_simulation_data.shape[0]} data points in betting simulation set.')
        
        odds_data = loader.load_odds()     
        betting_simulation_data, odds_data = match_odds_to_games(betting_simulation_data, odds_data)
        logging.info(f'{betting_simulation_data.shape[0]} games in betting simulation set for which we have published odds.\n')

        Datasets = namedtuple('Datasets', 'initial_training validation extended_training test final_training betting_simulation odds')
        preprocessed_datasets = Datasets(initial_training_data, 
                                        validation_data, 
                                        extended_training_data,
                                        test_data,
                                        final_training_data,
                                        betting_simulation_data,
                                        odds_data) 
        
        # sanity check
        validate_preprocessed_datasets(preprocessed_datasets, betting_simulation_data.shape[0])
    
        return preprocessed_datasets 
          
    # STEP 2 - FEATURE EXTRACTION
    def extract_features(self, preprocessed_datasets: namedtuple) -> namedtuple:
        """
        Extract features from the preprocessed data by removing features showing signs of covariate shift and scaling the data.
        Return the datasets, now ready for modelling, in a namedtuple.

        Args:
            preprocessed_datasets (namedtuple):
                namedtuple containing each of the datasets after the preprocessing stage.     

        Returns:
            namedtuple:
                namedtuple containing a pandas DataFrame for each of the modelling-ready datasets
        """        
        feature_extractor = FeatureExtractor(
            preprocessed_datasets.initial_training, 
            preprocessed_datasets.validation, 
            preprocessed_datasets.extended_training,
            preprocessed_datasets.test,
            preprocessed_datasets.final_training,
            preprocessed_datasets.betting_simulation,
        )
        
        (
            initial_training_data, 
            validation_data, 
            extended_training_data,
            test_data,
            final_training_data,
            betting_simulation_data
        ) = feature_extractor.extract_features()

        ModellingReadyDatasets = namedtuple('ModellingReadyDatasets', 'initial_training validation extended_training test final_training betting_simulation')
        modelling_ready_datasets = ModellingReadyDatasets(
            initial_training_data, 
            validation_data, 
            extended_training_data,
            test_data,
            final_training_data,
            betting_simulation_data,            
        )

        # sanity check
        validate_modelling_ready_datasets(modelling_ready_datasets)

        return modelling_ready_datasets

    # STEP 3 - FEATURE SELECTION
    def select_optimal_features(self, preprocessed_datasets: namedtuple, modelling_ready_datasets: namedtuple) -> namedtuple:
        """
        Select the optimal features for the predictive modelling problem.

        Args:
            preprocessed_datasets (namedtuple):
                Data sets after preprocessing. Needed to retreive the target variable from initial training and validation data sets.  
            modelling_ready_datasets (namedtuple):
                Data sets after feature extraction. Needed to retrieve the modelling-ready feature data for training and validation data sets.       

        Returns:
            optimal_features (namedtuple):
                Optimal accuracy-driven and calibration-driven feature sets.         
        """
        initial_training_data_X, validation_data_X = modelling_ready_datasets.initial_training, modelling_ready_datasets.validation
        initial_training_data_y, validation_data_y = preprocessed_datasets.initial_training['home_victory'].to_list(), preprocessed_datasets.validation['home_victory'].to_list()
        
        accuracy_feature_selector = FeatureSelector(initial_training_data_X, validation_data_X, initial_training_data_y, validation_data_y, metric='accuracy')
        calibration_feature_selector = FeatureSelector(initial_training_data_X, validation_data_X, initial_training_data_y, validation_data_y, metric='calibration')

        accuracy_optimal_features = accuracy_feature_selector.run_feature_selection() 
        logging.info(f'Optimal features for accuracy-driven predictive modelling: {accuracy_optimal_features}')   
        calibration_optimal_features = calibration_feature_selector.run_feature_selection()
        logging.info(f'Optimal features for calibration-driven predictive modelling: {calibration_optimal_features}')           

        OptimalFeatures = namedtuple('OptimalFeatures', 'accuracy_driven calibration_driven')
        optimal_features = OptimalFeatures(accuracy_optimal_features, calibration_optimal_features)

        return optimal_features    
        
    # STEP 4 - HYPER-PARAMETER OPTIMISATION
    def select_optimal_hyperparameters(
        self, 
        preprocessed_datasets: namedtuple, 
        modelling_ready_datasets: namedtuple, 
        optimal_features: namedtuple,
        ) -> namedtuple:
        """Select the optimal hyperparameter values for each candidate learning algorithm in the predictive modelling problem.

        Args:
            preprocessed_datasets (namedtuple):
                Data sets after preprocessing. Needed to retreive the target variable from initial training and validation data sets.  
            modelling_ready_datasets (namedtuple):
                Data sets after feature extraction. Needed to retrieve the modelling-ready feature data for training and validation data sets.  
            optimal_features (namedtuple):
                Optimal feature sets for accuracy-driven and calibration-driven predictive modelling. Needed for further predictive modelling steps.      

        Returns:
            optimal_hyperparameters (namedtuple):
                Optimal accuracy-driven and calibration-driven hyperparameter values for each candidate learning algorithm.         
        """
        initial_training_data_X, validation_data_X = modelling_ready_datasets.initial_training, modelling_ready_datasets.validation
        initial_training_data_y, validation_data_y = preprocessed_datasets.initial_training['home_victory'].to_list(), preprocessed_datasets.validation['home_victory'].to_list()
        accuracy_optimal_features, calibration_optimal_features = optimal_features.accuracy_driven, optimal_features.calibration_driven 
                
        accuracy_hpo = HPO(initial_training_data_X, validation_data_X, initial_training_data_y, validation_data_y, accuracy_optimal_features, metric='accuracy', num_trials=self.num_hpo_trials, num_scoring_runs=self.num_prediction_runs)
        calibration_hpo = HPO(initial_training_data_X, validation_data_X, initial_training_data_y, validation_data_y, calibration_optimal_features, metric='calibration', num_trials=self.num_hpo_trials, num_scoring_runs=self.num_prediction_runs)

        logging.info(f'\nRunning hyperparameter optimisation with {self.num_hpo_trials} trials and {self.num_prediction_runs} prediction runs.')
        logging.info('Accuracy-driven hyperparameter optimisation:')
        accuracy_optimal_hyperparameters = accuracy_hpo.run_hpo()
        logging.info(f'Optimal accuracy-driven hyperparameters: {accuracy_optimal_hyperparameters}')
        logging.info('Calibration-driven hyperparameter optimisation:')
        calibration_optimal_hyperparameters = calibration_hpo.run_hpo()
        logging.info(f'Optimal calibration-driven hyperparameters: {calibration_optimal_hyperparameters}')

        OptimalHyperparameters = namedtuple('OptimalHyperparameters', 'accuracy_driven calibration_driven')
        optimal_hyperparameters = OptimalHyperparameters(accuracy_optimal_hyperparameters, calibration_optimal_hyperparameters)

        return optimal_hyperparameters


    # STEP 5 - MODEL SELECTION
    def select_optimal_models(
        self, 
        preprocessed_datasets: namedtuple, 
        modelling_ready_datasets: namedtuple, 
        optimal_features: namedtuple, 
        optimal_hyperparameters: namedtuple
        ) -> namedtuple:
        """Select the optimal models for the predictive modelling problem.

        Args:
            preprocessed_datasets (namedtuple):
                Data sets after preprocessing. Needed to retreive the target variable from extended training and test data sets.
            modelling_ready_datasets (namedtuple):
                Data sets after feature extraction. Needed to retrieve the modelling-ready feature data for extended training and test data sets.  
            optimal_features (namedtuple):
                Optimal feature sets for accuracy-driven and calibration-driven predictive modelling. 
            optimal_hyperparameters (namedtuple):
                Optimal hyperparameter values for accuracy-driven and calibration-driven predictive modelling.                 

        Returns:
            optimal_models (namedtuple):
                Optimal accuracy-driven and calibration-driven models.
        """
        extended_training_data_X, test_data_X = modelling_ready_datasets.extended_training, modelling_ready_datasets.test
        extended_training_data_y, test_data_y = preprocessed_datasets.extended_training['home_victory'].to_list(), preprocessed_datasets.test['home_victory'].to_list()
        accuracy_optimal_features, calibration_optimal_features = optimal_features.accuracy_driven, optimal_features.calibration_driven 
        accuracy_optimal_hyperparameters, calibration_optimal_hyperparameters = optimal_hyperparameters.accuracy_driven, optimal_hyperparameters.calibration_driven 
                
        accuracy_model_selection = ModelSelector(extended_training_data_X, test_data_X, extended_training_data_y, test_data_y, accuracy_optimal_features, accuracy_optimal_hyperparameters, metric='accuracy', num_scoring_runs=self.num_prediction_runs)
        calibration_model_selection = ModelSelector(extended_training_data_X, test_data_X, extended_training_data_y, test_data_y, calibration_optimal_features, calibration_optimal_hyperparameters, metric='calibration', num_scoring_runs=self.num_prediction_runs)

        logging.info('Running accuracy-driven model selection.')
        accuracy_optimal_model = accuracy_model_selection.run_model_selection()    
        logging.info('Running calibration-driven model selection.')
        calibration_optimal_model = calibration_model_selection.run_model_selection()

        OptimalModels = namedtuple('OptimalModels', 'accuracy_driven calibration_driven')
        optimal_models = OptimalModels(accuracy_optimal_model, calibration_optimal_model)

        return optimal_models

    
    # STEP 6 - BETTING SIMULATION
    def run_betting_simulations(
        self, 
        preprocessed_datasets: namedtuple, 
        modelling_ready_datasets: namedtuple, 
        optimal_features: namedtuple, 
        optimal_hyperparameters: namedtuple,
        optimal_models: namedtuple,
        ) -> namedtuple:
        """Run betting simulations for the given NBA season.

        Args:
            preprocessed_datasets (namedtuple):
                Data sets after preprocessing. Needed to retreive the target variable for final training and betting simulation data set
                as well as the odds data for the betting simulation.
            modelling_ready_datasets (namedtuple):
                Data sets after feature extraction. Needed to retrieve the modelling-ready feature data 
                for final training and betting simulation data sets.  
            optimal_features (namedtuple):
                Optimal feature sets for accuracy-driven and calibration-driven predictive modelling. 
            optimal_hyperparameters (namedtuple):
                Optimal hyperparameter values for accuracy-driven and calibration-driven predictive modelling.
            optimal_models (namedtuple):
                Optimal models for accuracy-driven and calibration-driven predictive modelling.

        Returns:
            namedtuple:
                For each betting simulation, list of bankrolls at each point of the season, as well as dictionary containing predicted and implied probabilities for each value bet identified. 
        """
        final_training_data_X, betting_simulation_data_X = modelling_ready_datasets.final_training, modelling_ready_datasets.betting_simulation
        final_training_data_y, betting_simulation_data_y = preprocessed_datasets.final_training['home_victory'].to_list(), preprocessed_datasets.betting_simulation['home_victory'].to_list()
        odds = preprocessed_datasets.odds
        accuracy_optimal_features, calibration_optimal_features = optimal_features.accuracy_driven, optimal_features.calibration_driven 
        accuracy_optimal_hyperparameters, calibration_optimal_hyperparameters = optimal_hyperparameters.accuracy_driven, optimal_hyperparameters.calibration_driven 
        accuracy_optimal_model, calibration_optimal_model = optimal_models.accuracy_driven, optimal_models.calibration_driven
       
        accuracy_driven_betting_simulation = BettingSimulation(
            accuracy_optimal_model,
            accuracy_optimal_features,
            accuracy_optimal_hyperparameters,
            final_training_data_X,
            final_training_data_y,
            betting_simulation_data_X,
            betting_simulation_data_y,
            odds,
            metric = 'accuracy',
            num_runs=self.num_prediction_runs,
        )

        calibration_driven_betting_simulation = BettingSimulation(
            calibration_optimal_model,
            calibration_optimal_features,
            calibration_optimal_hyperparameters,
            final_training_data_X,
            final_training_data_y,
            betting_simulation_data_X,
            betting_simulation_data_y,
            odds,
            metric='calibration',
            num_runs=self.num_prediction_runs,
        )
        
        accuracy_driven_betting_simulation.generate_predictions()
        calibration_driven_betting_simulation.generate_predictions()
        
        accuracy_driven_betting_simulation.record_metrics()
        calibration_driven_betting_simulation.record_metrics()

        logging.info('Running accuracy-driven fixed betting simulation.')
        accuracy_fixed_betting_bankroll, accuracy_value_bets = accuracy_driven_betting_simulation.run_betting_simulation(rule='fixed')
        logging.info('Running accuracy-driven kelly betting simulation.')
        accuracy_kelly_betting_bankroll, accuracy_value_bets = accuracy_driven_betting_simulation.run_betting_simulation(rule='kelly')

        logging.info('Running calibration-driven fixed betting simulation.')
        calibration_fixed_betting_bankroll, calibration_value_bets = calibration_driven_betting_simulation.run_betting_simulation(rule='fixed')
        logging.info('Running calibration-driven kelly betting simulation.')
        calibration_kelly_betting_bankroll, calibration_value_bets = calibration_driven_betting_simulation.run_betting_simulation(rule='kelly')

        plotting_data = {
            'calibration_fixed_betting_bankroll': calibration_fixed_betting_bankroll,
            'calibration_kelly_betting_bankroll': calibration_kelly_betting_bankroll, 
            'calibration_value_bets': calibration_value_bets,
            'accuracy_fixed_betting_bankroll': accuracy_fixed_betting_bankroll,
            'accuracy_kelly_betting_bankroll': accuracy_kelly_betting_bankroll,
            'accuracy_value_bets': accuracy_value_bets
        }        

        with open(os.path.join('.', 'data', 'intermediate', 'betting_simulation_plotting_data.json'), 'w') as file:
            json.dump(plotting_data, file, indent=4)

        BettingSimulationPlottingData = namedtuple(
            'BettingSimulationPlottingData',
            'calibration_bankroll_fixed calibration_bankroll_kelly calibration_value_bets accuracy_bankroll_fixed accuracy_bankroll_kelly accuracy_value_bets'
            )

        betting_simulation_plotting_data = BettingSimulationPlottingData(
            calibration_fixed_betting_bankroll,
            calibration_kelly_betting_bankroll, 
            calibration_value_bets,
            accuracy_fixed_betting_bankroll,
            accuracy_kelly_betting_bankroll,
            accuracy_value_bets
        )

        return betting_simulation_plotting_data

    # STEP 7 - REPORT EXPERIMENT RESULTS
    def analyse_results(self, betting_simulation_plotting_data: namedtuple):
        """Plot graphs to analyse the results of the betting simulations.

        Args:
            betting_simulation_plotting_data (namedtuple): 
                namedtuple containing bankroll and value bets data from the betting simulations.
        """
        (
        calibration_fixed_betting_bankroll,
        calibration_kelly_betting_bankroll, 
        calibration_value_bets,
        accuracy_fixed_betting_bankroll,
        accuracy_kelly_betting_bankroll,
        accuracy_value_bets
        ) = (
        betting_simulation_plotting_data.calibration_bankroll_fixed,
        betting_simulation_plotting_data.calibration_bankroll_kelly, 
        betting_simulation_plotting_data.calibration_value_bets,
        betting_simulation_plotting_data.accuracy_bankroll_fixed,
        betting_simulation_plotting_data.accuracy_bankroll_kelly,
        betting_simulation_plotting_data.accuracy_value_bets
        )
        
        fixed_betting_simulation_plotter = BettingSimulationPlotter(
            calibration_fixed_betting_bankroll,
            accuracy_fixed_betting_bankroll,
            calibration_value_bets,
            accuracy_value_bets,
            rule='fixed'
        )
        
        kelly_betting_simulation_plotter = BettingSimulationPlotter(
            calibration_kelly_betting_bankroll,
            accuracy_kelly_betting_bankroll,
            calibration_value_bets,
            accuracy_value_bets,
            rule='kelly'
        )

        fixed_betting_simulation_plotter.plot_bankrolls()
        kelly_betting_simulation_plotter.plot_bankrolls()
        
        fixed_betting_simulation_plotter.plot_value_bet_distributions('calibration')
        fixed_betting_simulation_plotter.plot_value_bet_distributions('accuracy')         

        with open(os.path.join(".", "data", "output", "fixed_betting", "accuracy_driven_results.json"), "rb") as file:
            accuracy_driven_fixed_betting_results = json.load(file)
        with open(os.path.join(".", "data", "output", "kelly_betting", "accuracy_driven_results.json"), "rb") as file:
            accuracy_driven_kelly_betting_results = json.load(file)
        with open(os.path.join(".", "data", "output", "fixed_betting", "calibration_driven_results.json"), "rb") as file:
            calibration_driven_fixed_betting_results = json.load(file)
        with open(os.path.join(".", "data", "output", "kelly_betting", "calibration_driven_results.json"), "rb") as file:
            calibration_driven_kelly_betting_results = json.load(file)

        max_calibration_roi = np.maximum(
            calibration_driven_fixed_betting_results['ROI'], calibration_driven_kelly_betting_results['ROI']
        )
        average_calibration_roi = np.mean(
            [calibration_driven_fixed_betting_results['ROI'], calibration_driven_kelly_betting_results['ROI']]
        )        
        max_accuracy_roi = np.maximum(
            accuracy_driven_fixed_betting_results['ROI'], accuracy_driven_kelly_betting_results['ROI']
        )
        average_accuracy_roi = np.mean(
            [accuracy_driven_fixed_betting_results['ROI'], accuracy_driven_kelly_betting_results['ROI']]
        )

        roi_results = {
            'calibration': {
                'mean': average_calibration_roi,
                'max': max_calibration_roi,
            },
            'accuracy': {
                'mean': average_accuracy_roi,
                'max': max_accuracy_roi,
            }
        }

        with open(os.path.join(".", "data", "output", "central_hypothesis", "accuracy_driven_betting_simulation_metrics.json"), "rb") as file:
            accuracy_driven_model_metrics = json.load(file)
        with open(os.path.join(".", "data", "output", "central_hypothesis", "calibration_driven_betting_simulation_metrics.json"), "rb") as file:
            calibration_driven_model_metrics = json.load(file)

        logging.info(f'Accuracy-driven betting systems:')
        logging.info(f'Accuracy: {round(100 * accuracy_driven_model_metrics["accuracy"], 2)}%')
        logging.info(f'Class-wise ECE: {round(100 * accuracy_driven_model_metrics["calibration_score"], 2)}%')
        logging.info(f'Mean return on investment: {roi_results["accuracy"]["mean"]}%')
        logging.info(f'Max return on investment: {roi_results["accuracy"]["max"]}%\n')

        logging.info(f'Calibration-driven betting systems:')
        logging.info(f'Accuracy: {round(100 * calibration_driven_model_metrics["accuracy"], 2)}%')
        logging.info(f'Class-wise ECE: {round(100 * calibration_driven_model_metrics["calibration_score"], 2)}%')
        logging.info(f'Mean return on investment: {roi_results["calibration"]["mean"]}%')
        logging.info(f'Max return on investment: {roi_results["calibration"]["max"]}%')

        with open(os.path.join(".", "data", "output", "central_hypothesis", "roi.json"), "w") as file:
            json.dump(roi_results, file, indent=4)

        value_bet_prediction_variances = {
            'calibration': np.var(calibration_value_bets['predicted_probability']),
            'accuracy': np.var(accuracy_value_bets['predicted_probability']),
            }
        
        with open(os.path.join(".", "data", "output", "value_bets", "value_bets_prediction_variances.json"), "w") as file:
            json.dump(value_bet_prediction_variances, file, indent=4)
