import time 

from pipeline import NBABettingExperiment


def main(num_hpo_trials: int = 10, num_prediction_runs: int = 10):
    trial = NBABettingExperiment(num_hpo_trials=num_hpo_trials, num_prediction_runs=num_prediction_runs)
    preprocessed_datasets = trial.load_and_preprocess()
    modelling_ready_datasets = trial.extract_features(preprocessed_datasets)
    optimal_features = trial.select_optimal_features(preprocessed_datasets, modelling_ready_datasets)
    optimal_hyperparameters = trial.select_optimal_hyperparameters(preprocessed_datasets, modelling_ready_datasets, optimal_features)
    optimal_models = trial.select_optimal_models(preprocessed_datasets, modelling_ready_datasets, optimal_features, optimal_hyperparameters)
    betting_simulation_plotting_data = trial.run_betting_simulations(preprocessed_datasets, modelling_ready_datasets, optimal_features, optimal_hyperparameters, optimal_models)
    trial.analyse_results(betting_simulation_plotting_data)


if __name__ == '__main__':    
    start = time.time()
    main()
    end = time.time()
    print(f"NBA Betting Experiment took {round((end-start)/60)} minutes to run.")
