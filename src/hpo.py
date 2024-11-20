import os
import toml
import multiprocessing
import pandas as pd
import numpy as np
import json
from typing import Tuple, Dict, Union, List

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from calibra.errors import classwise_ece
from calibra.utils import bin_probabilities, get_classwise_bin_weights
from hyperopt import fmin, hp, tpe, STATUS_OK, Trials


config = toml.load(os.path.join('.', 'config.toml'))


class HPO:
    """Class to run hyperparameter optimisation.

    Given the optimal feature set, datasets, and evaluation metric, run bayesian optimisation using a Tree-parzen estimator to find the optimal hyperparameter values for each of the candidate models.

    Args:
        X_train (pd.DataFrame):
            Feature data for training set.
        X_validate (pd.DataFrame):
            Feature data for validation set.
        y_train (List[int]):
            Target data for training set.
        y_validate (List[int]):
            Target data for validation set.
        optimal_features (List[str]):
            Optimal feature set.
        metric (str):
            Model evaluation metric. 
        num_trials (int):
            Number of trial of Bayesian Optimisation to run.
        num_scoring_runs (int):
            Number of times to fit and score model (using different random seeds) once optimal hyperparameters have been found by BO, in order to evaluate model performance under given hyperparameters.

    Attributes:
        X_train (pd.DataFrame):
            Feature data for training set.
        X_validate (pd.DataFrame):
            Feature data for validation set.
        y_train (List[int]):
            Target data for training set.
        y_validate (List[int]):
            Target data for validation set.
        optimal_features (List[str]):
            Optimal feature set.
        metric (str):
            Model evaluation metric. 
        num_trials (int):
            Number of trial of Bayesian Optimisation to run.
        num_scoring_runs (int):
            Number of times to fit and score model (using different random seeds) once optimal hyperparameters have been found by BO, in order to evaluate model performance under given hyperparameters.
        lr_hp_search_space (Dict[str, Union[hp.lognormal, hp.choice]]):
            Search space of hyperparameters for Logistic Regression model.
        rf_hp_search_space (Dict[str, hp.choice]):
            Search space of hyperparameters for Random Forest model.
        svm_hp_search_space (Dict[str, Union[hp.uniform, hp.choice]]):
            Search space of hyperparameters for Support Vector Machine model.
        mlp_hp_search_space (Dict[str, Union[hp.loguniform, hp.choice]]):
            Search space of hyperparameters for Multi-Layer Perceptron model.
    """

    def __init__(
        self,
        X_train: pd.DataFrame,
        X_validate: pd.DataFrame,
        y_train: List[int],
        y_validate: List[int],
        optimal_features: List[str],
        metric: str,
        num_trials: int,
        num_scoring_runs: int,
    ):
        self.X_train = X_train[optimal_features]
        self.X_validate = X_validate[optimal_features]
        self.y_train = y_train.copy()
        self.y_validate = y_validate.copy()
        self.optimal_features = optimal_features
        self.metric = metric
        self.num_trials = num_trials
        self.num_scoring_runs = num_scoring_runs
        self.define_search_spaces()

    def define_search_spaces(self):
        """Define the hyperparameter search spaces for each model"""

        self.lr_hp_search_space = {
            "C": hp.lognormal("C", 0, 1.0),
            "solver": hp.choice("solver", ["liblinear", "lbfgs"]),
        }
        self.rf_hp_search_space = {
            "criterion": hp.choice("criterion", ["gini", "entropy"]),
            "max_depth": hp.choice("max_depth", list(range(5, 51))),
            "max_features": hp.choice("max_features", list(range(1, len(self.optimal_features) + 1))),
            "min_samples_leaf": hp.choice("min_samples_leaf", list(range(1, 12))),
            "min_samples_split": hp.choice("min_samples_split", list(range(2, 12))),
            "n_estimators": hp.choice("n_estimators", list(range(10, 101))),
        }
        self.svm_hp_search_space = {
            "C": hp.uniform("C", 0.1, 50),
            "kernel": hp.choice("kernel", ["linear", "poly", "rbf", "sigmoid"]),
            "degree": hp.choice("degree", [2, 3, 4]),
        }
        self.mlp_hp_search_space = {
            "hidden_layer_sizes": hp.choice(
                "hidden_layer_sizes",
                [
                    (3),
                    (4),
                    (5),
                    (6),
                    (3, 3),
                    (3, 4),
                    (3, 5),
                    (3, 6),
                    (4, 3),
                    (4, 4),
                    (4, 5),
                    (4, 6),
                    (5, 3),
                    (5, 4),
                    (5, 5),
                    (5, 6),
                    (6, 3),
                    (6, 4),
                    (6, 5),
                    (6, 6),
                ],
            ),
            "activation": hp.choice(
                "activation", ["identity", "logistic", "tanh", "relu"]
            ),
            "solver": hp.choice("solver", ["lbfgs", "sgd", "adam"]),
            "alpha": hp.loguniform("alpha", np.log(0.0001), np.log(0.1)),
            "batch_size": hp.choice("batch_size", [32, 64, 128]),
            "learning_rate": hp.choice(
                "learning_rate", ["constant", "invscaling", "adaptive"]
            ),
            "learning_rate_init": hp.loguniform(
                "learning_rate_init", np.log(0.001), np.log(0.1)
            ),
        }

    def get_score(self, model: Union[LogisticRegression, RandomForestClassifier, SVC, MLPClassifier]) -> float:
        """Evaluate the performance of the given model on the validation data under the given metric.

        If calibration is the metric, enforce constraint that at least 80% of bins must be non-empty.

        Args:
            model (Union[LogisticRegression, RandomForestClassifier, SVC, MLPClassifier]):
                Classifier under evaluation.

        Returns:
            float: 
                Score of model under given metric.
        """
        if self.metric == "calibration":
            num_bins = 20
            y_pred_proba = model.predict_proba(self.X_validate)
            binned_probabilities = bin_probabilities(y_pred_proba, self.y_validate, num_bins=num_bins)
            bin_weights = get_classwise_bin_weights(binned_probabilities)[0]
            non_empty_bins = [1 for weight in bin_weights if weight > 0]
            if sum(non_empty_bins) >= 0.8 * num_bins:
                return classwise_ece(y_pred_proba, self.y_validate, num_bins)
            return 1
        return model.score(self.X_validate, self.y_validate)

    def lr_objective(self, hp_search_space: Dict[str, Union[hp.lognormal, hp.choice]]) -> Dict[str, Union[float, str]]:
        """Evaluate the objective function to be minimised when running hyperparameter optimisation for the logistic regression model.

        Args:
            hp_search_space (Dict[str, Union[hp.lognormal, hp.choice]]):
                Search space of hyperparameters for Logistic Regression Model.
        Returns:
            Dict[str, Union[float, str]]:
                Loss and status of run.
        """
        model = LogisticRegression(
            C=hp_search_space["C"], solver=hp_search_space["solver"]
        )

        model.fit(self.X_train, self.y_train)
        sign = -1 if self.metric == "accuracy" else 1
        loss = self.get_score(model) * sign

        return {"loss": loss, "status": STATUS_OK}

    def rf_objective(self, hp_search_space: Dict[str, hp.choice]) -> Dict[str, Union[float, str]]:
        """Evaluate the objective function to be minimised when running hyperparameter optimisation for the random forest model.

        Args:
            hp_search_space (Dict[str, hp.choice]):
                Search space of hyperparameters for the Random Forest model.
        Returns:
            Dict[str, Union[float, str]]
                Loss and status of run.
        """
        model = RandomForestClassifier(
            criterion=hp_search_space["criterion"],
            max_depth=hp_search_space["max_depth"],
            max_features=hp_search_space["max_features"],
            min_samples_leaf=hp_search_space["min_samples_leaf"],
            min_samples_split=hp_search_space["min_samples_split"],
            n_estimators=hp_search_space["n_estimators"],
        )

        model.fit(self.X_train, self.y_train)
        sign = -1 if self.metric == "accuracy" else 1
        loss = self.get_score(model) * sign

        return {"loss": loss, "status": STATUS_OK}

    def svm_objective(self, hp_search_space: Dict[str, Union[hp.uniform, hp.choice]]) -> Dict[str, Union[float, str]]:
        """Evaluate the objective function to be minimised when running hyperparameter optimisation for the support vector machine model.

        Args:
            hp_search_space (Dict[str, Union[hp.uniform, hp.choice]]):
                Search space of hyperparameters for Support Vector Machine model.
        Returns:
            Dict[str, Union[float, str]]
                Loss and status of run.
        """
        model = SVC(
            C=hp_search_space["C"],
            kernel=hp_search_space["kernel"],
            degree=hp_search_space["degree"],
            probability=True,
        )

        model.fit(self.X_train, self.y_train)
        sign = -1 if self.metric == "accuracy" else 1
        loss = self.get_score(model) * sign

        return {"loss": loss, "status": STATUS_OK}

    def mlp_objective(self, hp_search_space: Dict[str, Union[hp.choice, hp.loguniform]]) -> Dict[str, Union[float, str]]:
        """Evaluate the objective function to be minimised when running hyperparameter optimisation for the multi-layer perceptron model.

        Args:
            hp_search_space (Dict[str, Union[hp.choice, hp.uniform]]):
                Search space of hyperparameters for Multi-Layer Perceptron model.

        Returns:
            Dict[str, Union[float, str]]:
                Loss and status of run.
        """
        model = MLPClassifier(
            hidden_layer_sizes=hp_search_space["hidden_layer_sizes"],
            activation=hp_search_space["activation"],
            solver=hp_search_space["solver"],
            alpha=hp_search_space["alpha"],
            batch_size=hp_search_space["batch_size"],
            learning_rate=hp_search_space["learning_rate"],
            learning_rate_init=hp_search_space["learning_rate_init"],
        )

        model.fit(self.X_train, self.y_train)
        sign = -1 if self.metric == "accuracy" else 1
        loss = self.get_score(model) * sign

        return {"loss": loss, "status": STATUS_OK}

    def run_single_lr_trial(self, i: int) -> Tuple[int, Dict[str, Union[float, Dict[str, Union[str, float]]]]]:
        """Run single trial of bayesian hyperparameter optimisation.

        Args:
            i (int): iterable to determine the random seed for the trial

        Returns:
            Tuple[int, Dict[str, Union[float, Dict[str, Union[str, float]]]]]: 
                Tuple of iterable used, and results of trial (best hyperparameter values and corresponding average score)
        """            
        np.random.seed(i)
        solvers = ["liblinear", "lbfgs"]

        trials = Trials()
        best = fmin(
            fn=self.lr_objective,
            space=self.lr_hp_search_space,
            algo=tpe.suggest,
            max_evals=64,
            trials=trials,
            rstate=np.random.default_rng(i),
        )

        trials_dict = {
            "hyperparameters": {
                "C": best["C"], # best['C'] is optimal value of C
                "solver": solvers[best['solver']] # best['solver'] is index of solver in search space
            }
        }

        cumulative_score = 0
        for r in range(self.num_scoring_runs):
            np.random.seed(r)
            best_lr = LogisticRegression(**trials_dict['hyperparameters'], random_state=r)
            best_lr.fit(self.X_train, self.y_train)
            score = self.get_score(best_lr)
            cumulative_score += score 
        average_score = cumulative_score / self.num_scoring_runs
        trials_dict["score"] = average_score

        return i, trials_dict

    def run_lr_trials(self) -> Dict[int, Dict[str, Union[float, Dict[str, Union[str, float]]]]]:
        """Run the bayesian optimisation trials for the logistic regression model.

        Returns:
            Dict[int, Dict[str, Union[float, Dict[str, Union[str, float]]]]]:
                Dictionary containing best hyperparameter values found for each trial run and corresponding average score of model on validation data.
        """ 
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            results = pool.map(self.run_single_lr_trial, range(self.num_trials))

        return {i: result for i, result in results}

    def run_single_rf_trial(self, i: int) -> Tuple[int, Dict[str, Union[float, Dict[str, Union[str, float]]]]]:
        """Run single trial of bayesian hyperparameter optimisation.

        Args:
            i (int): iterable to determine the random seed for the trial

        Returns:
            Tuple[int, Dict[str, Union[float, Dict[str, Union[str, float]]]]]: 
                Tuple of iterable used, and results of trial (best hyperparameter values and corresponding average score)
        """            
        np.random.seed(i)

        rf_hp_dict = {
            "criterion": ["gini", "entropy"],
            "max_depth": list(range(5, 51)),
            "max_features": list(range(1, len(self.optimal_features) + 1)),
            "min_samples_leaf": list(range(1, 12)),
            "min_samples_split": list(range(2, 12)),
            "n_estimators": list(range(10, 101)),
        }        

        trials = Trials()
        best = fmin(
            fn=self.rf_objective,
            space=self.rf_hp_search_space,
            algo=tpe.suggest,
            max_evals=64,
            trials=trials,
            rstate=np.random.default_rng(i),
        )
        
        # when search space defined by hp.choice, best[hyperparameter] will return index of best value in search space
        trials_dict = {
            "hyperparameters": {
                hyperparameter: rf_hp_dict[hyperparameter][best[hyperparameter]]
                for hyperparameter in rf_hp_dict
            }
        }

        cumulative_score = 0
        for r in range(self.num_scoring_runs):
            np.random.seed(r)
            best_rf = RandomForestClassifier(**trials_dict["hyperparameters"], random_state=r)
            best_rf.fit(self.X_train, self.y_train)
            score = self.get_score(best_rf)
            cumulative_score += score
        average_score = cumulative_score / self.num_scoring_runs
        trials_dict["score"] = average_score

        return i, trials_dict

    def run_rf_trials(self) -> Dict[int, Dict[str, Union[float, Dict[str, Union[str, int]]]]]:
        """Run the bayesian optimisation trials for the random forest model.

        Returns:
            Dict[int, Dict[str, Union[float, Dict[str, Union[str, int]]]]]:
                Dictionary containing best hyperparameter values found for each trial run and corresponding average score of model on validation data.
        """         
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            results = pool.map(self.run_single_rf_trial, range(self.num_trials))

        return {i: result for i, result in results}


    def run_single_svm_trial(self, i: int) -> Tuple[int, Dict[str, Union[float, Dict[str, Union[str, float]]]]]:
        """Run single trial of bayesian hyperparameter optimisation.

        Args:
            i (int): iterable to determine the random seed for the trial

        Returns:
            Tuple[int, Dict[str, Union[float, Dict[str, Union[str, float]]]]]: 
                Tuple of iterable used, and results of trial (best hyperparameter values and corresponding average score)
        """  
        np.random.seed(i)

        svm_hp_dict = {
            "degree": [2, 3, 4],
            "kernel": ["linear", "poly", "rbf", "sigmoid"],
        }

        trials = Trials()
        best = fmin(
            fn=self.svm_objective,
            space=self.svm_hp_search_space,
            algo=tpe.suggest,
            max_evals=64,
            trials=trials,
            rstate=np.random.default_rng(i),
        )

        trials_dict = {"hyperparameters": {}} 
        trials_dict["hyperparameters"]["C"] = best['C']
        trials_dict["hyperparameters"]["degree"] = svm_hp_dict['degree'][best['degree']]                                
        trials_dict["hyperparameters"]["kernel"] = svm_hp_dict["kernel"][best['kernel']]

        cumulative_score = 0            
        for r in range(self.num_scoring_runs):
            np.random.seed(r)
            best_svm = SVC(**trials_dict['hyperparameters'], random_state=r, probability=True)
            best_svm.fit(self.X_train, self.y_train)
            score = self.get_score(best_svm)
            cumulative_score += score
        average_score = cumulative_score / self.num_scoring_runs
        trials_dict["score"] = average_score
        
        return i, trials_dict

    def run_svm_trials(self) -> Dict[int, Dict[str, Union[float, Dict[str, Union[str, float, int]]]]]:
        """Run the bayesian optimisation trials for the suport vector machine model.

        Returns:
            Dict[int, Dict[str, Union[float, Dict[str, Union[str, float, int]]]]]:
                Dictionary containing best hyperparameter values found for each trial run and corresponding average score of model on validation data.
        """        
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            results = pool.map(self.run_single_svm_trial, range(self.num_trials))

        return {i: result for i, result in results}        

    def run_single_mlp_trial(self, i: int) -> Tuple[int, Dict[str, Union[float, Dict[str, Union[str, float]]]]]:
        """Run single trial of bayesian hyperparameter optimisation.

        Args:
            i (int): iterable to determine the random seed for the trial

        Returns:
            Tuple[int, Dict[str, Union[float, Dict[str, Union[str, float]]]]]: 
                Tuple of iterable used, and results of trial (best hyperparameter values and corresponding average score)
        """  
        np.random.seed(i)

        mlp_hp_dict = {
            "hidden_layer_sizes": [
                (3),
                (4),
                (5),
                (6),
                (3, 3),
                (3, 4),
                (3, 5),
                (3, 6),
                (4, 3),
                (4, 4),
                (4, 5),
                (4, 6),
                (5, 3),
                (5, 4),
                (5, 5),
                (5, 6),
                (6, 3),
                (6, 4),
                (6, 5),
                (6, 6),
            ],
            "activation": ["identity", "logistic", "tanh", "relu"],
            "solver": ["lbfgs", "sgd", "adam"],
            "batch_size": [32, 64, 128],
            "learning_rate": ["constant", "invscaling", "adaptive"],
        }

        trials = Trials()
        best = fmin(
            fn=self.mlp_objective,
            space=self.mlp_hp_search_space,
            algo=tpe.suggest,
            max_evals=64,
            trials=trials,
            rstate=np.random.default_rng(i),
        )

        trials_dict = {"hyperparameters": {}}            
        trials_dict["hyperparameters"]["alpha"] = best["alpha"]
        trials_dict["hyperparameters"]["learning_rate_init"] = best["learning_rate_init"]
        for hyperparameter in ["hidden_layer_sizes", "activation", "solver", "batch_size", "learning_rate"]:
            trials_dict["hyperparameters"][hyperparameter] = mlp_hp_dict[hyperparameter][best[hyperparameter]]

        cumulative_score = 0
        for r in range(self.num_scoring_runs):
            np.random.seed(r)
            best_mlp = MLPClassifier(**trials_dict["hyperparameters"], random_state=r)
            best_mlp.fit(self.X_train, self.y_train)
            score = self.get_score(best_mlp)
            cumulative_score += score
        average_score = cumulative_score / self.num_scoring_runs
        trials_dict["score"] = average_score

        return i, trials_dict


    def run_mlp_trials(self) -> Dict[int, Dict[str, Union[float, Dict[str, Union[str, Tuple[int], int, float]]]]]:
        """Run the bayesian optimisation trials for the multi-layer perceptron model.

        Returns:
            Dict[int, Dict[str, Union[float, Dict[str, Union[str, Tuple[int], int, float]]]]]:
                Dictionary containing best hyperparameter values found for each trial run and corresponding average score of model on validation data.
        """
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            results = pool.map(self.run_single_mlp_trial, range(self.num_trials))

        return {i: result for i, result in results}

    def extract_optimal_hyperparameters(
            self, 
            trials_dict: Dict[int, Dict[str, Union[float, Dict[str, Union[str, float, int, Tuple[int]]]]]]            
        ) -> Dict[str, Union[str, float, int, Tuple[int]]]:
        """Extract the hyperparameter values associated with the best trial.

        Args:
            trials_dict (Dict[int, Dict[str, Union[float, Dict[str, Union[str, float, int, Tuple[int]]]]]]): 
                Dictionary containing best hyperparameter values found for each trial run and corresponding average score of model on validation data.
        
        Returns:
            Dict[str, Union[str, float, int, Tuple[int]]]:
                Dictionary containing the optimal hyperparameter values.
        """
        sign = -1 if self.metric == "accuracy" else 1

        losses = []
        for trial in trials_dict:
            losses.append(trials_dict[trial]["score"] * sign)
        
        best_loss = min(losses)

        for trial in trials_dict:
            if trials_dict[trial]["score"] * sign == best_loss:
                return trials_dict[trial]["hyperparameters"]

    def save_optimal_hyperparameters(self, optimal_hyperparameters_dict: Dict[str, Dict[str, Union[str, float, int, Tuple[int]]]]):
        """Save optimal hyperparameter values to file.

        Args:
            optimal_hyperparameters_dict (Dict[str, Dict[str, Union[str, float, int, Tuple[int]]]]): 
                Dictionary containing optimal hyperparameter values (under the given metric) for each model.
        """        
        json_str = json.dumps(optimal_hyperparameters_dict, indent=4, default=str)

        file_path = os.path.join(".", "data", "output", "hpo", f"{self.metric}_driven_optimal_hyperparameters.json")
            
        with open(file_path, "w") as file:
            file.write(json_str)
        
    def run_hpo(self) -> Dict[str, Dict[str, Union[str, float, int, Tuple[int]]]]:
        """Run the bayesian optimisation to find the optimal hyperparameter values for each of the candidate models.
        
        Returns:
            Dict[str, Dict[str, Union[str, float, int, Tuple[int]]]]:
                Dictionary containing the optimal hyperparameter values for each of the candidate models.
        """        
        lr_trials_dict = self.run_lr_trials()
        rf_trials_dict = self.run_rf_trials()
        svm_trials_dict = self.run_svm_trials()
        mlp_trials_dict = self.run_mlp_trials()

        optimal_hyperparameters_dict = {}
        optimal_hyperparameters_dict['lr'] = self.extract_optimal_hyperparameters(lr_trials_dict)
        optimal_hyperparameters_dict['rf'] = self.extract_optimal_hyperparameters(rf_trials_dict)    
        optimal_hyperparameters_dict['svm'] = self.extract_optimal_hyperparameters(svm_trials_dict)        
        optimal_hyperparameters_dict['mlp'] = self.extract_optimal_hyperparameters(mlp_trials_dict)
        
        self.save_optimal_hyperparameters(optimal_hyperparameters_dict)
        return optimal_hyperparameters_dict
    