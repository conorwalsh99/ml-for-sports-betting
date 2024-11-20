import pytest
import pandas as pd
import numpy as np 

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from src.hpo import HPO


@pytest.fixture
def get_model_and_validation_data():
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)    
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    X_train = pd.DataFrame(X_train, columns=feature_names)
    X_test = pd.DataFrame(X_test, columns=feature_names)
    good_model = LogisticRegression()
    good_model.fit(X_train, y_train)
    return good_model, X_train, X_test, y_train, y_test, feature_names

@pytest.fixture
def get_bin_constraint_unfriendly_input():
    # With only 10 samples in the data set, the bin constraint will always be broken.
    X, y = make_classification(n_samples=10, n_features=5, n_informative=4, n_redundant=0, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)    
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    X_train = pd.DataFrame(X_train, columns=feature_names)
    X_test = pd.DataFrame(X_test, columns=feature_names)
    good_model = LogisticRegression()
    good_model.fit(X_train, y_train)
    return good_model, X_train, X_test, y_train, y_test, feature_names

@pytest.fixture
def get_bin_constraint_unfriendly_expected():
    return 1

@pytest.fixture
def extract_optimal_hyperparameters_input():
        return {
            0: {
                'score': 0.9,
                'hyperparameters': {
                    'C': 1.2,
                    'degree': 3,
                    'kernel': 'rbf',
                }
            },

            1: {
                'score': 0.03,
                'hyperparameters': {
                    'C': 1.5,
                    'degree': 2,
                    'kernel': 'linear',
                }
            },

            2: {
                'score': 0.5,
                'hyperparameters': {
                    'C': 1.0,
                    'degree': 5,
                    'kernel': 'rbf',
                }
            }
        }

@pytest.fixture
def extract_optimal_hyperparameters_expected_accuracy():
    return {
        'C': 1.2,
        'degree': 3,
        'kernel': 'rbf',
    }

@pytest.fixture
def extract_optimal_hyperparameters_expected_calibration():
    return {
        'C': 1.5,
        'degree': 2,
        'kernel': 'linear',
    }


class TestHPO:
    def test_get_score_normal_accuracy(self, get_model_and_validation_data):
        model, X_train, X_validate, y_train, y_validate, features = get_model_and_validation_data
        hpo = HPO(X_train, X_validate, y_train, y_validate, features, 'accuracy', 1, 1)
        result = hpo.get_score(model)
        assert result >= 0.7, 'Sanity check failed.'

    def test_get_score_normal_calibration(self, get_model_and_validation_data):
        model, X_train, X_validate, y_train, y_validate, features = get_model_and_validation_data
        hpo = HPO(X_train, X_validate, y_train, y_validate, features, 'calibration', 1, 1)
        result = hpo.get_score(model)
        assert result <= 0.2, 'Sanity check failed.'

    def test_get_score_bin_constraint_not_met(self, get_bin_constraint_unfriendly_input, get_bin_constraint_unfriendly_expected):
        model, X_train, X_validate, y_train, y_validate, features = get_bin_constraint_unfriendly_input
        hpo = HPO(X_train, X_validate, y_train, y_validate, features, 'calibration', 1, 1)
        result = hpo.get_score(model)
        expected = get_bin_constraint_unfriendly_expected
        assert result == expected, f'HPO.get_score returned {result} instead of {expected}.'

    def test_extract_optimal_hyperparameters_accuracy(self, get_model_and_validation_data, extract_optimal_hyperparameters_input, extract_optimal_hyperparameters_expected_accuracy):
        model, X_train, X_validate, y_train, y_validate, features = get_model_and_validation_data
        hpo = HPO(X_train, X_validate, y_train, y_validate, features, 'accuracy', 3, 3)
        trials_dict = extract_optimal_hyperparameters_input
        result = hpo.extract_optimal_hyperparameters(trials_dict)
        expected = extract_optimal_hyperparameters_expected_accuracy
        assert result == expected, f'HPO.extract_optimal_hyperparameters returned {result} instead of {expected}.'

    def test_extract_optimal_hyperparameters_calibration(self, get_model_and_validation_data, extract_optimal_hyperparameters_input, extract_optimal_hyperparameters_expected_calibration):
        model, X_train, X_validate, y_train, y_validate, features = get_model_and_validation_data
        hpo = HPO(X_train, X_validate, y_train, y_validate, features, 'calibration', 3, 3)
        trials_dict = extract_optimal_hyperparameters_input
        result = hpo.extract_optimal_hyperparameters(trials_dict)
        expected = extract_optimal_hyperparameters_expected_calibration
        assert result == expected, f'HPO.extract_optimal_hyperparameters returned' 
