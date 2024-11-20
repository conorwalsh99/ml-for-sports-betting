import pytest
import pandas as pd
import numpy as np 

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from src.model_selection import ModelSelector


@pytest.fixture
def get_model_and_validation_data():
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)    
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    hyperparameters = {            
        "solver": "liblinear",
        "C": 0.10231034090300602
    }    
    X_train = pd.DataFrame(X_train, columns=feature_names)
    X_test = pd.DataFrame(X_test, columns=feature_names)
    good_model = LogisticRegression(**hyperparameters)
    good_model.fit(X_train, y_train)
    return good_model, X_train, X_test, y_train, y_test, feature_names, hyperparameters

@pytest.fixture
def get_bin_constraint_unfriendly_input():
    # With only 10 samples in the data set, the bin constraint will always be broken.
    X, y = make_classification(n_samples=10, n_features=5, n_informative=4, n_redundant=0, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)    
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    hyperparameters = {            
        "solver": "liblinear",
        "C": 0.10231034090300602
    }        
    X_train = pd.DataFrame(X_train, columns=feature_names)
    X_test = pd.DataFrame(X_test, columns=feature_names)
    good_model = LogisticRegression(**hyperparameters)
    good_model.fit(X_train, y_train)
    return good_model, X_train, X_test, y_train, y_test, feature_names, hyperparameters

@pytest.fixture
def get_bin_constraint_unfriendly_expected():
    return 1

@pytest.fixture
def extract_best_model_input():
    return {
        'lr': 0.99,
        'rf': 0.5,
        'svm': 0.5,
        'mlp': 0.01,
    }

@pytest.fixture
def extract_best_model_expected_accuracy():
    return 'lr'

@pytest.fixture
def extract_best_model_expected_calibration():
    return 'mlp'


class TestModelSelector:
    def test_get_score_normal_accuracy(self, get_model_and_validation_data):
        model, X_train, X_validate, y_train, y_validate, features, hyperparameters = get_model_and_validation_data
        model_selector = ModelSelector(X_train, X_validate, y_train, y_validate, features, hyperparameters, 'accuracy', 1)
        result = model_selector.get_score(model)
        assert result >= 0.7, 'Sanity check failed.'

    def test_get_score_normal_calibration(self, get_model_and_validation_data):
        model, X_train, X_validate, y_train, y_validate, features, hyperparameters = get_model_and_validation_data
        model_selector = ModelSelector(X_train, X_validate, y_train, y_validate, features, hyperparameters, 'calibration', 1)
        result = model_selector.get_score(model)
        assert result <= 0.2, 'Sanity check failed.'

    def test_get_score_bin_constraint_not_met(self, get_bin_constraint_unfriendly_input, get_bin_constraint_unfriendly_expected):
        model, X_train, X_validate, y_train, y_validate, features, hyperparameters = get_bin_constraint_unfriendly_input
        model_selector = ModelSelector(X_train, X_validate, y_train, y_validate, features, hyperparameters, 'calibration', 1)
        result = model_selector.get_score(model)
        expected = get_bin_constraint_unfriendly_expected
        assert result == expected, f'ModelSelector.get_score returned {result} instead of {expected}.'

    def test_extract_best_model_accuracy(self, get_model_and_validation_data, extract_best_model_input, extract_best_model_expected_accuracy):
        model, X_train, X_validate, y_train, y_validate, features, hyperparameters = get_model_and_validation_data
        model_selector = ModelSelector(X_train, X_validate, y_train, y_validate, features, hyperparameters, 'accuracy', 3)
        model_scores = extract_best_model_input
        model_selector.model_scores = model_scores
        result = model_selector.extract_best_model()
        expected = extract_best_model_expected_accuracy
        assert result == expected, f'ModelSelector.extract_best_model returned {result} instead of {expected}.'


    def test_extract_best_model_calibration(self, get_model_and_validation_data, extract_best_model_input, extract_best_model_expected_calibration):
        model, X_train, X_validate, y_train, y_validate, features, hyperparameters = get_model_and_validation_data
        model_selector = ModelSelector(X_train, X_validate, y_train, y_validate, features, hyperparameters, 'calibration', 3)
        model_scores = extract_best_model_input
        model_selector.model_scores = model_scores
        result = model_selector.extract_best_model()
        expected = extract_best_model_expected_calibration
        assert result == expected, f'ModelSelector.extract_best_model returned {result} instead of {expected}.'

