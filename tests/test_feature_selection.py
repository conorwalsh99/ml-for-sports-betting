import pytest
import pandas as pd 
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from src.feature_selection import FeatureSelector


@pytest.fixture
def add_target_back_to_data_input():
    X_train = pd.DataFrame(
        {
            'A': [1, 0, 0, 9],
            'B': [0, 8, 7, 4],
        }
    )

    X_validate = pd.DataFrame(
        {
            'A': [6, 7, 3, 9],
            'B': [0, 54, 2, 3],
        }
    )

    y_train = [0, 0, 1, 1]
    y_validate = [0, 1, 1, 0]
    
    return X_train, X_validate, y_train, y_validate

@pytest.fixture
def add_target_back_to_data_expected():
    return pd.DataFrame(
        {
            'A': [1, 0, 0, 9],
            'B': [0, 8, 7, 4],
            'home_victory': [0, 0, 1, 1]
        }
    ) 

@pytest.fixture
def get_correlation_matrix():
    return pd.DataFrame(
        {
    'A': {'A': 1, 'B': -1, 'C': 0, 'D': 0, 'E': 0, 'target': 0.4},
    'B': {'A': -1, 'B': 1, 'C': 0, 'D': 0, 'E': 0, 'target': -0.4},
    'C': {'A': 0, 'B': 0, 'C': 1, 'D': 0.9, 'E': 0.6, 'target': 0.5},
    'D': {'A': 0, 'B': 0, 'C': 0.9, 'D': 1, 'E': 0.8, 'target': 0.45},
    'E': {'A': 0, 'B': 0, 'C': 0.6, 'D': 0.8, 'E': 1, 'target':0.3},
    'target': {'A': 0.4, 'B': -0.4, 'C': 0.5, 'D': 0.45, 'E': 0.3, 'target': 1.0}

        }
    )

@pytest.fixture
def forward_selection_input():
    N_FEATURES = 5
    N_INFORMATIVE = 5
    X, y = make_classification(n_samples=100_000, n_features=N_FEATURES, 
                                            n_informative=N_INFORMATIVE,  n_redundant = 0,
                                            n_repeated=0, shuffle=False, random_state=42)
    X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(N_FEATURES)])
    X_train, X_validate = X.iloc[:7500], X.iloc[7500:]
    y_train, y_validate = y[:7500], y[7500:]

    return X_train, y_train, X_validate, y_validate, N_INFORMATIVE  

@pytest.fixture
def test_extract_best_feature_subset_input():
    return [
        {
            'feature_subset': ['A'],
            'score': 0.6
        },

        {
            'feature_subset': ['A', 'C'],
            'score': 0.7
        },

        {
            'feature_subset': ['A', 'C', 'B'],
            'score': 0.75
        },

        {
            'feature_subset': ['A', 'C', 'B', 'D'],
            'score': 0.6
        },                
    ]

@pytest.fixture
def test_extract_best_feature_subset_expected_accuracy():
    return ['A', 'C', 'B']
    
@pytest.fixture
def test_extract_best_feature_subset_expected_calibration():
    return ['A', 'C', 'B', 'D']    


@pytest.fixture
def ranked_features_expected():
    """
    Return the list of features ranked in descending order of correlation with target variable. Return all valid answers.
    """
    return (['C', 'D', 'A', 'B', 'E'], ['C', 'D', 'B', 'A', 'E'])

@pytest.fixture
def correlated_features_to_remove_expected():
    """
    Return the list of features to be removed due to correlation with other features. Return all valid answers.
    """
    return (['B', 'D'], ['A', 'D'])

@pytest.fixture
def get_model_and_validation_data():
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)    
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    X_train = pd.DataFrame(X_train, columns=feature_names)
    X_test = pd.DataFrame(X_test, columns=feature_names)
    good_model = LogisticRegression()
    good_model.fit(X_train, y_train)
    return good_model, X_test, y_test, feature_names

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
    return good_model, X_test, y_test, feature_names

@pytest.fixture
def get_bin_constraint_unfriendly_expected():
    return 1

class TestFeatureSelector:
    def test_add_target_back_to_data(self, add_target_back_to_data_input, add_target_back_to_data_expected):
        X_train, X_validate, y_train, y_validate = add_target_back_to_data_input
        feature_selector_obj = FeatureSelector(X_train, X_validate, y_train, y_validate, metric='accuracy')
        result = feature_selector_obj.data_train
        expected = add_target_back_to_data_expected
        pd.testing.assert_frame_equal(result, expected), f'Returned {result} instead of {expected}'

    def test_rank_features_by_target_corr(self, get_correlation_matrix, ranked_features_expected):
        result = FeatureSelector.rank_features_by_target_corr(get_correlation_matrix, 'target')
        expected = ranked_features_expected
        message = f'rank_features_by_target_corr returned {result} instead of a valid answer such as: {expected[0]} or {expected[1]}'        
        assert set(result) == set(expected[0]) or set(result) == set(expected[1]), message

    def test_identify_correlated_features_to_remove(self, get_correlation_matrix, ranked_features_expected, correlated_features_to_remove_expected):
        result = FeatureSelector.identify_correlated_features_to_remove(get_correlation_matrix, ranked_features_expected[0], 'target')
        expected = correlated_features_to_remove_expected
        assert set(result) == set(expected[0]) or set(result) == set(expected[1]), f'identify_correlated_features_to_remove returned {result} instead \
        of a valid answer such as: {expected[0]} or {expected[1]}'

    def test_extract_best_feature_subset_accuracy(self, test_extract_best_feature_subset_input, test_extract_best_feature_subset_expected_accuracy, forward_selection_input):
        X_train, y_train, X_validate, y_validate, N_INFORMATIVE = forward_selection_input
        feature_selector_obj = FeatureSelector(X_train, X_validate, y_train, y_validate, metric='accuracy')
        result = feature_selector_obj.extract_best_feature_subset(feature_subset_scores=test_extract_best_feature_subset_input)
        expected = test_extract_best_feature_subset_expected_accuracy
        assert result == expected, f'extract_best_feature_subset() returned {result} instead of {expected}.'

    def test_extract_best_feature_subset_calibration(self, test_extract_best_feature_subset_input, test_extract_best_feature_subset_expected_calibration, forward_selection_input):
        X_train, y_train, X_validate, y_validate, N_INFORMATIVE = forward_selection_input
        feature_selector_obj = FeatureSelector(X_train, X_validate, y_train, y_validate, metric='calibration')
        result = feature_selector_obj.extract_best_feature_subset(feature_subset_scores=test_extract_best_feature_subset_input)
        expected = test_extract_best_feature_subset_expected_calibration
        assert result == expected, f'extract_best_feature_subset() returned {result} instead of {expected}.'

    def test_get_score_bad_metric(self, get_model_and_validation_data):
        good_model, X_test, y_test, feature_names = get_model_and_validation_data
        with pytest.raises(ValueError):
            FeatureSelector.get_score(good_model, X_test, y_test, feature_names, metric='precision')

    def test_get_score_normal_accuracy(self, get_model_and_validation_data):
        good_model, X_test, y_test, feature_names = get_model_and_validation_data
        result = FeatureSelector.get_score(good_model, X_test, y_test, feature_names, metric='accuracy')
        assert result >= 0.7, 'Sanity check failed.'

    def test_get_score_normal_calibration(self, get_model_and_validation_data):
        good_model, X_test, y_test, feature_names = get_model_and_validation_data
        result = FeatureSelector.get_score(good_model, X_test, y_test, feature_names, metric='calibration')
        assert result <= 0.2, 'Sanity check failed.'

    def test_get_score_bin_constraint(self, get_bin_constraint_unfriendly_input, get_bin_constraint_unfriendly_expected):
        good_model, X_test, y_test, feature_names = get_bin_constraint_unfriendly_input
        result = FeatureSelector.get_score(good_model, X_test, y_test, feature_names, metric='calibration')
        expected = get_bin_constraint_unfriendly_expected
        assert result == expected, f'FeatureSelector.get_score returned {result} instead of {expected}.'
