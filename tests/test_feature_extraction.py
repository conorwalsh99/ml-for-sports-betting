import pytest
import pandas as pd 
import numpy as np 


from src.feature_extraction import FeatureExtractor

feature_A_reference = np.random.normal(loc=0, scale=1, size=100)
feature_B_reference = np.random.uniform(low=0, high=1, size=100)
feature_A_test = np.random.normal(loc=0, scale=1, size=100)
feature_B_test = np.random.uniform(low=80, high=100, size=100)

@pytest.fixture
def identify_shift_input():
    reference_data = pd.DataFrame(
        {
            'feature_A': feature_A_reference,
            'feature_B': feature_B_reference        
        }
    )
    test_data = pd.DataFrame(
        {
            'feature_A': feature_A_test,
            'feature_B': feature_B_test
        }
    )
    return reference_data, test_data

@pytest.fixture
def identify_shift_expected():
    return ['feature_B']

@pytest.fixture
def remove_shifting_features_expected():
    reference_data = pd.DataFrame(
        {
            'feature_A': feature_A_reference
        }
    )
    test_data = pd.DataFrame(
        {
            'feature_A': feature_A_test
        }
    )
    return reference_data, test_data

@pytest.fixture
def scale_feature_data_input():
    reference_data = pd.DataFrame(
        {
            'feature_A': [0, 1, 2, 3],            
        }
    )
    test_data = pd.DataFrame(
        {
            'feature_A': [1, 1, 1, 4]
        }
    )
    return reference_data, test_data


@pytest.fixture
def scale_feature_data_expected():
    reference_data = pd.DataFrame(
        {
            'feature_A': [-1.341641, -0.447214, 0.447214, 1.341641],            
        }
    )
    test_data = pd.DataFrame(
        {
            'feature_A': [-0.447214, -0.447214, -0.447214, 2.23606797749979]
        }
    )
    return reference_data, test_data


class TestFeatureExtractor:

    def test_identify_shift(self, identify_shift_input, identify_shift_expected):
        reference_data, test_data = identify_shift_input
        feature_extractor = FeatureExtractor(initial_training_data=reference_data, 
                                            validation_data=test_data,
                                            extended_training_data=reference_data,
                                            test_data=test_data,
                                            final_training_data=reference_data,
                                            betting_simulation_data=test_data,
                                            features=['feature_A', 'feature_B'])
        feature_extractor.identify_shift()
        result = feature_extractor.shifting_features
        expected = identify_shift_expected
        assert result == expected, f'identify_shift returned {result} instead of {expected}.'


    def test_remove_shifting_features(self, identify_shift_input, remove_shifting_features_expected):
        reference_data, test_data = identify_shift_input
        feature_extractor = FeatureExtractor(initial_training_data=reference_data, 
                                            validation_data=test_data,
                                            extended_training_data=reference_data,
                                            test_data=test_data,
                                            final_training_data=reference_data,
                                            betting_simulation_data=test_data,
                                            features=['feature_A', 'feature_B'])
        feature_extractor.shifting_features = ['feature_B']
        feature_extractor.remove_shifting_features()
        result_initial_training_data = feature_extractor.initial_training_data
        result_validation_data = feature_extractor.validation_data
        expected_initial_training_data, expected_validation_data = remove_shifting_features_expected
        pd.testing.assert_frame_equal(result_initial_training_data, expected_initial_training_data), f'remove_shifting_features returned {result_initial_training_data} instead of {expected_initial_training_data}.'
        pd.testing.assert_frame_equal(result_validation_data, expected_validation_data), f'remove_shifting_features returned {result_validation_data} instead of {expected_validation_data}.'

    def test_scale_feature_data(self, scale_feature_data_input, scale_feature_data_expected):
        reference_data, test_data = scale_feature_data_input
        feature_extractor = FeatureExtractor(initial_training_data=reference_data, 
                                            validation_data=test_data,
                                            extended_training_data=reference_data,
                                            test_data=test_data,
                                            final_training_data=reference_data,
                                            betting_simulation_data=test_data,
                                            features=['feature_A'])
        feature_extractor.scale_feature_data()
        (
            result_initial_training_data,
            result_validation_data,
            result_extended_training_data,
            result_test_data,
            result_final_training_data,
            result_betting_simulation_data
        ) = (
            feature_extractor.initial_training_data,
            feature_extractor.validation_data,
            feature_extractor.extended_training_data,
            feature_extractor.test_data,
            feature_extractor.final_training_data,
            feature_extractor.betting_simulation_data
        )
        expected_reference, expected_test = scale_feature_data_expected
        expected_initial_training_data = expected_extended_training_data = expected_final_training_data = expected_reference
        expected_validation_data = expected_test_data = expected_betting_simulation_data = expected_test

        pd.testing.assert_frame_equal(result_initial_training_data, expected_initial_training_data), f'returned {result_initial_training_data} instead of {expected_initial_training_data}'
        pd.testing.assert_frame_equal(result_validation_data, expected_validation_data), f'returned {result_validation_data} instead of {expected_validation_data}'
        pd.testing.assert_frame_equal(result_extended_training_data, expected_extended_training_data), f'returned {result_extended_training_data} instead of {expected_extended_training_data}'
        pd.testing.assert_frame_equal(result_test_data, expected_test_data), f'returned {result_test_data} instead of {expected_test_data}'
        pd.testing.assert_frame_equal(result_final_training_data, expected_final_training_data), f'returned {result_final_training_data} instead of {expected_final_training_data}'
        pd.testing.assert_frame_equal(result_betting_simulation_data, expected_betting_simulation_data), f'returned {result_betting_simulation_data} instead of {expected_betting_simulation_data}'                        

    def test_extract_features(self, identify_shift_input):
        reference_data, test_data = identify_shift_input
        feature_extractor = FeatureExtractor(initial_training_data=reference_data, 
                                            validation_data=test_data,
                                            extended_training_data=reference_data,
                                            test_data=test_data,
                                            final_training_data=reference_data,
                                            betting_simulation_data=test_data,
                                            features = ['feature_A', 'feature_B'])
        result = feature_extractor.extract_features(save_results=False)
        result_initial_training_data_scaled = feature_extractor.initial_training_data
        expected_features = ['feature_A']
        expected_mean = 0.0
        expected_std = 1.0
        
        assert result_initial_training_data_scaled.columns == expected_features, f'Returned{result_initial_training_data_scaled.columns} instead of {expected_features}'
        assert np.isclose(result_initial_training_data_scaled['feature_A'].mean(), expected_mean, atol=0.01), f"Returned {result_initial_training_data_scaled['feature_A'].mean()} \
        instead of {expected_mean}"
        assert np.isclose(result_initial_training_data_scaled['feature_A'].std(), expected_std, atol=0.01), f"Returned {result_initial_training_data_scaled['feature_A'].std()} \
        instead of {expected_std}"
