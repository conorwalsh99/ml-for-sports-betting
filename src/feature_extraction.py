import os
import toml
import json
import logging
import pandas as pd 
import numpy as np 
from scipy.stats import ks_2samp 
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple


config = toml.load(os.path.join('.', 'config.toml'))
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)


class FeatureExtractor:
    """
    Class to perform feature extraction. 
    
    Given preprocessed dataframe for each dataset with features constructed, perform following steps:
    - Remove features which show signs of covariate shift.
    - Scale data.

    Args:
        initial_training_data (pd.DataFrame):
            DataFrame composed of early seasons (that models are fitted with during feature selection and hyperparameter optimisation) 
        validation_data (pd.DataFrame):
            DataFrame composed of early seasons(s) following those in initial training data (that models are evaluated on during feature selection and hyperparameter optimisation)
        extended_training_data (pd.DataFrame):
            DataFrame composed of all seasons in initial training data and validation data (that models are fitted with before model selection)
        test_data (pd.DataFrame):
            DataFrame composed of season(s) following those in extended training data (that models are evaluated on during model selection)        
        final_training_data (pd.DataFrame):
            DataFrame composed of all seasons in extended training data and test data (that models are fitted with before the betting simulation)
        betting_simulation_data (pd.DataFrame):
            DataFrame composed of season(s) following those in final training data (that betting simulation is run on)                  
        threshold (float):
            p-value threshold (Kolmogorov-Smirnov test) to determine whether or not covariate shift has occurred. Defaults to 0.01 (strong evidence required).
        scale (bool):
            Flag that indicates whether or not to scale features. Defaults to True.  
        features (List[str]):
            List of features to be used for predictive modelling. Defaults to None.

    Attributes:
        initial_training_data (pd.DataFrame):
            DataFrame composed of early seasons (that models are fitted with during feature selection and hyperparameter optimisation) 
        validation_data (pd.DataFrame):
            DataFrame composed of early seasons(s) following those in initial training data (that models are evaluated on during feature selection and hyperparameter optimisation)
        extended_training_data (pd.DataFrame):
            DataFrame composed of all seasons in initial training data and validation data (that models are fitted with before model selection)
        test_data (pd.DataFrame):
            DataFrame composed of season(s) following those in extended training data (that models are evaluated on during model selection)        
        final_training_data (pd.DataFrame):
            DataFrame composed of all seasons in extended training data and test data (that models are fitted with before the betting simulation)
        betting_simulation_data (pd.DataFrame):
            DataFrame composed of season(s) following those in final training data (that betting simulation is run on)                  
        threshold (float):
            p-value threshold (Kolmogorov-Smirnov test) to detemrine whether or not covariate shift is present. Defaults to 0.01 (strong evidence required).
        scale (bool):
            Flag that indicates whether or not to scale features. Defaults to True.
        features (List[str]):
            List of features to be used for predictive modelling. If None, use features as specified in configuration file.
        self.shifting_features (List[str]): 
            List of features for which covariate shift has been detected.    
        self.ks_results (Dict[str, Dict[str, float]]):
            Dictionary containing the results (test statistic and corresponding p-value) of the Kolmogorov-Smirnov test for each feature.
        self.ks_significant_pvals (List[float]):
            List of p-values less than the specified significance threshold.
    """
    def __init__(self, 
                initial_training_data: pd.DataFrame,
                validation_data: pd.DataFrame,
                extended_training_data: pd.DataFrame,
                test_data: pd.DataFrame,
                final_training_data: pd.DataFrame,
                betting_simulation_data: pd.DataFrame,
                threshold: float = 0.01, 
                scale: bool = True,
                features: List[str] = None
        ):
        self.initial_training_data = initial_training_data
        self.validation_data = validation_data
        self.extended_training_data = extended_training_data
        self.test_data = test_data
        self.final_training_data = final_training_data
        self.betting_simulation_data = betting_simulation_data
        self.threshold = threshold
        self.scale = scale                
        self.features = features if features else config['modelling']['BOX_SCORE_STATISTICS'] + config['modelling']['ADDITIONAL_FEATURES']
        self.shifting_features = []
        
    def identify_shift(self):
        """
        Run kolmogorov-smirnov tests to identify features for which the initial training and validation data distributions differ.        
        """
        self.ks_results = {}
        self.ks_significant_pvals = []
        for feature in self.features:
            reference_data = self.initial_training_data[feature]
            test_data = self.validation_data[feature]
            ks_test_stat, ks_pval = ks_2samp(reference_data, test_data, alternative='two-sided')
            self.ks_results[feature] = {
                'statistic': ks_test_stat,
                'p_value': ks_pval
            } 
            if ks_pval < self.threshold:
                self.shifting_features.append(feature)
                self.ks_significant_pvals.append(ks_pval)
        
    def record_feature_extraction_results(self):
        """Record results of feature extraction including:
        
        - Features removed due to showing signs of covariate shift
        - Corresponding significant p-values on the Kolmogorov-Smirnov test
        - Features remaining after this step.
        """
        results_dict = {            
            'remaining': [feature for feature in self.features if feature not in self.shifting_features],
            'dropped': self.shifting_features,
            'significant_p_vals': self.ks_significant_pvals,
        }

        with open(os.path.join(".", "data", "output", "feature_selection", "drift_detection.json"), "w") as file:
            json.dump(results_dict, file, indent=4)

        with open(os.path.join(".", "data", "output", "feature_selection", "kolmogorov_smirnov_results.json"), "w") as file:
            json.dump(self.ks_results, file, indent=4)

    def remove_shifting_features(self):
        """
        Remove those features that have shown signs of shift in the kolmogorov-smirnov tests.
        """
        self.initial_training_data = self.initial_training_data.drop(columns=self.shifting_features)
        self.validation_data = self.validation_data.drop(columns=self.shifting_features)
        self.extended_training_data = self.extended_training_data.drop(columns=self.shifting_features)
        self.test_data = self.test_data.drop(columns=self.shifting_features)
        self.final_training_data = self.final_training_data.drop(columns=self.shifting_features)
        self.betting_simulation_data = self.betting_simulation_data.drop(columns=self.shifting_features)

    def standardise(self, reference_distribution: np.ndarray, apply_distribution: np.ndarray) -> np.ndarray:
        """
        Standardise an array according to the supplied reference distribution
        (so that the reference distribution standardised by itself would have a mean of 0 and standard deviation of 1).

        Args:
            reference_distribution (np.ndarray):
                Data used to standardise apply_distribution.
            apply_distribution (np.ndarray):
                Data to be standardised.
                                
        Returns:
            np.ndarray:
                Standardised apply dataframe.
        """
        scaler = StandardScaler()
        scaler.fit(reference_distribution)
        standardised_df_apply = pd.DataFrame(scaler.transform(apply_distribution), columns=apply_distribution.columns, index=apply_distribution.index)
        return standardised_df_apply

    def scale_feature_data(self):
        """
        Scale the feature data based on its own distribution for all training sets, and the distribution of all prior seasons for 
        all test/apply sets (validation, testing and betting simulation data). 

        Note: Scale data and store in new dataframe, then after all scaling is complete, set class attributes to the scaled data.
        Doing this too early could result in errors e.g. if we set self.initial_training_data = initial_training_data_scaled before we scale
        the validation_data, then we will end up scaling the validation data according to the distribution of the scaled initial training data,
        rather than its original distribution.
        """
        features = [feature for feature in self.features if feature not in self.shifting_features]

        initial_training_data_scaled = self.standardise(self.initial_training_data[features], self.initial_training_data[features])
        validation_data_scaled = self.standardise(self.initial_training_data[features], self.validation_data[features])
        extended_training_data_scaled = self.standardise(self.extended_training_data[features], self.extended_training_data[features])
        test_data_scaled = self.standardise(self.extended_training_data[features], self.test_data[features])
        final_training_data_scaled = self.standardise(self.final_training_data[features], self.final_training_data[features])
        betting_simulation_data_scaled = self.standardise(self.final_training_data[features], self.betting_simulation_data[features])

        self.initial_training_data = initial_training_data_scaled
        self.validation_data = validation_data_scaled
        self.extended_training_data = extended_training_data_scaled
        self.test_data = test_data_scaled
        self.final_training_data = final_training_data_scaled
        self.betting_simulation_data = betting_simulation_data_scaled 

    def extract_features(self, save_results: bool = True) -> Tuple[pd.DataFrame]:
        """
        Run all feature extraction steps and return modelling-ready datasets.

        Args:
            save_results (bool): Flag to indicate whether or not to save results to file. Defaults to True.        

        Returns:
            Tuple[pd.DataFrame]:
                Modelling-ready datasets.
        """         
        self.identify_shift()
        self.remove_shifting_features()
        logging.info(f'Drift detected (at {100 * self.threshold}% level of significance) in following features: {self.shifting_features}')
        
        if save_results:
            self.record_feature_extraction_results()
        
        if self.scale:
            self.scale_feature_data()

        return (
            self.initial_training_data,
            self.validation_data,
            self.extended_training_data,
            self.test_data,      
            self.final_training_data,
            self.betting_simulation_data
        )
