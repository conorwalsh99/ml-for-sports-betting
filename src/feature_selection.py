import os
import toml
import logging
import json
import pandas as pd 
from typing import List, Dict, Union
 
from calibra.errors import classwise_ece
from calibra.utils import bin_probabilities, get_classwise_bin_weights
from utils import sort_labels_by_values
from sklearn.linear_model import LogisticRegression


config = toml.load(os.path.join('.', 'config.toml'))
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)


class FeatureSelector:
    """
    Class to perform feature selection.

    Remove highly correlated features (keeping only the feature most correlated with the target, out of a group of highly correlated features).
    Perform Sequential Forward Selection (SFS) to find the best performing feature subset under a given evaluation metric.

    Args:
        X_train (pd.DataFrame):
            Feature data from the training set. 
        X_validate (pd.DataFrame):
            Feature data from the validation set.
        y_train (List[int]):
            Target data from the training set.
        y_validate (List[int]):
            Target data from the validation set.
        metric (str):
            Metric used to evaluate model performance.
        correlation_method (str):
            Method used to calculate correlation of variables. Defaults to 'spearman'.
        threshold (float):
            Threshold used to define 'high correlation'. Defaults to 0.7.

    Attributes:
        X_train (pd.DataFrame):
            Feature data from the training set 
        X_validate (pd.DataFrame):
            Feature data from the validation set
        y_train (List[int]):
            Target data from the training set 
        y_validate (List[int]):
            Target data from the validation set
        metric (str):
            Metric used to evaluate model performance
        correlation_method (str):
            Method used to calculate correlation of variables. Defaults to 'spearman'.
        threshold (float):
            Threshold used to define 'high correlation'. Defaults to 0.7  
        TARGET (str): Target of classification problem.  
    """
    TARGET = 'home_victory'

    def __init__(self, 
                X_train: pd.DataFrame, 
                X_validate: pd.DataFrame, 
                y_train: List[int], 
                y_validate: List[int],
                metric: str,
                correlation_method: str = 'spearman',
                threshold: float = 0.7,
                ):
        self.X_train = X_train.copy(deep=True) 
        self.X_validate = X_validate.copy(deep=True)
        self.y_train = y_train.copy()
        self.y_validate = y_validate.copy()
        self.metric = metric
        self.correlation_method = correlation_method
        self.threshold = threshold
        self.add_target_back_to_data()

    def add_target_back_to_data(self):
        """
        Create new training data dataframe that contains both features and target.
        """
        self.data_train = self.X_train.copy(deep=True)
        self.data_train[FeatureSelector.TARGET] = self.y_train.copy()

    def get_correlation_matrix(self) -> pd.DataFrame:
        """
        Get correlation matrix for training data.

        Returns:
            pd.DataFrame:
                correlation matrix for training data.
        """
        return self.data_train.corr(method=self.correlation_method)

    @staticmethod
    def rank_features_by_target_corr(corr_matrix: pd.DataFrame, target: str) -> List[str]:
        """
        Rank the features in descending order of (absolute value of) correlation with the target.

        Args:
            corr_matrix (pd.DataFrame):
                Correlation matrix for the training data.
            target (str):
                Target variable.

        Returns:
            List[str]:
                List of features ranked in descending order of (absolute value of) correlation with the target.
        """
        target_corr_column = corr_matrix.abs()[target]
        features = [col for col in corr_matrix.columns if col != target]
        corr_scores = [target_corr_column[feature] for feature in features]
        ranked_features = sort_labels_by_values(corr_scores, features, descending=True, return_values=False)
        return ranked_features

    @staticmethod
    def identify_correlated_features_to_remove(corr_matrix: pd.DataFrame, ranked_features: List[str], target: str, threshold: float = 0.7) -> List[str]:
        """
        Identify groups of highly correlated features according to the method and threshold specified, and return all except those most correlated with the target.

        Loop through list of features ranked in descending order of correlation with the target, and for each, remove the other features it is highly correlated with.
        If a feature is already due to be dropped from the data, we need not remove the other features it is highly correlated with.

        Args:
            corr_matrix (pd.DataFrame):
                Correlation matrix for the training data.
            ranked_features (list):
                List of features in descending order of (absolute value of) correlation with the target.
            target (str):
                Target variable.
            threshold (float):
                Threshold used to define 'high correlation'. Defaults to 0.7. 
                
        Returns:
            List[str]:
                List of highly correlated features to be dropped from the dataset.
        """
        corr_matrix = corr_matrix.abs()
        features_to_drop = []

        for feature in ranked_features:
            if feature not in features_to_drop:
                columns_highly_correlated_with_feature = corr_matrix.index[corr_matrix[feature] >= threshold].to_list()
                other_features_highly_correlated_with_feature = list(
                    set(columns_highly_correlated_with_feature) - {feature, target}
                )
                features_to_drop.extend(other_features_highly_correlated_with_feature)
        
        return list(set(features_to_drop))

    def remove_highly_correlated_features(self, features_to_drop: List[str]):
        """
        Given a list of features highly correlated with others in the feature set, remove these from the data.

        Args:
            features_to_drop (List[str]):
                List of features to be dropped from data due to high correlation with other features (that will remain in the dataset).
        """
        self.data_train = self.data_train.drop(columns=features_to_drop)
        self.X_train = self.X_train.drop(columns=features_to_drop)
        self.X_validate = self.X_validate.drop(columns=features_to_drop)

    @staticmethod
    def get_score(model: LogisticRegression, X_validate: pd.DataFrame, y_validate: List[int], features: List[str], metric: str) -> float:
        """Calculate the score for the model as evaluated on the validation data, for the specified metric.

        Args:
            model (LogisticRegression): Model used for forward selection.
            X_validate (pd.DataFrame): Validation feature data.
            y_validate (list): Validation target data.
            features (list): Features under consideration for predictive modelling problem.
            metric (str): Model evaluation metric. One of 'accuracy' or 'calibration'.

        Raises:
            ValueError: Metric must be one of ["accuracy", "calibration"].

        Returns:
            float: Score of model under given metric.
        """
        if metric not in ['accuracy', 'calibration']:
            raise ValueError('Metric must be one of ["accuracy", "calibration"].')
        
        if metric == 'accuracy':
            return model.score(X_validate[features], y_validate)
        
        else:            
            num_bins = 20
            y_pred_proba = model.predict_proba(X_validate[features])
            binned_probabilities = bin_probabilities(y_pred_proba, y_validate, num_bins=num_bins)
            bin_weights = get_classwise_bin_weights(binned_probabilities)[0]
            non_empty_bins = [1 for weight in bin_weights if weight > 0]
            if sum(non_empty_bins) >= 0.8 * num_bins:
                return classwise_ece(y_pred_proba, y_validate, num_bins)
            else:
                return 1

    def extract_best_feature_subset(self, feature_subset_scores: List[Dict[str, Union[float, List[str]]]]) -> List[str]:
        """Extract the feature subset that achieved the best score under the given metric.
        
        Select the largest feature subset that achieved the best score, in the case of a tie (as feature space is already small).

        Args:
            feature_subset_scores (List[Dict[str, Union[float, List[str]]]]): 
                List of dictionaries containing feature subsets and corresponding score under the given metric.

        Returns:
            List[str]: Optimal feature subset list.
        """
        sign = -1 if self.metric == "accuracy" else 1

        scores = [
            score_dict['score'] * sign for score_dict in feature_subset_scores
        ]

        best_score = min(scores)

        for score_dict in feature_subset_scores:
            if score_dict['score'] * sign == best_score:
                best_subset = score_dict['feature_subset']
            
        return best_subset

    @staticmethod
    def run_forward_selection(
        X_train: pd.DataFrame, 
        y_train: List[int], 
        X_validate: pd.DataFrame, 
        y_validate: List[int], 
        metric: str = 'accuracy'
        ) -> List[Dict[str, Union[float, List[str]]]]:
        """
        Run sequential forward selection to determine the optimal feature set under the given metric. 

        NOTE: score <= best_score inequality comparison necessary for calibration
        (using score < best_score would result in never selecting the best feature on the first run because of the non-empty bin constraint).

        Args:
            X_train (pd.DataFrame):
                Feature data the model is fitted with.
            y_train (List[int]):
                Target data the model is fitted with.
            X_validate (pd.DataFrame):
                Feature data the model is scored on.
            y_validate (List[int]):
                Target data the model is scored on.
            metric: (str):
                Model evaluation metric.

        Raises:
            ValueError: Metric must be one of ["accuracy", "calibration"].
            
        Returns:
            List[Dict[str, Union[float, List[str]]]]:
                List of feature subsets and their corresponding score.                
        """
        if metric not in ['accuracy', 'calibration']:
            raise ValueError('Metric must be one of ["accuracy", "calibration"].')
                     
        full_feature_set = list(X_train.columns)
        current_feature_set = []
        feature_subset_scores = []        

        while len(current_feature_set) < len(full_feature_set):            
            features_to_test = [feature for feature in full_feature_set if feature not in current_feature_set]  
            best_score = 1 if metric == 'calibration' else 0

            for feature in features_to_test:                
                feature_set = current_feature_set + [feature]
                clf = LogisticRegression()
                clf.fit(X_train[feature_set], y_train)
                score = FeatureSelector.get_score(clf, X_validate, y_validate, feature_set, metric)              
                if (metric == 'accuracy' and score > best_score) or (metric == 'calibration' and score <= best_score): 
                    best_score = score
                    next_best_feature = feature

            current_feature_set.append(next_best_feature)
            feature_subset_scores.append(
                {                    
                    'feature_subset': current_feature_set.copy(), # must use copy to avoid referencing same list for every entry. 
                    'score': best_score
                }
            )

        return feature_subset_scores
    
    def record_feature_selection_results(
            self,
            full_feature_set: List[str], 
            filter_method_features_dropped: List[str],
            optimal_feature_set: List[str]
            ):
        """
        Record the results of the feature selection process.

        Args:
            full_feature_set (List[str]): List of features at beginning of feature selection process.
            filter_method_features_dropped (List[str]): List of features dropped after the filter method.
            optimal_feature_set (List[str]): List of optimal features found after filter and wrapper methods.
        """
        features_remaining_after_filter = [feature for feature in full_feature_set if feature not in filter_method_features_dropped]
        
        results_dict = {
            'full_feature_set': full_feature_set,
            'dropped_by_filter': filter_method_features_dropped,
            'remaining_after_filter': features_remaining_after_filter,
            'dropped_by_wrapper': [feature for feature in features_remaining_after_filter if feature not in optimal_feature_set],
            'remaining_after_wrapper': optimal_feature_set,
        }

        with open(os.path.join(".", "data", "output", "feature_selection", f"{self.metric}_driven_feature_selection.json"), "w") as file:
            json.dump(results_dict, file, indent=4)

    def run_feature_selection(self, save_results: bool = True) -> List[str]:
        """
        Run feature selection process.

        Args:
            save_results (bool): Flag to indicate whether or not to save results to file. Defaults to True.

        Returns:
            List[str]:
                Optimal feature set.
        """
        logging.info(f'Identifying highly correlated features by {self.correlation_method} method with threshold of {self.threshold}.')
        corr_matrix = self.get_correlation_matrix()
        ranked_features = FeatureSelector.rank_features_by_target_corr(corr_matrix, FeatureSelector.TARGET)
        features_to_drop = FeatureSelector.identify_correlated_features_to_remove(corr_matrix, ranked_features, FeatureSelector.TARGET, self.threshold)
        logging.info(f'Removing the following highly correlated features: {features_to_drop}')
        self.remove_highly_correlated_features(features_to_drop)
        feature_subset_scores = FeatureSelector.run_forward_selection(self.X_train, self.y_train, self.X_validate, self.y_validate, self.metric)
        optimal_feature_set = self.extract_best_feature_subset(feature_subset_scores)
        
        if save_results:
            self.record_feature_selection_results(
                full_feature_set = ranked_features,
                filter_method_features_dropped = features_to_drop,
                optimal_feature_set = optimal_feature_set
            )

        return optimal_feature_set
    