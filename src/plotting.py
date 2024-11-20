import os
import numpy as np 
import matplotlib as mpl
import matplotlib.pyplot as plt

from typing import Dict, List


class BettingSimulationPlotter:
    """Class to plot results of betting simulations for in-depth analysis.

    Args:
        calibration_driven_bankroll_tracker (List[float]):
            List containing the bettor's bankroll at each point in the season, for a calibration-driven betting simulation.
        accuracy_driven_bankroll_tracker (List[float]):
            List containing the bettor's bankroll at each point in the season, for an accuracy-driven betting simulation.
        calibration_driven_value_bets (Dict[str, List[float]]):
            Dictionary containing the predicted and implied probability for each value bet identified by the calibration-driven betting system.
        accuracy_driven_value_bets (Dict[str, List[float]]):
            Dictionary containing the predicted and implied probability for each value bet identified by the accuracy-driven betting system.
        rule (str):
            Betting rule used for the simulations.

    Attributes:
        calibration_driven_bankroll_tracker (List[float]):
            List containing the bettor's bankroll at each point in the season, for a calibration-driven betting simulation.
        accuracy_driven_bankroll_tracker (List[float]):
            List containing the bettor's bankroll at each point in the season, for n accuracy-driven betting simulation.
        calibration_driven_value_bets (Dict[str, List[float]]):
            Dictionary containing the predicted and implied probability for each value bet identified by the calibration-driven betting system.
        accuracy_driven_value_bets (Dict[str, List[float]]):
            Dictionary containing the predicted and implied probability for each value bet identified by the accuracy-driven betting system.
        rule (str):
            Betting rule used for the simulations.
    """
    def __init__(
            self, 
            calibration_driven_bankroll_tracker: List[float],
            accuracy_driven_bankroll_tracker: List[float],
            calibration_driven_value_bets: Dict[str, List[float]],
            accuracy_driven_value_bets: Dict[str, List[float]],
            rule: str,  
        ):
        self.calibration_driven_bankroll_tracker = calibration_driven_bankroll_tracker
        self.accuracy_driven_bankroll_tracker = accuracy_driven_bankroll_tracker
        self.calibration_driven_value_bets = calibration_driven_value_bets
        self.accuracy_driven_value_bets = accuracy_driven_value_bets
        self.rule = rule

    def plot_bankrolls(self):
        """Plot bankrolls of both betting systems over the given season to compare their performances.
        """
        mpl.rcParams["legend.frameon"] = False # Cleaner look without borders around legend
        fig, ax = plt.subplots(figsize=[10, 5])

        ax.plot(self.calibration_driven_bankroll_tracker, label='Calibration-Driven')
        ax.plot(self.accuracy_driven_bankroll_tracker, label='Accuracy-Driven')
        
        plt.ylim(
            0, 
            np.maximum(
                max(self.calibration_driven_bankroll_tracker) + 5000,
                max(self.accuracy_driven_bankroll_tracker) +5000
            )
        )

        plt.xlabel('Games', fontsize=15)
        plt.ylabel('Bankroll ($)', fontsize=15)
        plt.axhline(y=self.calibration_driven_bankroll_tracker[0], color='black', linestyle='--', label='Starting Bankroll')
        plt.legend(loc='upper left', prop={'size': 12})
        
        axins=ax.inset_axes([1.125, .35, .5, .5])
        axins.plot(
            list(range(1000, len(self.calibration_driven_bankroll_tracker))), 
            self.calibration_driven_bankroll_tracker[1000:]
            )
        axins.plot(
            list(range(1000, len(self.accuracy_driven_bankroll_tracker))),
            self.accuracy_driven_bankroll_tracker[1000:]
        )
        axins.axhline(y=self.accuracy_driven_bankroll_tracker[0], color='black', linestyle='--', label='Starting Bankroll')
        
        axins.set_title('End of Season Bankroll of Each System')
        axins.set_xlabel('Games', fontsize=12)
        axins.set_ylabel('Bankroll ($)', fontsize=12)
        axins.set_ylim(
            0,
            np.maximum(self.calibration_driven_bankroll_tracker[-1] + 1000, self.accuracy_driven_bankroll_tracker[-1] + 1000)
        )
        ax.indicate_inset_zoom(axins, edgecolor='grey', alpha=1)
        plt.savefig(os.path.join(".", "data", "output", f"{self.rule}_betting", "bankrolls_comparison.png"), dpi=300)
        plt.show()

    def plot_value_bet_distributions(self, metric: str):
        """Plot the distribution of value bets for the given betting system.

        Args:
            metric (str): Metric used for evaluation of system during predictive modelling. Can be either 'accuracy' or 'calibration'.                    
        """      
        value_bets_data = {
            'calibration': self.calibration_driven_value_bets,
            'accuracy': self.accuracy_driven_value_bets,
        }

        fig, ax = plt.subplots()
        ax.scatter(
            value_bets_data[metric]['predicted_probability'],
            value_bets_data[metric]['implied_probability']
        )
        ax.set_xlabel("Predicted Probability")
        ax.set_ylabel("Implied Probability")

        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]), 
            np.max([ax.get_xlim(), ax.get_ylim()]),  
        ]
        
        ax.plot(lims, lims, 'red', alpha=1, zorder=1)
        plt.savefig(os.path.join(".", "data", "output", "value_bets", f"{metric}_value_bets_distribution.png"), dpi=300)
        plt.show()

