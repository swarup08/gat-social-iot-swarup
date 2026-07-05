"""
====================================================
Pruning Decision Logger
Project : GAT-based Secure Social IoT Framework
Author  : Swarup Sarkar
====================================================

Logs every pruning decision taken by the RL agent.
"""

import os
import csv


class PruningLogger:
    """
    Logger for RL pruning decisions.
    """

    def __init__(self, log_path="results/logs/pruning_log.csv"):

        self.log_path = log_path

        os.makedirs(os.path.dirname(log_path), exist_ok=True)

        # Create CSV with header only once
        if not os.path.exists(log_path):

            with open(log_path, "w", newline="") as file:

                writer = csv.writer(file)

                writer.writerow([
                    "Step",
                    "Source_Node",
                    "Destination_Node",
                    "Attention_Score",
                    "Source_Features",
                    "Destination_Features",
                    "Infection_Ratio",
                    "Connectivity_Ratio",
                    "Reward",
                ])
    def log(
        self,
        step,
        source,
        destination,
        attention_score,
        source_features,
        destination_features,
        infection_ratio,
        connectivity_ratio,
        reward,
    ):
        """
        Append one pruning decision.
        """

        with open(self.log_path, "a", newline="") as file:

            writer = csv.writer(file)

            writer.writerow([
                step,
                source,
                destination,
                attention_score,
                source_features,
                destination_features,
                infection_ratio,
                connectivity_ratio,
                reward,
            ])