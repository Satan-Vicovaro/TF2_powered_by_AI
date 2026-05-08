from enum import Enum
from math import log
from typing import Any, Dict, List, Tuple
import logger
import json
import matplotlib.pyplot as plt


class Severity(Enum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class DataCollector:
    def __init__(self):
        self.iteration = 0
        self.data: List[Dict[Tuple[Severity, str], Any]] = [{}]

    def reset(self):
        self.iteration = 0
        self.data = [{}]

    def next_iteration(self):
        self.iteration += 1
        self.data.append({})

    def append(self, element_type: str, data: Any, severity: Severity = Severity.INFO):
        self.data[-1][(severity, element_type)] = data

    def print_interation_data(self):
        logger.logger.info(f"Iteration: {self.iteration}")
        for (severity, element_type), data_value in self.data[-1].items():
            message = f"{element_type} : {data_value}"
            log_method = getattr(logger.logger, severity.value)
            log_method(message)

    def save_data(self, file_name="log_data.json"):
        export_data = []
        for iteration_dict in self.data:
            cleaned_dict = {}
            for (severity, element_type), value in iteration_dict.items():
                # Still dropping severity for the save file
                cleaned_dict[element_type] = value
            export_data.append(cleaned_dict)

        export_dict = {"iteration": self.iteration, "data": export_data}

        with open("statistics_and_data/" + file_name, "w", encoding="utf-8") as f:
            json.dump(export_dict, f, indent=2)

        logger.logger.info(f"Data saved to {file_name}")

    def load_data(self, file_name="log_data_2.json"):
        try:
            with open("statistics_and_data/" + file_name, "r", encoding="utf-8") as f:
                imported_dict = json.load(f)

            self.iteration = imported_dict.get("iteration", 0)
            raw_loaded_data = imported_dict.get("data", [{}])

            self.data = []
            for iteration_dict in raw_loaded_data:
                restored_dict = {}
                for element_type, value in iteration_dict.items():
                    restored_dict[(Severity.INFO, element_type)] = value
                self.data.append(restored_dict)

            logger.logger.info(f"Data successfully loaded from {file_name}")

        except FileNotFoundError:
            logger.logger.error(f"Could not find file: {file_name}")
        except json.JSONDecodeError:
            logger.logger.error(f"File {file_name} is corrupted or not valid JSON.")

    def _get_rolling_stats(
        self, values: List[float], window: int
    ) -> Tuple[List[float], List[float]]:
        """Helper method to calculate moving average and rolling standard deviation."""
        rolling_avg = []
        rolling_std = []

        for i in range(len(values)):
            # Define the current window limits
            start_idx = max(0, i - window + 1)
            current_window = values[start_idx : i + 1]

            # Mean
            mean = sum(current_window) / len(current_window)
            rolling_avg.append(mean)

            # Standard Deviation (Dispersion)
            variance = sum((x - mean) ** 2 for x in current_window) / len(current_window)
            std_dev = variance**0.5
            rolling_std.append(std_dev)

        return rolling_avg, rolling_std

    def plot_data(self, window_size=1000):
        """
        Extracts Average_reward and Sum_reward from the stored data
        and plots them using matplotlib.
        """
        iterations = []
        avg_rewards = []
        sum_rewards = []

        # Iterate through the data list to extract values
        for i, iteration_dict in enumerate(self.data):
            avg_val = None
            sum_val = None

            # Unpack the tuple key and get the values
            for (severity, element_type), value in iteration_dict.items():
                try:
                    if element_type == "Average_reward":
                        avg_val = float(value)
                    elif element_type == "Sum_reward":
                        sum_val = float(value)
                except (ValueError, TypeError):
                    continue  # Skip if the data isn't a valid number

            # Only plot iterations that actually contain the reward data
            if avg_val is not None and sum_val is not None:
                iterations.append(i)
                avg_rewards.append(avg_val)
                sum_rewards.append(sum_val)

        if not iterations:
            print("No reward data found to plot.")
            return

        avg_smooth, avg_std = self._get_rolling_stats(avg_rewards, window_size)
        sum_smooth, sum_std = self._get_rolling_stats(sum_rewards, window_size)

        # Plotting
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # --- Average Reward Plot ---
        # 1. Plot the smoothed line
        ax1.plot(iterations, avg_smooth, color="blue", label=f"Moving Avg (window={window_size})")
        # 2. Add the dispersion shaded area (Mean + StdDev, Mean - StdDev)
        ax1.fill_between(
            iterations,
            [m - s for m, s in zip(avg_smooth, avg_std)],
            [m + s for m, s in zip(avg_smooth, avg_std)],
            color="blue",
            alpha=0.2,
            label="±1 Std Deviation",
        )

        ax1.set_title("Average Reward over Time")
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Average Reward")
        ax1.legend()
        ax1.grid(True, linestyle="--", alpha=0.7)

        # --- Sum Reward Plot ---
        ax2.plot(iterations, sum_smooth, color="green", label=f"Moving Avg (window={window_size})")
        ax2.fill_between(
            iterations,
            [m - s for m, s in zip(sum_smooth, sum_std)],
            [m + s for m, s in zip(sum_smooth, sum_std)],
            color="green",
            alpha=0.2,
            label="±1 Std Deviation",
        )

        ax2.set_title("Sum Reward over Time")
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("Sum Reward")
        ax2.legend()
        ax2.grid(True, linestyle="--", alpha=0.7)

        plt.tight_layout()
        plt.show()


shared_collector = DataCollector()
