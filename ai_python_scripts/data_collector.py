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

    def load_data(self, file_name="log_data.json"):
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
        Extracts Average_reward, Sum_reward and Hit_counter from the stored data
        and plots them using matplotlib.
        """
        iterations = []
        avg_rewards = []
        sum_rewards = []
        sigma_changes = []
        hit_iterations = []
        hit_counters = []

        # Iterate through the data list to extract values
        for i, iteration_dict in enumerate(self.data):
            avg_val = None
            sum_val = None
            hit_val = None

            # Unpack the tuple key and get the values
            for (severity, element_type), value in iteration_dict.items():
                try:
                    if element_type == "Average_reward":
                        avg_val = float(value)
                    elif element_type == "Sum_reward":
                        sum_val = float(value)
                    elif element_type == "Hit_counter":
                        hit_val = float(value)
                    elif element_type == "Sigma_change":
                        sigma_changes.append((i, float(value)))
                except (ValueError, TypeError):
                    continue  # Skip if the data isn't a valid number

            # Only plot iterations that actually contain the reward data
            if avg_val is not None and sum_val is not None:
                iterations.append(i)
                avg_rewards.append(avg_val)
                sum_rewards.append(sum_val)
            
            if hit_val is not None:
                hit_iterations.append(i)
                hit_counters.append(hit_val)

        if not iterations:
            print("No reward data found to plot.")
            return

        avg_smooth, avg_std = self._get_rolling_stats(avg_rewards, window_size)
        sum_smooth, sum_std = self._get_rolling_stats(sum_rewards, window_size)

        # Plotting
        if hit_iterations:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21, 6))
        else:
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
        
        for it, val in sigma_changes:
            ax1.axvline(x=it, color='red', linestyle=':', alpha=0.6)
            
        handles1, labels1 = ax1.get_legend_handles_labels()
        if sigma_changes:
            from matplotlib.lines import Line2D
            handles1.append(Line2D([0], [0], color='red', linestyle=':', alpha=0.6))
            labels1.append('Sigma Changed')
            
        ax1.legend(handles1, labels1)
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
        
        for it, val in sigma_changes:
            ax2.axvline(x=it, color='red', linestyle=':', alpha=0.6)
            
        handles2, labels2 = ax2.get_legend_handles_labels()
        if sigma_changes:
            from matplotlib.lines import Line2D
            handles2.append(Line2D([0], [0], color='red', linestyle=':', alpha=0.6))
            labels2.append('Sigma Changed')
            
        ax2.legend(handles2, labels2)
        ax2.grid(True, linestyle="--", alpha=0.7)

        # --- Hit Counter Plot ---
        if hit_iterations:
            hit_smooth, hit_std = self._get_rolling_stats(hit_counters, window_size)
            ax3.plot(hit_iterations, hit_smooth, color="orange", label=f"Moving Avg (window={window_size})")
            ax3.fill_between(
                hit_iterations,
                [m - s for m, s in zip(hit_smooth, hit_std)],
                [m + s for m, s in zip(hit_smooth, hit_std)],
                color="orange",
                alpha=0.2,
                label="±1 Std Deviation",
            )

            ax3.set_title("Hit Counter over Time")
            ax3.set_xlabel("Iteration")
            ax3.set_ylabel("Hit Counter")
            
            for it, val in sigma_changes:
                ax3.axvline(x=it, color='red', linestyle=':', alpha=0.6)
                
            handles3, labels3 = ax3.get_legend_handles_labels()
            if sigma_changes:
                from matplotlib.lines import Line2D
                handles3.append(Line2D([0], [0], color='red', linestyle=':', alpha=0.6))
                labels3.append('Sigma Changed')
                
            ax3.legend(handles3, labels3)
            ax3.grid(True, linestyle="--", alpha=0.7)

        plt.tight_layout()
        plt.show()


shared_collector = DataCollector()
