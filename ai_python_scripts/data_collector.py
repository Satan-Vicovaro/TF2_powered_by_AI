from enum import Enum
from math import log
from typing import Any, Dict, List, Tuple
import logger
import json


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


shared_collector = DataCollector()
