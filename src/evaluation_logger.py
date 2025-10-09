# src/evaluation_logger.py

import os
import json
import csv
import datetime

class EvaluationLogger:
    def __init__(self, log_path, csv_path):
        self.log_path = log_path
        self.csv_path = csv_path
        self.best_metric = 0.0
        self.log_data = []

        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, 'w') as f:
            f.write(f"Evaluation Log - {datetime.datetime.now()}\n")

        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "val_metric"])

    def log_epoch(self, epoch, val_metric):
        entry = {
            "epoch": epoch,
            "val_metric": val_metric
        }
        self.log_data.append(entry)

        with open(self.log_path, 'a') as f:
            f.write(json.dumps(entry) + "\n")

        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, val_metric])

    def is_best(self, val_metric):
        if val_metric > self.best_metric:
            self.best_metric = val_metric
            return True
        return False