# src/training_logger.py

import os
import json
import csv
import datetime

class TrainingLogger:
    def __init__(self, log_path, csv_path):
        self.log_path = log_path
        self.csv_path = csv_path
        self.best_loss = float("inf")
        self.log_data = []

        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, 'w') as f:
            f.write(f"Training Log - {datetime.datetime.now()}\n")

        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss"])

    def log_epoch(self, epoch, train_loss):
        entry = {
            "epoch": epoch,
            "train_loss": train_loss
        }
        self.log_data.append(entry)

        with open(self.log_path, 'a') as f:
            f.write(json.dumps(entry) + "\n")

        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss])

    def is_best(self, train_loss):
        if train_loss < self.best_loss:
            self.best_loss = train_loss
            return True
        return False