"""
CSV Logger for training metrics.
"""

import csv
import os
from datetime import datetime


class CSVLogger:
    def __init__(self, base_dir: str = "research/results"):
        os.makedirs(base_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filename = os.path.join(base_dir, f"training_log_{timestamp}.csv")
        self.headers = ['epoch', 'step', 'train_loss', 'val_loss', 'perplexity']
        
        # Create file with headers
        with open(self.filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.headers)
    
    def log(self, epoch: int, step: int, train_loss: float, val_loss: float, perplexity: float):
        with open(self.filename, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, step, train_loss, val_loss, perplexity])