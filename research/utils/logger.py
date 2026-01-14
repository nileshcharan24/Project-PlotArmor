"""
CSV Logger for training metrics.
"""

import csv
import os


class CSVLogger:
    def __init__(self, filename: str):
        self.filename = filename
        self.headers = ['epoch', 'step', 'train_loss', 'val_loss', 'perplexity']
        
        # Create directory if needed
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Create file with headers if it doesn't exist
        if not os.path.exists(filename):
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(self.headers)
    
    def log(self, epoch: int, step: int, train_loss: float, val_loss: float, perplexity: float):
        with open(self.filename, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, step, train_loss, val_loss, perplexity])