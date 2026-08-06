import os
import time
import psutil
from typing import Dict, Any

class KPITracker:
    """
    Tracks and reports Key Performance Indicators (KPIs) such as RAM usage,
    training time, and data volume for the battery prognostic models.
    """
    def __init__(self):
        self.start_time = 0.0
        self.end_time = 0.0
        self.process = psutil.Process(os.getpid())
        self.peak_ram_mb = 0.0
        self.num_batteries = 0
        self.total_cycles = 0

    def start(self, num_batteries: int, total_cycles: int):
        """
        Starts the timer and initializes data volume metrics.
        
        Args:
            num_batteries (int): Number of unique batteries used in this run.
            total_cycles (int): Total number of cycles processed in this run.
        """
        self.num_batteries = num_batteries
        self.total_cycles = total_cycles
        self.peak_ram_mb = self.process.memory_info().rss / (1024 * 1024)
        self.start_time = time.time()

    def update_ram(self):
        """
        Updates the peak RAM usage tracked so far. Should be called periodically
        during training loops to accurately capture peak memory.
        """
        current_ram = self.process.memory_info().rss / (1024 * 1024)
        if current_ram > self.peak_ram_mb:
            self.peak_ram_mb = current_ram

    def stop(self) -> Dict[str, Any]:
        """
        Stops the timer and returns the collected KPIs.
        
        Returns:
            Dict[str, Any]: A dictionary containing training time, peak RAM, and data volume.
        """
        self.end_time = time.time()
        self.update_ram()
        
        training_time_sec = self.end_time - self.start_time
        return {
            "training_time_seconds": training_time_sec,
            "training_time_minutes": training_time_sec / 60.0,
            "peak_ram_mb": self.peak_ram_mb,
            "num_batteries": self.num_batteries,
            "total_cycles": self.total_cycles
        }
