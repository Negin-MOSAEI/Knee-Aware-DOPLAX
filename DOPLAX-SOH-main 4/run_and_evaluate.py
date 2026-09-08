import os
import sys

print("Starting end-to-end training pipeline...")
import master_runner
master_runner.run_pipeline()
print("Training finished. Generating plots and summaries...")

import evaluate_results
evaluate_results.main()
