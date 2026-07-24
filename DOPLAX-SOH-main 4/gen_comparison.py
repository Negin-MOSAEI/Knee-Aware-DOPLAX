"""Generate the overall comparison table from saved metrics."""
import pandas as pd
from pathlib import Path

root = Path("outputs")
datasets = [
    ("XJTU", "3C", 16),
    ("TJU", "NCM_NCA", 16),
    ("MIT", "one_batch", 16),
    ("HUST", "one_batch", 16),
]

rows = []
for ds, batch, gdim in datasets:
    df = pd.read_csv(root / ds / "metrics.csv")
    d = dict(zip(df["Metric"].str.lower(), df["Value"]))
    d["final_loss"] = d.get("final_val_loss", 0)
    rows.append({"Dataset": ds, "Batch": batch, "g_dim": gdim, **d})

out = pd.DataFrame(rows)
out.to_csv(root / "overall_dataset_comparison.csv", index=False)

print()
print("=" * 95)
print("  OVERALL DATASET COMPARISON -- LAX Model")
print("=" * 95)
hdr = "{:<10s} {:<12s} {:>10s} {:>10s} {:>10s} {:>10s} {:>10s} {:>12s}".format(
    "Dataset", "Batch", "MSE", "RMSE", "MAPE", "MAE", "Pearson R", "Final Loss"
)
print(hdr)
print("-" * 95)
for r in rows:
    print("{:<10s} {:<12s} {:>10.6f} {:>10.6f} {:>10.6f} {:>10.6f} {:>10.6f} {:>12.6f}".format(
        r["Dataset"], r["Batch"], r["mse"], r["rmse"], r["mape"], r["mae"], r["pearson_r"], r["final_loss"]
    ))
print("=" * 95)
print()
print("Saved to outputs/overall_dataset_comparison.csv")
