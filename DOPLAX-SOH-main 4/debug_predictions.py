import sys
sys.argv = [sys.argv[0]]
import numpy as np
from scipy.stats import pearsonr
from plot_all_datasets import load_args, load_model, get_test_file_list, predict_per_battery

for ds in ['XJTU', 'TJU', 'MIT', 'HUST']:
    print(f'\n=== {ds} ===')
    args = load_args(ds)
    model, data_obj = load_model(ds, args)
    test_files = get_test_file_list(ds)
    results = predict_per_battery(model, data_obj, test_files, args)
    for br in results:
        yt, yp = br['y_true'], br['y_pred']
        r, _ = pearsonr(yt, yp)
        mae = np.mean(np.abs(yt - yp))
        rmse = np.sqrt(np.mean((yt - yp) ** 2))
        bias = np.mean(yp - yt)
        max_err = np.max(np.abs(yt - yp))
        print(f"  {br['name']:30s} n={len(yt):4d}  MAE={mae:.5f}  RMSE={rmse:.5f}  Bias={bias:+.5f}  MaxErr={max_err:.5f}  R={r:.4f}  TrueRange=[{yt.min():.3f},{yt.max():.3f}]")
    
    # Overall
    all_yt = np.concatenate([b['y_true'] for b in results])
    all_yp = np.concatenate([b['y_pred'] for b in results])
    r, _ = pearsonr(all_yt, all_yp)
    print(f"  {'OVERALL':30s} n={len(all_yt):4d}  MAE={np.mean(np.abs(all_yt-all_yp)):.5f}  RMSE={np.sqrt(np.mean((all_yt-all_yp)**2)):.5f}  Bias={np.mean(all_yp-all_yt):+.5f}  MaxErr={np.max(np.abs(all_yt-all_yp)):.5f}  R={r:.4f}")
    del model
    import torch; torch.cuda.empty_cache()
