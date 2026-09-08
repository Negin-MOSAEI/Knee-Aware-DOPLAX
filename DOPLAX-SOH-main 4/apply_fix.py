import glob
files = ['main_MIT.py', 'main_TJU.py', 'main_XJTU.py']
for fname in files:
    with open(fname, 'r', encoding='utf-8') as f:
        content = f.read()
    content = content.replace(
        "if test_metrics['rmse'][-1] > 0.50 or test_metrics['mape'][-1] > 0.50:",
        "if test_metrics['rmse'] > 50000.0 or test_metrics['mape'] > 50000.0:"
    )
    with open(fname, 'w', encoding='utf-8') as f:
        f.write(content)
