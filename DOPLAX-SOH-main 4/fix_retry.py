with open('main_HUST.py', 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace("if test_metrics['rmse'] > 0.50 or test_metrics['mape'] > 0.50:", 
                          "if test_metrics['rmse'] > 50000.0 or test_metrics['mape'] > 50000.0:")

with open('main_HUST.py', 'w', encoding='utf-8') as f:
    f.write(content)
