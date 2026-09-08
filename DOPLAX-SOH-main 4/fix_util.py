import re

with open('Model/utils/util.py', 'r') as f:
    content = f.read()

replacement = '''
    best_epoch_match = re.search(r"The best model created at epoch.*", text)
    if best_epoch_match:
        best_epoch_number = best_epoch_match.group(0)
        best_epoch_number = int(re.findall(r"The best model created at epoch (\d+)", best_epoch_number)[0])
        best_epoch_line = re.search(r"\[Valid\] epoch:{}.*".format(best_epoch_number), text).group(0)
    else:
        valid_lines = re.findall(r"\[Valid\] epoch:\d+.*", text)
        if valid_lines:
            best_epoch_line = valid_lines[-1]
        else:
            return 0.0, 0.0
            
    mse_val = float(re.findall(r"MSE: (\d+\.\d+e[+-]\d+|\d+\.\d+)", best_epoch_line)[0])
'''

old_code = '''
    best_epoch_number = re.search(r"The best model created at epoch.*", text).group(0)
    best_epoch_number = int(re.findall(r"The best model created at epoch (\d+)", best_epoch_number)[0])
    best_epoch_line = re.search(r"\[Valid] epoch:{}.*".format(best_epoch_number), text).group(0)
    mse_val = float(re.findall(r"MSE: (\d+\.\d+e[+-]\d+|\d+\.\d+)", best_epoch_line)[0])
'''

content = content.replace(old_code.strip(), replacement.strip())

with open('Model/utils/util.py', 'w') as f:
    f.write(content)
