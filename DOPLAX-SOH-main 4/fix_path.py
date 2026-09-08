import glob
import os

for f in glob.glob('Model/**/*.py', recursive=True):
    with open(f, 'r', encoding='utf-8') as file:
        content = file.read()
    
    target = 'batch_num, exp_num = "one_batch", y_true_path.split(\'/\')[-2]'
    if target in content:
        content = content.replace(target, 'batch_num = "one_batch"\n            exp_num = os.path.normpath(y_true_path).split(os.sep)[-2]')
        with open(f, 'w', encoding='utf-8') as file:
            file.write(content)
