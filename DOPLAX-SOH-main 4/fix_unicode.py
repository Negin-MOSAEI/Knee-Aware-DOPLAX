import glob, re
import os

for f in glob.glob('Model/**/*.py', recursive=True):
    with open(f, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Use regex to find print(f" New best ... saved to {self.ckpt_path}")
    # and replace the arrow with ->
    new_content = re.sub(r'print\(f" New best .*? saved to \{self\.ckpt_path\}"\)', 
                         'print(f" New best -> saved to {self.ckpt_path}")', content)
                         
    if new_content != content:
        with open(f, 'w', encoding='utf-8') as file:
            file.write(new_content)
