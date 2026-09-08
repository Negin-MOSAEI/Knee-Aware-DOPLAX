import glob, re
import os

with open('Investigating_Losses.py', 'r', encoding='utf-8') as f:
    content = f.read()
    
content = content.replace("import glob\n                      actual_lable = np.load(glob.glob(os.path.join(self.root_path, '**', self.batch, experiment, 'true_label.npy'), recursive=True)[0])", 
                          "actual_lable = np.load(glob.glob(os.path.join(self.root_path, '**', self.batch, experiment, 'true_label.npy'), recursive=True)[0])")
                          
with open('Investigating_Losses.py', 'w', encoding='utf-8') as f:
    f.write(content)
