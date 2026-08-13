import os

files_to_update = [
    "Model/Backbones/layers/AutoCorrelation.py",
    "Model/DD_nets/MLP.py",
    "Model/DD_nets/PINNsFormer.py",
    "Model/PI_nets/DeepOPINN.py",
    "Model/PI_nets/LAX.py",
    "Model/utils/util.py",
    "pipeline/phase2_infer_kpd.py",
    "pipeline/phase6_final_soh.py",
    "pipeline/phase7_infer_kpis.py",
    "pipeline/plot_training_kpis.py",
    "run_all.py",
    "utils/plot_utils.py",
    "utils/util.py"
]

style_block = """
import matplotlib as mpl
mpl.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 18,
    "axes.titlesize": 18,
    "axes.labelsize": 18,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 18,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.linewidth": 0.8,
    "lines.linewidth": 2,
    "figure.autolayout": True,
})
"""

for file_path in files_to_update:
    full_path = os.path.join(os.getcwd(), file_path)
    if not os.path.exists(full_path):
        continue
    
    with open(full_path, "r", encoding="utf-8") as f:
        content = f.read()
        
    if '"font.family": "Times New Roman"' in content:
        continue # already has style
        
    # Find import matplotlib.pyplot as plt
    if "import matplotlib.pyplot as plt" in content:
        content = content.replace("import matplotlib.pyplot as plt", "import matplotlib.pyplot as plt\n" + style_block)
    elif "import matplotlib as mpl" in content:
        content = content.replace("import matplotlib as mpl", style_block)

    with open(full_path, "w", encoding="utf-8") as f:
        f.write(content)
print("Style applied to all files.")
