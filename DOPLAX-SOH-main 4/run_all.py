import os
os.environ['DDE_BACKEND'] = 'pytorch'
import os
import sys

from pipeline import (
    phase0_init,
    phase1_train_tst,
    phase2_infer_kpd,
    phase3_train_deepopinn,
    phase4_train_lax,
    phase5_train_fusion,
    phase6_final_soh,
    phase7_infer_kpis
)

def run_all_phases():
    project_root = os.path.abspath(os.path.dirname(__file__))
    
    print("="*50)
    print("BATTERY PROGNOSTICS 8-PHASE PIPELINE")
    print("="*50)
    
    # Phase 0
    phase0_init.run_phase0(project_root)
    print("-"*50)
    
    # Phase 1
    phase1_train_tst.run_phase1(project_root, num_epochs=1)
    print("-"*50)
    
    # Phase 2
    phase2_infer_kpd.run_phase2(project_root)
    print("-"*50)
    
    # Phase 3
    phase3_train_deepopinn.run_phase3(project_root, num_epochs=1)
    print("-"*50)
    
    # Phase 4
    phase4_train_lax.run_phase4(project_root, num_epochs=1)
    
    # Phase 5
    phase5_train_fusion.run_phase5(project_root, num_epochs=1)
    print("-"*50)
    
    # Phase 6
    phase6_final_soh.run_phase6(project_root)
    print("-"*50)
    
    # Phase 7
    phase7_infer_kpis.run_phase7(project_root)
    print("="*50)
    print("PIPELINE COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    run_all_phases()

