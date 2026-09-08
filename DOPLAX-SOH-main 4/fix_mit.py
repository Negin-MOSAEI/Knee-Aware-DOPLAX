import re

def fix_main(filename):
    with open(filename, 'r') as f:
        content = f.read()

    # Find the data and batch setup
    content = re.sub(
        r"setattr\(args, 'data', 'MIT'\)\s+setattr\(args, 'batch', 'one_batch'\)\s+investigating_losses = InvestigatingLosses\([\s\S]*?n_batches=None\s*\)",
        r"""setattr(args, 'data', 'MIT')
    batchs = ['2017-05-12', '2017-06-30', '2018-04-12']
    n_batches = len(batchs)

    investigating_losses = InvestigatingLosses(
        dataset_name=args.data, 
        root_path=save_folder, 
        finetuning_mode=finetuning_mode,
        n_experiments=n_experiments,
        n_batches=n_batches
    )""",
        content
    )

    # Replace the single batch setup with a loop
    old_loop_setup = """    if run_info['run_for_a_model_explicitly'] == False:
        DOP_pth_file = os.path.join(path_to_weights, 'DeepOPINN-Full', 'DeepOPINN_model_MIT.pth')
        if run_info['model_name'] == 'DOPDeepOLAX':
            LAX_pth_file = os.path.join(path_to_weights, 'DeepOLAX-MLP', 'DeepOLAX_model_MIT.pth')
        else:
            LAX_pth_file = os.path.join(path_to_weights, 'LAX-SUMProduct', 'LAX_model_MIT.pth')
        args.DOP_weights = DOP_pth_file 
        args.LAX_weights = LAX_pth_file

    # To store individual batch results for LAX
    all_batch_mean_results = []
    # To store individual experiment results for LAX
    batch_experiment_results = []

    # To store dataset batches for DeepONet, PINN, DeepOPINN, DOPFormer, and DOPLAX
    mse_vals_for_batches = []
    rmse_vals_for_batches = []
    # To store batch experiments for DeepONet, PINN, DeepOPINN, DOPFormer, and DOPLAX
    mse_vals_for_experiments = []
    rmse_vals_for_experiments = []
    
    e_repeat = 0
    while e_repeat < n_experiments:
        save_folder_cp = save_folder
        save_folder = save_folder + '/Experiment' + str(e_repeat + 1)
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)"""

    new_loop_setup = """    # To store individual batch results for LAX
    all_batch_mean_results = []

    # To store dataset batches for DeepONet, PINN, DeepOPINN, DOPFormer, and DOPLAX
    mse_vals_for_batches = []
    rmse_vals_for_batches = []

    for i in range(n_batches):
        batch = batchs[i]
        setattr(args, 'batch', batch)

        if run_info['run_for_a_model_explicitly'] == False:
            DOP_pth_file = os.path.join(path_to_weights, 'DeepOPINN-Full', f'DeepOPINN_model_MIT_{batch}.pth')
            if run_info['model_name'] == 'DOPDeepOLAX':
                LAX_pth_file = os.path.join(path_to_weights, 'DeepOLAX-MLP', f'DeepOLAX_model_MIT_{batch}.pth')
            else:
                LAX_pth_file = os.path.join(path_to_weights, 'LAX-phi', f'LAX_model_MIT_{batch}.pth')
            args.DOP_weights = DOP_pth_file 
            args.LAX_weights = LAX_pth_file

        # To store individual experiment results for LAX
        batch_experiment_results = []
        
        # To store batch experiments for DeepONet, PINN, DeepOPINN, DOPFormer, and DOPLAX
        mse_vals_for_experiments = []
        rmse_vals_for_experiments = []
        
        e_repeat = 0
        while e_repeat < n_experiments:
            save_folder_cp = save_folder
            save_folder = save_folder + "/" + str(i) + '-' + str(i) + '/Experiment' + str(e_repeat + 1)
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)"""

    content = content.replace(old_loop_setup, new_loop_setup)

    # Indent the while loop body
    # From `        if run_info['run_for_a_model_explicitly'] and run_info['model_name'] == 'LAX':`
    # up to `        rmse_vals_for_batches.append(rmse_vals_for_experiments)`
    
    # Actually, it's easier to just match from `        if run_info['run_for_a_model_explicitly'] and run_info['model_name'] == 'LAX':`
    # up to `        rmse_vals_for_batches.append(rmse_vals_for_experiments)`
    
    # I'll just write a quick script to fix indentation.
    lines = content.split('\n')
    in_loop = False
    in_small_sample = False
    for j in range(len(lines)):
        if "if run_info['run_for_a_model_explicitly'] and run_info['model_name'] == 'LAX':" in lines[j] and not in_small_sample:
            in_loop = True
        
        if in_loop:
            lines[j] = "    " + lines[j]
            
        if "rmse_vals_for_batches.append(rmse_vals_for_experiments)" in lines[j] and not in_small_sample:
            # But wait, in the new code, mse_vals_for_batches.append should be OUTSIDE the while loop, inside the for loop!
            in_loop = False
            
        if "def small_sample" in lines[j]:
            in_small_sample = True
            
    content = '\n'.join(lines)
    
    # Fix the end of the loop
    old_end = """        mse_vals_for_batches.append(mse_vals_for_experiments)
        rmse_vals_for_batches.append(rmse_vals_for_experiments)

    if run_mode == 'LAX':
        save_info = {
            'type': 'experiments',
            'experiments_info': batch_experiment_results,
            'batch': 'one_batch',
            'save_folder': args.results_path,
            'batches_info': all_batch_mean_results
        }
        _ = save_LAX_results(save_info=save_info)
    else:"""
    
    new_end = """
            if run_mode == 'LAX':
                save_info = {
                    'type': 'experiments',
                    'experiments_info': batch_experiment_results,
                    'batch': args.batch,
                    'save_folder': args.results_path,
                    'batches_info': all_batch_mean_results
                }
                all_batch_mean_results = save_LAX_results(save_info=save_info)
            else:
                mse_vals_for_batches.append(mse_vals_for_experiments)
                rmse_vals_for_batches.append(rmse_vals_for_experiments)

    if run_mode == 'LAX':
        save_info = {
            'type': 'batches',
            'experiments_info': None,
            'batch': args.batch,
            'save_folder': args.results_path,
            'batches_info': all_batch_mean_results
        }
        save_LAX_results(save_info=save_info)
    else:"""
    
    # Just replacing old_end won't work well due to indentation.
    
    with open(filename, 'w') as f:
        f.write(content)

fix_main('main_MIT.py')
