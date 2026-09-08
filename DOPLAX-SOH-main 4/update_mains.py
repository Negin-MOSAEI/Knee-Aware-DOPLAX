import os

files = ['main_HUST.py', 'main_MIT.py', 'main_TJU.py']
for f in files:
    with open(f, 'r') as file:
        text = file.read()
    
    text = text.replace('["PINN", "DeepONet", "PINNsFormer", "DONG", "DeepOPINN", "DOPFormer", "FormerPINN", "LAX"]', '["PINN", "DeepONet", "PINNsFormer", "DONG", "DeepOPINN", "DOPFormer", "FormerPINN", "LAX", "TCN"]')
    text = text.replace(", 'LAX'\")", ", 'LAX', or 'TCN'\")")
    
    # Save folder
    lines = text.split('\n')
    for i, line in enumerate(lines):
        if "'FormerPINN':" in line and "results of reviewer" in lines[i+1]:
            if "TCN" not in lines[i+2]:
                dataset = f.split('_')[1].split('.')[0]
                lines.insert(i+2, f"            elif run_info['model_name'] == 'TCN':")
                lines.insert(i+3, f"                save_folder = f'results of reviewer/TCN_{{current_time}}_for_{dataset}/{dataset} results'")
            break
            
    text = '\n'.join(lines)
    
    # args
    lines = text.split('\n')
    for i, line in enumerate(lines):
        if "if run_info['model_name'] == 'PINN':" in line:
            if "TCN" not in lines[i+3]:
                dataset = f.split('_')[1].split('.')[0]
                lines.insert(i+3, f"        elif run_info['model_name'] == 'TCN':")
                lines.insert(i+4, f"            args = get_DeepONet_args()")
                lines.insert(i+5, f"            args.batch_size = args.batch_size_{dataset}")
            break
            
    text = '\n'.join(lines)
    
    # import TCN and train
    lines = text.split('\n')
    for i, line in enumerate(lines):
        if "elif run_info['model_name'] == 'DOPFormer':" in line:
            for j in range(i, len(lines)):
                if "else:" in lines[j] and "LAX" in lines[j]:
                    if "TCN" not in lines[j-1]:
                        lines.insert(j, "                elif run_info['model_name'] == 'TCN':")
                        lines.insert(j+1, "                    from Model.DD_nets import TCN")
                        lines.insert(j+2, "                    setattr(args, 'run_mode', 'TCN')")
                        lines.insert(j+3, "                    tcn = TCN.Model(args)")
                        lines.insert(j+4, "                    train_result, metric_values = tcn.Train(trainloader=dataloader['train_3'], validloader=dataloader['test_3'], testloader=dataloader['test_3'])")
                        lines.insert(j+5, "                    if train_result == 'invalid':")
                        lines.insert(j+6, "                        rmse_value, mape_value = metric_values[0], metric_values[1]")
                        lines.insert(j+7, "                        info = f\"Experiment {e_repeat+1} invalid (RMSE={rmse_value:.4f}, MAPE={mape_value:.4f}). Retrying...\"")
                        lines.insert(j+8, "                        file_path = os.path.join(os.path.dirname(args.save_folder), 'faild_experiments.txt')")
                        lines.insert(j+9, "                        write_to_file(file_path, info)")
                        lines.insert(j+10, "                        invalid_experiment = True")
                        lines.insert(j+11, "")
                    break
            break
            
    text = '\n'.join(lines)
    
    with open(f, 'w') as file:
        file.write(text)

print('Updated all main files!')
