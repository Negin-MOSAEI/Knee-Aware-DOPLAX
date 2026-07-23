import torch



pth_file = "PINN_XJTU_0.pth" # DOPLAX_model_XJTU_2C.pth 

checkpoint = torch.load(pth_file, map_location="cpu")
print("checkpoint:", checkpoint.keys())
print('*'*45)

if "PINN" in pth_file:
   subnets = checkpoint
elif "DOPlAX" in pth_file:
   subnets = list(checkpoint)[1:]
else:
   raise NotImplementedError
   
total_params = 0
for sub_net in subnets:
   sub_net_params = sum(p.numel() for p in checkpoint[sub_net].values())
   total_params += sub_net_params
print(f"Total Parameters for {pth_file}:", total_params)

