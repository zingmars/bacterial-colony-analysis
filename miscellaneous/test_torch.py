import torch
torch.cuda.is_available() # True
torch.cuda.device_count() # 1 or more
torch.cuda.current_device() # 0 (not cpu)
