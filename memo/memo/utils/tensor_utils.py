import torch
import numpy as np
import os

def safe_save_tensor(tensor, filepath, create_dir=True):
    """
    Safely save a tensor to a file, handling BFloat16 and other data types.
    
    Args:
        tensor (torch.Tensor): The tensor to save
        filepath (str): Path where to save the tensor
        create_dir (bool, optional): Whether to create parent directories if they don't exist. Defaults to True.
    """
    # Create directory if it doesn't exist
    if create_dir:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Handle tensor conversion
    if isinstance(tensor, torch.Tensor):
        # Convert to CPU and ensure it's float32 to avoid BFloat16 errors
        numpy_array = tensor.detach().cpu().float().numpy()
    else:
        numpy_array = tensor
    
    # Save the tensor
    np.save(filepath, numpy_array) 