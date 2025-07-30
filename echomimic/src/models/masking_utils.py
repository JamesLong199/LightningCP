import torch
import math
import cv2
import numpy as np
import os
from einops import rearrange
     
def get_fg_tokens(mask_fg, hidden_states):
    """
    Get the foreground tokens from the mask_fg dictionary.
    """
    feature_size = int(math.sqrt(hidden_states.shape[1]))
    assert feature_size in mask_fg.keys()
    fg_indices = mask_fg[feature_size]["fg"]
    hidden_states = torch.index_select(hidden_states, 1, fg_indices)
    
    return hidden_states


def merge_bg_tokens(mask_fg, bg_features, feature_name, hidden_states):
    """
    Merge the background tokens from the mask_fg dictionary.
    """
    # print('merge_bg_tokens 37 feature_name:', feature_name, flush=True)
    # print('merge_bg_tokens 38 bg_features.keys():', bg_features.keys(), flush=True)
    assert feature_name in bg_features.keys()
    bg_feature = bg_features[feature_name]

    num_parallel = int(hidden_states.shape[0] // bg_feature.shape[0])
    if num_parallel > 1:
        bg_feature = bg_feature.repeat(num_parallel, 1, 1)

    b, n, d = bg_feature.shape

    feature_size = int(math.sqrt(n))
    # print('merge_bg_tokens 43 feature_size:', feature_size, flush=True)

    assert feature_size in mask_fg.keys()
    # print('merge_bg_tokens 46 feature_size in mask_fg.keys():', flush=True)
    fg_indices = mask_fg[feature_size]["fg"]
    bg_indices = mask_fg[feature_size]["bg"]

    hidden_states_fg = torch.zeros_like(bg_feature)
    # hidden_states_bg = torch.zeros_like(bg_feature)

    hidden_states_fg.scatter_(1, 
        fg_indices.unsqueeze(0).unsqueeze(-1).expand(b, -1, d), 
        hidden_states)
    
    hidden_states_fg[:, bg_indices, :] = bg_feature[:, bg_indices, :]
    hidden_states = hidden_states_fg
    
    return hidden_states


def get_masked_tokens(mask_fg, hidden_states, feature_name):
    """
    Get the background tokens from the mask_fg dictionary.
    """
    
    # feature_size = int(math.sqrt(hidden_states.shape[1]))
    if 'encoder' in feature_name:
        feature_size = int(math.sqrt(hidden_states.shape[1] // 2))
    else:
        feature_size = int(math.sqrt(hidden_states.shape[1]))
    assert feature_size in mask_fg.keys()
    indices = mask_fg[feature_size][feature_name]
    hidden_states = torch.index_select(hidden_states, 1, indices)

    return hidden_states


def merge_bgfg_features(mask_fg, bg_features, fg_features):
    """
    Merge the background tokens from the mask_fg dictionary.
    """

    assert bg_features.shape[0] == fg_features.shape[0] and bg_features.shape[2] == fg_features.shape[2]
    
    full_length = bg_features.shape[1] + fg_features.shape[1]
    feature_size = int(math.sqrt(full_length))
    assert feature_size in mask_fg.keys()

    batch_size, _, hidden_dim = bg_features.shape
    full_feature = torch.zeros(batch_size, full_length, hidden_dim, device=bg_features.device, dtype=bg_features.dtype)

    fg_indices = mask_fg[feature_size]["fg"]
    bg_indices = mask_fg[feature_size]["bg"]

    batch_indices = torch.arange(batch_size, device=bg_features.device).view(-1, 1)
    batch_indices = batch_indices.expand(-1, fg_indices.size(0))

    full_feature.scatter_(1, 
        fg_indices.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, hidden_dim), 
        fg_features)
    
    full_feature.scatter_(1, 
        bg_indices.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, hidden_dim), 
        bg_features)
    
    return full_feature
    


def batch_wise_index_select(tensor, indices):
    """
    Select different elements from each batch according to batch-specific indices.
    
    Args:
        tensor (torch.Tensor): Input tensor of shape [batch_size, seq_len, hidden_dim]
        indices (torch.Tensor): Indices tensor of shape [batch_size, num_indices]
        
    Returns:
        tuple: (selected_tensor, original_tensor) where:
               - selected_tensor is of shape [batch_size, num_indices, hidden_dim]
               - original_tensor is a copy of the input tensor
    """
    # Keep a copy of the original tensor
    # original_tensor = tensor.clone()
    
    batch_size = tensor.shape[0]
    batch_indices = torch.arange(batch_size, device=indices.device).unsqueeze(1).expand(-1, indices.shape[1])
    
    # Perform batch-wise indexing using advanced indexing
    selected_tensor = tensor[batch_indices, indices]
    
    # return selected_tensor, original_tensor
    return selected_tensor


def batch_wise_index_update(processed_tensor, original_tensor, indices):
    """
    Update the original tensor with processed values at the specified indices.
    
    Args:
        processed_tensor (torch.Tensor): Processed tensor of shape [batch_size, num_indices, hidden_dim]
        original_tensor (torch.Tensor): Original tensor to update of shape [batch_size, seq_len, hidden_dim]
        indices (torch.Tensor): The indices used for selection of shape [batch_size, num_indices]
        
    Returns:
        torch.Tensor: Updated tensor of the same shape as original_tensor
    """
    batch_size = processed_tensor.shape[0]
    hidden_dim = processed_tensor.shape[2]
    device = processed_tensor.device
    
    # Create a copy of the original tensor to modify
    updated_tensor = original_tensor.clone()
    
    # Create batch indices tensor
    batch_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, indices.shape[1])
    
    # Use indexing to update only the selected positions
    updated_tensor[batch_indices, indices] = processed_tensor
    
    return updated_tensor

def load_mask_fg(mask_fg_path, device):

    # Prepare foreground mask
    fg_mask = np.load(mask_fg_path).astype(np.uint8)
    fg_mask_64 = cv2.resize(fg_mask, (64, 64)).flatten()
    bool_fg_mask_64 = torch.from_numpy(fg_mask_64).bool()
    bool_bg_mask_64 = ~bool_fg_mask_64
    bool_fg_mask_64_encoder = torch.cat([bool_fg_mask_64, bool_fg_mask_64], dim=0)
    bool_bg_mask_64_encoder = torch.cat([bool_bg_mask_64, bool_bg_mask_64], dim=0)
    
    fg_indices_64 = torch.nonzero(bool_fg_mask_64).squeeze()
    bg_indices_64 = torch.nonzero(bool_bg_mask_64).squeeze()
    fg_encoder_indices_64 = torch.nonzero(bool_fg_mask_64_encoder).squeeze()
    bg_encoder_indices_64 = torch.nonzero(bool_bg_mask_64_encoder).squeeze()
    
    
    source_fg_mask = {
        64: {
            "fg": fg_indices_64.to(device=device),
            "bg": bg_indices_64.to(device=device),
            "fg_encoder": fg_encoder_indices_64.to(device=device),
            "bg_encoder": bg_encoder_indices_64.to(device=device),
        }
    }

    return source_fg_mask

def save_attention_data(save_dir, save_type, hidden_states, block_name):
    """
    Save the attention data for the given block name.
    """
    timestep = block_name.split('_')[0]
    os.makedirs(f'{save_dir}/{timestep}', exist_ok=True)
    
    # Convert PyTorch tensor to NumPy array and save with np.save
    if isinstance(hidden_states, torch.Tensor):
        numpy_array = hidden_states.detach().cpu().float().numpy()
    else:
        numpy_array = hidden_states
    np.save(f'{save_dir}/{timestep}/{block_name}_{save_type}.npy', numpy_array)

def reshape_attn_data(hidden_states):
    """
    Reshape the attention data to the original shape.
    """
    hidden_states = rearrange(hidden_states, "b c f h w -> (b f) c h w")
    batch, _, height, weight = hidden_states.shape
    inner_dim = hidden_states.shape[1]
    hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(
        batch, height * weight, inner_dim
    )

    return hidden_states

def save_attention_weights(save_dir, attn_dict, block_name):
    """
    Save the attention weights for the given block name.
    Uses NumPy arrays and saves as .npy files.
    """
    import os
    import numpy as np
    
    timestep = block_name.split('_')[0]
    os.makedirs(f'{save_dir}/{timestep}', exist_ok=True)

    # Create filename based on block name
    filename = f"{save_dir}/{timestep}/{block_name}.npy"
    
    # Save the dictionary to disk as numpy array
    np.save(filename, attn_dict)
    
    return filename


    