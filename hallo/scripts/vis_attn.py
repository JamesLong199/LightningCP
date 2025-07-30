import torch
import math
import matplotlib.pyplot as plt
import numpy as np
import os

def compute_attention_from_saved_dict(file_path, save_visualization=True, save_dir='attention_visualizations'):
    """
    Compute attention scores from a saved attention dictionary.
    """
    # Load the saved dictionary
    print(f"Loading attention data from {file_path}")
    attn_dict = torch.load(file_path)
    
    # Extract components
    query = attn_dict['query']
    key = attn_dict['key']
    value = attn_dict['value']
    heads = attn_dict['heads']
    head_dim = attn_dict['head_dim']
    attention_name = attn_dict['attention_name']
    
    # Print shapes for debugging
    print(f"Query shape: {query.shape}")
    print(f"Key shape: {key.shape}")
    print(f"Value shape: {value.shape}")
    print(f"Number of heads: {heads}")
    print(f"Head dimension: {head_dim}")
    
    # Compute attention scores
    scale_factor = 1 / math.sqrt(head_dim)
    
    # Take just the first batch item for visualization
    batch_idx = 0
    query_batch = query[batch_idx:batch_idx+1]
    key_batch = key[batch_idx:batch_idx+1]
    value_batch = value[batch_idx:batch_idx+1]
    
    # Compute attention weights
    print("Computing attention weights...")
    attn_weights = torch.matmul(query_batch, key_batch.transpose(-2, -1)) * scale_factor
    
    # Apply softmax to get attention probabilities
    print("Applying softmax...")
    attn_probs = torch.nn.functional.softmax(attn_weights, dim=-1)
    
    # Compute attention output
    print("Computing attention output...")
    attention_output = torch.matmul(attn_probs, value_batch)
    
    # Save visualizations if requested
    if save_visualization:
        visualize_attention(attn_probs, attention_name, save_dir)
    
    return attn_probs, attention_output

def visualize_attention(attention_probs, attention_name, save_dir='attention_visualizations'):
    """
    Create and save visualizations of attention maps with correct aspect ratio.
    """
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Get dimensions
    batch_size, num_heads, seq_len_q, seq_len_k = attention_probs.shape
    print(f"Visualizing attention map of shape: {attention_probs.shape}")
    
    # Determine if this is likely spatial attention (square-ish dimensions)
    is_spatial = False
    spatial_size_q = int(math.sqrt(seq_len_q) + 0.5)  # Round to nearest int
    if abs(spatial_size_q**2 - seq_len_q) < 0.1 * seq_len_q:  # Within 10% of perfect square
        is_spatial = True
        spatial_size_k = int(math.sqrt(seq_len_k) + 0.5)
        print(f"Detected spatial attention: {spatial_size_q}x{spatial_size_q} -> {spatial_size_k}x{spatial_size_k}")
    
    # For very large attention maps, downsample for visualization while preserving aspect ratio
    if seq_len_q > 1024 or seq_len_k > 1024:
        print(f"Downsampling large attention map for visualization")
        
        # Calculate target size while preserving aspect ratio
        max_dim = 512
        aspect_ratio = seq_len_k / seq_len_q
        
        if seq_len_q > seq_len_k:
            target_q = max_dim
            target_k = int(max_dim * aspect_ratio)
        else:
            target_k = max_dim
            target_q = int(max_dim / aspect_ratio)
        
        # Calculate stride to achieve approximately the target size
        stride_q = max(1, seq_len_q // target_q)
        stride_k = max(1, seq_len_k // target_k)
        
        # Downsample by taking strided samples
        downsampled_attn = attention_probs[:, :, ::stride_q, ::stride_k]
        
        # Get actual downsampled dimensions
        actual_q = downsampled_attn.shape[2]
        actual_k = downsampled_attn.shape[3]
        
        print(f"Downsampled to shape: {downsampled_attn.shape}")
        print(f"Original aspect ratio: {seq_len_k/seq_len_q:.2f}, downsampled aspect ratio: {actual_k/actual_q:.2f}")
        
        # Update dimensions for visualization
        seq_len_q = actual_q
        seq_len_k = actual_k
        attention_probs = downsampled_attn
    
    # Create a figure for each head
    for h in range(num_heads):
        # Calculate figure size based on aspect ratio
        aspect_ratio = seq_len_k / seq_len_q
        fig_width = 12
        fig_height = fig_width / aspect_ratio
        
        plt.figure(figsize=(fig_width, fig_height))
        
        # Get attention map for this head (first batch item)
        attn_map = attention_probs[0, h].cpu().numpy()
        
        # Plot the attention map
        plt.imshow(attn_map, cmap='viridis', aspect='auto')
        plt.colorbar(label='Attention Weight')
        plt.title(f'Head {h}, Attention Map')
        plt.xlabel('Key Position')
        plt.ylabel('Query Position')
        
        # Save figure
        plt.savefig(f"{save_dir}/{attention_name}_head{h}.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    # Also create an average across all heads
    aspect_ratio = seq_len_k / seq_len_q
    fig_width = 12
    fig_height = fig_width / aspect_ratio
    
    plt.figure(figsize=(fig_width, fig_height))
    avg_all_heads = attention_probs[0].mean(0).cpu().numpy()
    
    plt.imshow(avg_all_heads, cmap='viridis', aspect='auto')
    plt.colorbar(label='Attention Weight')
    plt.title(f'Average Across All Heads')
    plt.xlabel('Key Position')
    plt.ylabel('Query Position')
    plt.savefig(f"{save_dir}/{attention_name}_avg_all_heads.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved attention visualizations to {save_dir}")

if __name__ == "__main__":
    # Example usage
    file_path = "attention_data/cross_attn_downblock_0_layer_0_spatial_0_attn1.pt"
    attn_probs, attention_output = compute_attention_from_saved_dict(file_path)
    
    # Print statistics about the attention
    print(f"Attention statistics:")
    print(f"  Min: {attn_probs.min().item()}")
    print(f"  Max: {attn_probs.max().item()}")
    print(f"  Mean: {attn_probs.mean().item()}")
    print(f"  Standard deviation: {attn_probs.std().item()}")