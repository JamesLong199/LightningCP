import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import math
from PIL import Image, ImageOps
import matplotlib.cm as cm


def visualize_center_pixel_attention(file_path, save_dir='attention_visualizations'):
    """
    Visualize the attention map for the center pixel in a 64x64 feature map,
    showing how it attends to a 64x128 concatenated feature map.
    
    Args:
        file_path: Path to the saved attention dictionary
        save_dir: Directory to save visualizations
    """
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
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
    
    # Verify dimensions
    batch_size, num_heads, seq_len_q, _ = query.shape
    _, _, seq_len_k, _ = key.shape
    
    # Check if dimensions match our expectations
    if seq_len_q != 4096 or seq_len_k != 8192:
        print(f"Warning: Expected query length 4096 (64x64) and key length 8192 (64x128), "
              f"but got {seq_len_q} and {seq_len_k}")
    
    # Compute attention scores for the first batch item
    scale_factor = 1 / math.sqrt(head_dim)
    
    # Compute attention weights
    print("Computing attention weights...")
    attn_weights = torch.matmul(query[0], key[0].transpose(-2, -1)) * scale_factor
    
    # Apply softmax to get attention probabilities
    print("Applying softmax...")
    attn_probs = torch.nn.functional.softmax(attn_weights, dim=-1)
    
    # Find the center pixel index in the 64x64 grid
    center_y, center_x = 32, 32  # 0-indexed, so center is at 32,32
    center_idx = center_y * 64 + center_x
    
    # Extract attention map for the center pixel
    center_attn = attn_probs[:, center_idx, :].cpu().numpy()
    
    # Reshape the attention map to 2D grid (64x128)
    center_attn_2d = center_attn.reshape(num_heads, 64, 128)
    
    # Visualize the attention map for each head
    for h in range(num_heads):
        plt.figure(figsize=(16, 8))
        
        # Plot the attention map
        plt.imshow(center_attn_2d[h], cmap='viridis')
        plt.colorbar(label='Attention Weight')
        
        # Add a vertical line to separate A and B
        plt.axvline(x=63.5, color='red', linestyle='-', linewidth=2)
        
        # Add labels
        plt.title(f'Head {h}, Center Pixel Attention Map')
        plt.xlabel('Key Position (A | B)')
        plt.ylabel('Key Position (rows)')
        
        # Add text labels for A and B regions
        plt.text(32, -3, 'Feature Map A', horizontalalignment='center')
        plt.text(96, -3, 'Feature Map B', horizontalalignment='center')
        
        # Mark the center pixel position in the A region
        plt.plot(center_x, center_y, 'rx', markersize=10)
        
        # Save figure
        plt.savefig(f"{save_dir}/{attention_name}_center_pixel_head{h}.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    # Also create an average across all heads
    plt.figure(figsize=(16, 8))
    avg_all_heads = center_attn_2d.mean(0)
    
    plt.imshow(avg_all_heads, cmap='viridis')
    plt.colorbar(label='Attention Weight')
    plt.axvline(x=63.5, color='red', linestyle='-', linewidth=2)
    plt.title(f'Average Across All Heads, Center Pixel Attention Map')
    plt.xlabel('Key Position (A | B)')
    plt.ylabel('Key Position (rows)')
    plt.text(32, -3, 'Feature Map A', horizontalalignment='center')
    plt.text(96, -3, 'Feature Map B', horizontalalignment='center')
    plt.plot(center_x, center_y, 'rx', markersize=10)
    plt.savefig(f"{save_dir}/{attention_name}_center_pixel_avg_all_heads.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved center pixel attention visualizations to {save_dir}")
    
    # Also create a heatmap of attention to the center pixel from all query positions
    print("Creating heatmap of attention to center pixel...")
    
    # We need to compute the full attention matrix for this
    # This shows which query pixels attend to the center pixel in the key
    center_key_idx_A = center_idx  # Center pixel in A part of the key
    
    # Extract attention weights for all queries to the center pixel
    attn_to_center = attn_probs[:, :, center_key_idx_A].cpu().numpy()
    
    # Reshape to 2D grid (heads, 64, 64)
    attn_to_center_2d = attn_to_center.reshape(num_heads, 64, 64)
    
    # Visualize for each head
    for h in range(num_heads):
        plt.figure(figsize=(10, 10))
        
        # Plot the attention map
        plt.imshow(attn_to_center_2d[h], cmap='viridis')
        plt.colorbar(label='Attention Weight')
        
        # Mark the center pixel
        plt.plot(center_x, center_y, 'rx', markersize=10)
        
        # Add labels
        plt.title(f'Head {h}, Attention TO Center Pixel')
        plt.xlabel('Query Position (columns)')
        plt.ylabel('Query Position (rows)')
        
        # Save figure
        plt.savefig(f"{save_dir}/{attention_name}_to_center_pixel_head{h}.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    # Average across all heads
    plt.figure(figsize=(10, 10))
    avg_to_center = attn_to_center_2d.mean(0)
    
    plt.imshow(avg_to_center, cmap='viridis')
    plt.colorbar(label='Attention Weight')
    plt.plot(center_x, center_y, 'rx', markersize=10)
    plt.title(f'Average Across All Heads, Attention TO Center Pixel')
    plt.xlabel('Query Position (columns)')
    plt.ylabel('Query Position (rows)')
    plt.savefig(f"{save_dir}/{attention_name}_to_center_pixel_avg_all_heads.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved attention to center pixel visualizations to {save_dir}")

import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import math
from PIL import Image, ImageOps
import matplotlib.cm as cm

def overlay_attention_on_image(file_path, image_path, save_dir='attention_overlays'):
    """
    Visualize the attention map for the center pixel overlaid on the grayscale source image.
    Normalizes A and B attention maps jointly for direct comparison of attention strengths.
    
    Args:
        file_path: Path to the saved attention dictionary
        image_path: Path to the source image
        save_dir: Directory to save visualizations
    """
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
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
    
    # Get number of heads from the tensor shape
    num_heads = query.shape[1]
    
    # Compute attention scores for the first batch item
    scale_factor = 1 / math.sqrt(head_dim)
    
    # Compute attention weights
    print("Computing attention weights...")
    attn_weights = torch.matmul(query[0], key[0].transpose(-2, -1)) * scale_factor
    
    # Apply softmax to get attention probabilities
    print("Applying softmax...")
    attn_probs = torch.nn.functional.softmax(attn_weights, dim=-1)
    
    # Find the center pixel index in the 64x64 grid
    center_y, center_x = 32, 32  # 0-indexed, so center is at 32,32
    center_idx = center_y * 64 + center_x
    
    # Extract attention map for the center pixel
    center_attn = attn_probs[:, center_idx, :].cpu().numpy()
    
    # Use the correct separate A+B reshaping
    # Split the attention into A and B parts (first 4096 and last 4096)
    center_attn_A = center_attn[:, :4096].reshape(num_heads, 64, 64)
    center_attn_B = center_attn[:, 4096:].reshape(num_heads, 64, 64)
    
    # Load and resize the source image
    img = Image.open(image_path)
    
    # Convert to grayscale
    img_gray = ImageOps.grayscale(img)
    
    # Create separate overlays for A and B parts
    img_A = img_gray.resize((64, 64), Image.LANCZOS)
    img_B = img_gray.resize((64, 64), Image.LANCZOS)
    
    # Convert to numpy arrays
    img_A_np = np.array(img_A) / 255.0  # Normalize to [0, 1]
    img_B_np = np.array(img_B) / 255.0
    
    # Expand grayscale to 3 channels for blending
    img_A_np = np.stack([img_A_np, img_A_np, img_A_np], axis=2)
    img_B_np = np.stack([img_B_np, img_B_np, img_B_np], axis=2)
    
    # Visualize the attention map for each head
    for h in range(num_heads):
        # Get attention maps for A and B parts
        attn_A = center_attn_A[h]
        attn_B = center_attn_B[h]

        print(f'head {h}')
        print(f"Attention map A sum: {attn_A.sum()}")
        print(f"Attention map B sum: {attn_B.sum()}")
        
        # Create a figure with 2 subplots: A and B
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # Find the global max across both A and B for this head
        global_max = max(np.max(attn_A), np.max(attn_B))
        attn_A_norm = attn_A / global_max if global_max > 0 else attn_A
        attn_B_norm = attn_B / global_max if global_max > 0 else attn_B
        
        # Create a heatmap from the attention
        cmap = cm.get_cmap('hot')  # 'hot' colormap is more visible on grayscale
        attn_A_colored = cmap(attn_A_norm)[:, :, :3]  # Get RGB, drop alpha
        attn_B_colored = cmap(attn_B_norm)[:, :, :3]
        
        # Dim the grayscale image to make attention more visible
        img_A_dimmed = img_A_np * 0.4  # Reduce brightness more
        img_B_dimmed = img_B_np * 0.4
        
        # Blend image and attention with higher weight on attention
        overlay_A = img_A_dimmed * 0.5 + attn_A_colored * 0.5
        overlay_B = img_B_dimmed * 0.5 + attn_B_colored * 0.5
        
        # Plot on first subplot
        axes[0].imshow(overlay_A)
        axes[0].set_title(f'Head {h}, Self-Attention (A)')
        axes[0].axis('off')
        
        # Mark the center pixel
        axes[0].plot(center_x, center_y, 'gx', markersize=10)  # Green X for better visibility
        
        # Plot on second subplot
        axes[1].imshow(overlay_B)
        axes[1].set_title(f'Head {h}, Cross-Attention (B)')
        axes[1].axis('off')
        
        # Mark the center pixel in B as well
        axes[1].plot(center_x, center_y, 'gx', markersize=10)
        
        # Add text about joint normalization
        plt.figtext(0.5, 0.01, f"Attention maps normalized jointly (max={global_max:.6f})", 
                   ha="center", fontsize=10, bbox={"facecolor":"white", "alpha":0.5})
        
        # Save figure
        plt.tight_layout()
        plt.savefig(f"{save_dir}/{attention_name}_overlay_head{h}.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    # Also create an average across all heads
    avg_attn_A = center_attn_A.mean(0)
    avg_attn_B = center_attn_B.mean(0)
    
    # Create a figure with 2 subplots for the average
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Find the global max across both A and B for the average
    global_max_avg = max(np.max(avg_attn_A), np.max(avg_attn_B))
    avg_attn_A_norm = avg_attn_A / global_max_avg if global_max_avg > 0 else avg_attn_A
    avg_attn_B_norm = avg_attn_B / global_max_avg if global_max_avg > 0 else avg_attn_B
    
    # Create colored heatmaps
    avg_attn_A_colored = cmap(avg_attn_A_norm)[:, :, :3]
    avg_attn_B_colored = cmap(avg_attn_B_norm)[:, :, :3]
    
    # Blend with grayscale images
    overlay_A_avg = img_A_dimmed * 0.5 + avg_attn_A_colored * 0.5
    overlay_B_avg = img_B_dimmed * 0.5 + avg_attn_B_colored * 0.5
    
    axes[0].imshow(overlay_A_avg)
    axes[0].set_title('Average Across Heads, Self-Attention (A)')
    axes[0].axis('off')
    axes[0].plot(center_x, center_y, 'gx', markersize=10)
    
    axes[1].imshow(overlay_B_avg)
    axes[1].set_title('Average Across Heads, Cross-Attention (B)')
    axes[1].axis('off')
    axes[1].plot(center_x, center_y, 'gx', markersize=10)
    
    # Add text about joint normalization
    plt.figtext(0.5, 0.01, f"Attention maps normalized jointly (max={global_max_avg:.6f})", 
               ha="center", fontsize=10, bbox={"facecolor":"white", "alpha":0.5})
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{attention_name}_overlay_avg_all_heads.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved attention overlay visualizations to {save_dir}")


if __name__ == "__main__":
    # Example usage
    # file_path = "attention_data/cross_attn_downblock_0_layer_0_spatial_0_attn1.pt"
    file_path = "attention_data/cross_attn_upblock_3_layer_2_spatial_0_attn1.pt"
    image_path = "examples/reference_images/1.jpg"  # Path to your source image
    overlay_attention_on_image(file_path, image_path)
    # visualize_center_pixel_attention(file_path)