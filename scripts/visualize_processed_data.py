import os
import random
import torch
import numpy as np  # Make sure numpy is imported
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg') # Use non-interactive backend before importing pyplot
from mpl_toolkits.axes_grid1 import make_axes_locatable # For better colorbar control

def visualize_random_samples(data_dir, output_path, num_samples=10, dpi=300):
    """
    Randomly load data samples from the specified directory and visualize initial topography,
    final topography, uplift field, and parameters.

    Args:
        data_dir (str): Directory path containing .pt data files.
        output_path (str): File path to save the output image (e.g., 'visualization.png').
        num_samples (int): Number of samples to randomly select and visualize.
        dpi (int): Resolution of the output image (dots per inch).
    """
    # Check if data directory exists
    if not os.path.isdir(data_dir):
        print(f"Error: Data directory '{data_dir}' does not exist.")
        return

    # Recursively get all .pt files
    all_file_paths = []
    try:
        print(f"Recursively searching for .pt files in '{data_dir}' and its subdirectories...")
        for root, _, files in os.walk(data_dir):
            for file in files:
                if file.endswith('.pt'):
                    all_file_paths.append(os.path.join(root, file))

        if not all_file_paths:
            print(f"Error: No .pt files found in '{data_dir}' and its subdirectories.")
            return

        print(f"Found {len(all_file_paths)} .pt files.")

        if len(all_file_paths) < num_samples:
            print(f"Warning: Number of files found ({len(all_file_paths)}) is less than the requested number of samples ({num_samples}). Using all found files.")
            num_samples = len(all_file_paths)
            if num_samples == 0:
                return # Exit if adjusted sample count is 0

        # Randomly select file paths from the complete path list
        selected_file_paths = random.sample(all_file_paths, num_samples)
    except OSError as e:
        print(f"Error: Error accessing directory '{data_dir}': {e}")
        return
    except Exception as e:
        print(f"Unknown error occurred while selecting files: {e}")
        return

    # --- Plot Setup ---
    # Create a large figure window to hold all subplots
    # Each sample is 2x2, total of num_samples samples
    # We create a grid of num_samples rows * 4 columns of subplots
    # Adjust figsize to fit content, can adjust width and height ratio as needed
    fig_width = 50 # Figure width (inches)
    fig_height = 10 * num_samples # Figure height (inches), allocate 4 inches height per row
    fig, axes = plt.subplots(num_samples, 4, figsize=(fig_width, fig_height), squeeze=False)

    fig.suptitle(f'Visualization of {num_samples} Randomly Selected Data Samples\nData Source: {os.path.basename(data_dir)}', fontsize=16, y=1.0) # y > 1 to avoid overlap with subplots

    print(f"Processing {num_samples} samples...")

    for i, file_path in enumerate(selected_file_paths):
        # file_path is already the full path
        filename = os.path.basename(file_path) # Get filename for display
        print(f"  Loading and plotting sample {i+1}/{num_samples}: {file_path}") # Print full path

        try:
            # Load data - assuming .pt file contains a dictionary
            data = torch.load(file_path, map_location=torch.device('cpu')) # Ensure loading on CPU

            # --- Extract Data ---
            # Extract data based on field names saved by generate_data.py
            initial_topo = data.get('initial_topo')  # Initial topography
            final_topo = data.get('final_topo')      # Final topography
            uplift = data.get('uplift_rate')         # Uplift rate

            # Build parameter dictionary
            params = {}
            param_keys = ['uplift_rate', 'k_f', 'k_d', 'm', 'n', 'run_time']
            for key in param_keys:
                if key in data:
                    params[key] = data[key]

            # Check if data exists and is a Tensor
            if initial_topo is None or not isinstance(initial_topo, torch.Tensor):
                print(f"Warning: 'initial_topo' not found in {file_path} or is not a Tensor. Skipping this subplot.")
                axes[i, 0].text(0.5, 0.5, 'Data Missing', ha='center', va='center', fontsize=10, color='red')
                axes[i, 0].set_xticks([])
                axes[i, 0].set_yticks([])
            else:
                # Remove batch dimension (if exists)
                if initial_topo.dim() > 2:
                    initial_topo = initial_topo.squeeze(0)  # Remove first dimension
                im = axes[i, 0].imshow(initial_topo.numpy(), cmap='terrain', origin='lower')
                axes[i, 0].set_title(f'Sample {i+1}: Initial Topography')
                axes[i, 0].set_xticks([])
                axes[i, 0].set_yticks([])
                divider = make_axes_locatable(axes[i, 0])
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(im, cax=cax)

            if final_topo is None or not isinstance(final_topo, torch.Tensor):
                print(f"Warning: 'final_topo' not found in {file_path} or is not a Tensor. Skipping this subplot.")
                axes[i, 1].text(0.5, 0.5, 'Data Missing', ha='center', va='center', fontsize=10, color='red')
                axes[i, 1].set_xticks([])
                axes[i, 1].set_yticks([])
            else:
                # Remove batch dimension (if exists)
                if final_topo.dim() > 2:
                    final_topo = final_topo.squeeze(0)  # Remove first dimension
                im = axes[i, 1].imshow(final_topo.numpy(), cmap='terrain', origin='lower')
                axes[i, 1].set_title(f'Sample {i+1}: Final Topography')
                axes[i, 1].set_xticks([])
                axes[i, 1].set_yticks([])
                divider = make_axes_locatable(axes[i, 1])
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(im, cax=cax)

            if uplift is None:
                print(f"Warning: 'uplift_rate' not found in {file_path}. Skipping this subplot.")
                axes[i, 2].text(0.5, 0.5, 'Data Missing', ha='center', va='center', fontsize=10, color='red')
                axes[i, 2].set_xticks([])
                axes[i, 2].set_yticks([])
            else:
                # Successfully found 'uplift_rate'
                try:
                    # Handle different types (Tensor or NumPy array)
                    if isinstance(uplift, torch.Tensor):
                        # Remove batch dimension (if exists)
                        if uplift.dim() > 2:
                            uplift = uplift.squeeze(0)  # Remove first dimension
                        uplift_data = uplift.numpy()
                    elif isinstance(uplift, np.ndarray):
                        # Already a NumPy array
                        uplift_data = uplift
                        # Remove batch dimension if needed
                        if uplift_data.ndim > 2 and uplift_data.shape[0] == 1:
                            uplift_data = uplift_data[0]
                    else:
                        # Try to convert to NumPy array
                        uplift_data = np.array(uplift)

                    # Display the data
                    im = axes[i, 2].imshow(uplift_data, cmap='coolwarm', origin='lower')
                    axes[i, 2].set_title(f'Sample {i+1}: Uplift Field (uplift_rate)')
                    axes[i, 2].set_xticks([])
                    axes[i, 2].set_yticks([])
                    divider = make_axes_locatable(axes[i, 2])
                    cax = divider.append_axes("right", size="5%", pad=0.05)
                    plt.colorbar(im, cax=cax)
                except Exception as e:
                    print(f"Error displaying uplift_rate: {e}")
                    axes[i, 2].text(0.5, 0.5, f'Error: {str(e)[:20]}...', ha='center', va='center', fontsize=8, color='red')
                    axes[i, 2].set_xticks([])
                    axes[i, 2].set_yticks([])


            # --- Display Parameters ---
            param_ax = axes[i, 3]
            param_ax.set_title(f'Sample {i+1}: Parameters')
            param_ax.set_xticks([])
            param_ax.set_yticks([])
            param_ax.spines['top'].set_visible(False)
            param_ax.spines['right'].set_visible(False)
            param_ax.spines['bottom'].set_visible(False)
            param_ax.spines['left'].set_visible(False)

            if not params:
                print(f"Warning: No parameters found in {file_path}. Skipping this subplot.")
                param_ax.text(0.5, 0.5, 'Parameters Missing', ha='center', va='center', fontsize=10, color='red')
            else:
                param_text = ""
                if isinstance(params, dict):
                    # If params is a dictionary
                    for key, value in params.items():
                        if isinstance(value, torch.Tensor):
                            # If value is a Tensor, try to get its value
                            if value.numel() == 1:
                                param_text += f"{key}: {value.item():.4f}\n"
                            else:
                                # If it's a multi-dimensional Tensor, may need special handling or just show shape
                                param_text += f"{key}: Tensor shape {tuple(value.shape)}\n"
                        else:
                            try:
                                param_text += f"{key}: {float(value):.4f}\n" # Try to convert to float and format
                            except (TypeError, ValueError):
                                param_text += f"{key}: {value}\n" # Other types display directly
                elif isinstance(params, torch.Tensor):
                    # If params is a Tensor
                    param_text += "Parameters (Tensor):\n"
                    if params.numel() < 10: # If not many elements, display directly
                         param_text += str(params.numpy())
                    else: # Otherwise show shape
                         param_text += f"Shape: {tuple(params.shape)}"
                else:
                    # Other unknown types
                    param_text = str(params)

                # Display text in center of subplot
                param_ax.text(0.05, 0.95, param_text.strip(), ha='left', va='top', fontsize=9, wrap=True)

        except FileNotFoundError:
            print(f"Error: File not found '{file_path}'.")
            axes[i, 0].text(0.5, 0.5, 'File Not Found', ha='center', va='center', fontsize=10, color='red')
            axes[i, 1].text(0.5, 0.5, 'File Not Found', ha='center', va='center', fontsize=10, color='red')
            axes[i, 2].text(0.5, 0.5, 'File Not Found', ha='center', va='center', fontsize=10, color='red')
            axes[i, 3].text(0.5, 0.5, 'File Not Found', ha='center', va='center', fontsize=10, color='red')
            for j in range(4):
                axes[i, j].set_xticks([])
                axes[i, j].set_yticks([])
        except Exception as e:
            print(f"Error occurred while processing file '{file_path}': {e}")
            # Display error message in corresponding row
            axes[i, 0].text(0.5, 0.5, f'Loading/Processing Error\n{e}', ha='center', va='center', fontsize=8, color='red', wrap=True)
            axes[i, 1].text(0.5, 0.5, 'Error', ha='center', va='center', fontsize=10, color='red')
            axes[i, 2].text(0.5, 0.5, 'Error', ha='center', va='center', fontsize=10, color='red')
            axes[i, 3].text(0.5, 0.5, 'Error', ha='center', va='center', fontsize=10, color='red')
            for j in range(4):
                axes[i, j].set_xticks([])
                axes[i, j].set_yticks([])


    # Adjust layout to prevent overlap
    plt.tight_layout(rect=[0, 0.03, 1, 0.98]) # rect=[left, bottom, right, top] leave space for title

    # Save image
    try:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"\nVisualization image saved to: {output_path}")
    except Exception as e:
        print(f"Error saving image: {e}")

    plt.close(fig) # Close figure, release memory

# --- Usage Example ---
if __name__ == "__main__":
    # !! Modify to your actual data directory !!
    data_directory = os.path.join( 'data', 'processed10')

    # !! Modify to the path and filename where you want to save the image !!
    output_image_path = os.path.join('PINN_Framework', 'data', 'processed_data_visualization.png') # Back to scripts directory

    # Ensure output directory exists
    output_dir = os.path.dirname(output_image_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    visualize_random_samples(data_directory, output_image_path, num_samples=10, dpi=300)