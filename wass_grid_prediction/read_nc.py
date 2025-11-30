# Este script lee un NetCDF, elige un solo time step,
# y visualiza en 3D la grid 

import xarray as xr
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# librerias
# pip install xarray netcdf4 matplotlib numpy

def visualize_3d_grid(file_path):
    """
    Loads a NetCDF file, selects the first time step, and creates a 3D plot.

    Args:
        file_path (str): The path to the .nc file (local or URL).
    """
    try:
        # Open the dataset with xarray. The `decode_cf=False` and `chunks`
        # arguments help with memory efficiency and lazy loading.
        print(f"Loading dataset from: {file_path}")
        ds = xr.open_dataset(file_path, engine='netcdf4', decode_cf=False)
        
        print("\nDataset opened successfully. Contents (variables and dimensions):\n")
        
        # Print dimensions and variables for confirmation
        print("Variables:", list(ds.data_vars.keys()))
        print("Z shape:", ds['Z'].shape)
        
        # Select the first time step and relevant variables
        time_index = 0
        ds_slice = ds.isel(time=time_index)

        # Get the raw data, X, Y coordinates, and mask
        raw_data = ds_slice['Z'].values
        X = ds_slice['X'].values
        Y = ds_slice['Y'].values


        # Get and apply scale_factor and add_offset for correct physical values
        scale_factor = ds['Z'].attrs.get('scale_factor', 1.0)
        add_offset = ds['Z'].attrs.get('add_offset', 0.0)
        data = (raw_data * scale_factor) + add_offset
                
        X= (X * scale_factor) + add_offset
        Y= (Y * scale_factor) + add_offset
        # Use the X and Y arrays directly as they are already in meters
        X_meters = X 
        Y_meters = Y 

        # Load the Z mask and apply it to the Z data
        if 'mask_Z' in ds_slice.data_vars:
            mask_data = ds_slice['mask_Z'].values
            # Apply the mask. Values in mask_Z of 0 likely indicate invalid data.
            data = np.ma.masked_where(mask_data == 0, data)
            print("\nMask 'mask_Z' applied to the Z data.")
        else:
            # If no mask_Z is found, fall back to the fill value.
            fill_value = ds['Z'].attrs.get('_FillValue', None)
            if fill_value is not None:
                print(f"\nDetected _FillValue: {fill_value}. Masking these values.")
                data = np.ma.masked_equal(data, fill_value)
        
        print(f"\nSuccessfully loaded and processed a single 3D mesh for time index: {time_index}")

        # --- Debugging Prints ---
        print("\n--- Plotting Debug Info ---")
        print(f"Shape of X_meters: {X_meters.shape}")
        print(f"Shape of Y_meters: {Y_meters.shape}")
        print(f"Shape of data: {data.shape}")
        print(f"Min/Max of X_meters: {np.min(X_meters)}, {np.max(X_meters)}")
        print(f"Min/Max of Y_meters: {np.min(Y_meters)}, {np.max(Y_meters)}")
        print(f"Min/Max of data: {np.min(data)}, {np.max(data)}")
        print("--- End Debug Info ---\n")

        # Create a 3D plot.
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        surf = ax.plot_surface(X_meters, Y_meters, data, cmap='viridis', rstride=1, cstride=1, antialiased=False)

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z Value (m)')
        ax.set_title(f'3D Grid of Z at Time Index {time_index}')

        fig.colorbar(surf, shrink=0.5, aspect=5, label='Z Value (m)')

        plt.show()

    except FileNotFoundError:
        print(f"Error: The file at '{file_path}' was not found.")
        print("Please ensure the path is correct or the URL is valid and accessible.")
    except Exception as e:
        print(f"An error occurred: {e}")

# IMPORTANT: Replace the placeholder below with the actual path or URL of your .nc file.
file_path = "Surfaces_20150305_103500.nc"

# Run the visualization function.
if __name__ == "__main__":
    visualize_3d_grid(file_path)
