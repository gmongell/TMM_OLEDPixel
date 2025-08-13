import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use the Agg backend
import matplotlib.pyplot as plt
from tmm import coh_tmm

# Constants
wavelengths = np.linspace(400, 700, 100)  # Visible spectrum: 400 nm to 700 nm

# Define the refractive indices of each layer
n_air = 1.0  # Air
n_glass = 1.5  # Glass substrate
n_ito = 1.9  # ITO (Indium Tin Oxide)
n_alq3 = 1.7  # Alq3 refractive index
n_silver = 0.14 + 4.49j  # Silver (Ag) metal back contact (complex refractive index)

# Define layer thicknesses in nm
thickness_air = np.inf  # Air is semi-infinite
thickness_glass = 1000  # Glass substrate
thickness_ito = 100  # ITO layer thickness
thickness_alq3 = 100  # Organic emitting layer (Alq3)
thickness_silver = 100  # Silver back contact is semi-infinite in the model

# Define n_list (refractive indices) and d_list (layer thicknesses)
n_list = [n_air, n_glass, n_ito, n_alq3, n_silver, n_air]  # List of refractive indices
d_list = [thickness_air, thickness_glass, thickness_ito, thickness_alq3, thickness_silver, thickness_air]  # List of layer thicknesses

# Ensure n_list and d_list have the same length
assert len(n_list) == len(d_list), "n_list and d_list must have the same length"

# Initialize results for the angle-dependent efficiency across wavelengths
theta_list_deg = np.linspace(0, 90, 500)  # Angle in degrees
theta_list_rad = np.radians(theta_list_deg)  # Convert angles to radians

# Initialize a 2D array to store the outcoupling efficiency for each (angle, wavelength)
outcoupling_efficiency_2D = np.zeros((len(wavelengths),len(theta_list_deg)))

# Loop over wavelengths and angles
for i, wavelength in enumerate(wavelengths):
    R_TE, T_TE, R_TM, T_TM = [], [], [], []
    
    for theta in theta_list_rad:
        # Check if the angle is valid for the first layer
        if theta < np.arcsin(n_air / n_alq3):  # Ensure we are not exceeding critical angle
            # TE polarization (s-polarized light)
            coh_TE=coh_tmm('s',n_list,d_list,theta,wavelength)
            R_TE.append(coh_TE['R'])
            T_TE.append(coh_TE['T'])
            
            # TM polarization (p-polarized light)
            coh_TM=coh_tmm('p',n_list,d_list,theta,wavelength)
            R_TM.append(coh_TM['R'])
            T_TM.append(coh_TM['T'])
        else:
            R_TE.append(1.0)  # Total reflection
            T_TE.append(0.0)  # No transmission
            R_TM.append(1.0)  # Total reflection
            T_TM.append(0.0)  # No transmission
    
    # Calculate unpolarized reflection and transmission
    R_TE = np.array(R_TE)
    T_TE = np.array(T_TE)
    R_TM = np.array(R_TM)
    T_TM = np.array(T_TM)
    
    R_unpolarized = (R_TE + R_TM) / 2
    T_unpolarized = (T_TE + T_TM) / 2
    
    # Outcoupling efficiency is the transmission to air (T_unpolarized)
    outcoupling_efficiency = T_unpolarized
    
    # Store outcoupling efficiency for this wavelength and angle
    outcoupling_efficiency_2D[i, :] = outcoupling_efficiency

# Create a 2D plot (heatmap) of outcoupling efficiency as a function of angle and wavelength
plt.figure(figsize=(12, 8))
plt.imshow(outcoupling_efficiency_2D,aspect='auto',extent=[0,90,400,700],origin='lower',cmap='inferno')
plt.colorbar(label='Outcoupling Efficiency')
plt.xlabel('Angle (degrees)')
plt.ylabel('Wavelength (nm)')
plt.title('Outcoupling Efficiency of OLED (Alq3 Layer, Ag Back Contact)')
plt.show() 

# Save the plot as an image file
plt.savefig('outcoupling_efficiency_heatmap.png',dpi=300,bbox_inches='tight')  # Save as PNG with 300 dpi

# Optionally, calculate the total outcoupling efficiency by integrating over angles and wavelengths
total_efficiency_wavelengths = np.trapz(outcoupling_efficiency_2D, theta_list_deg, axis=1)
total_efficiency = np.trapz(total_efficiency_wavelengths, wavelengths) / (90 * (700 - 400))
print(f"Total Outcoupling Efficiency (Integrated over Wavelength and Angle): {total_efficiency:.4f}")
