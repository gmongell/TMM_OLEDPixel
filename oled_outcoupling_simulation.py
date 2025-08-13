import numpy as np
import matplotlib.pyplot as plt

# Define wavelength range (in nm)
wavelengths = np.linspace(400, 700, 301)

# Define refractive indices (n) and extinction coefficients (k) of the 5 sub-layers
# Example values (replace with actual data)
n_layers = [
    lambda wl: 1.5,  # Layer 1
    lambda wl: 1.7,  # Layer 2
    lambda wl: 1.8,  # Layer 3
    lambda wl: 1.6,  # Layer 4
    lambda wl: 1.4   # Layer 5
]
k_layers = [
    lambda wl: 0.0,  # Layer 1
    lambda wl: 0.01, # Layer 2
    lambda wl: 0.02, # Layer 3
    lambda wl: 0.01, # Layer 4
    lambda wl: 0.0   # Layer 5
]
thicknesses = [50, 100, 150, 200, 100]  # Thicknesses in nm

# Define the refractive indices of the surrounding medium (air) and substrate (glass)
n_medium = 1.0
n_substrate = 1.5

def transfer_matrix_method(wavelength, n_layers, k_layers, thicknesses, n_medium, n_substrate):
    # Calculate the complex refractive index for each layer
    n_complex = [n(wavelength) + 1j * k(wavelength) for n, k in zip(n_layers, k_layers)]
    
    # Initialize the total transfer matrix
    M_total = np.array([[1, 0], [0, 1]], dtype=complex)
    
    # Incident medium matrix (air)
    n_inc = n_medium
    theta_inc = 0  # Normal incidence
    k0 = 2 * np.pi / wavelength  # Wavenumber in vacuum

    # Loop through each layer and calculate the transfer matrix
    for i in range(len(n_complex)):
        n_layer = n_complex[i]
        d_layer = thicknesses[i] * 1e-9  # Convert thickness to meters
        
        # Calculate the phase shift in the layer
        delta = k0 * n_layer * d_layer * np.cos(theta_inc)
        
        # Calculate the transfer matrix for the layer
        M_layer = np.array([
            [np.cos(delta), 1j * np.sin(delta) / n_layer],
            [1j * n_layer * np.sin(delta), np.cos(delta)]
        ], dtype=complex)
        
        # Multiply the total transfer matrix by the current layer matrix
        M_total = np.dot(M_total, M_layer)
    
    # Substrate medium matrix (glass)
    n_sub = n_substrate
    
    # Calculate the reflection and transmission coefficients
    r = (n_inc * M_total[0, 0] + n_inc * n_sub * M_total[0, 1] - M_total[1, 0] - n_sub * M_total[1, 1]) / \
        (n_inc * M_total[0, 0] + n_inc * n_sub * M_total[0, 1] + M_total[1, 0] + n_sub * M_total[1, 1])
    t = 2 * n_inc / (n_inc * M_total[0, 0] + n_inc * n_sub * M_total[0, 1] + M_total[1, 0] + n_sub * M_total[1, 1])
    
    # Calculate the reflectance and transmittance
    R = np.abs(r)**2
    T = np.abs(t)**2 * (n_sub / n_inc).real
    
    # The outcoupling efficiency is equivalent to the transmittance in this simplified model
    return T

# Calculate outcoupling efficiency for each wavelength
outcoupling_efficiency = [
    transfer_matrix_method(wl, n_layers, k_layers, thicknesses, n_medium, n_substrate) for wl in wavelengths
]

# Plot the results
plt.plot(wavelengths, outcoupling_efficiency)
plt.xlabel('Wavelength (nm)')
plt.ylabel('Outcoupling Efficiency')
plt.title('Wavelength-Dependent Outcoupling Efficiency of OLED')
plt.grid(True)
plt.show()