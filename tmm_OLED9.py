import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use Agg backend for headless environments
import matplotlib.pyplot as plt

# Dispersion relationships for materials
def n_alq3(wavelength):
    A = 1.65
    B = 0.5  # Adjust as necessary
    return A + B / (wavelength ** 2)

def n_silver(wavelength):
    n_0 = 0.16  # Real part
    kappa = 3.16  # Imaginary part
    C = 1.5  # Coefficient for the dispersion
    return n_0 - (C / (wavelength ** 2)) + 1j * kappa

# Wavelength and angle ranges
wavelengths = np.linspace(400, 700, 1000)  # Wavelength range (in nm)
theta_list_deg = np.linspace(0, 90, 180)  # Angles from 0 to 90 degrees
theta_list_rad = np.radians(theta_list_deg)  # Convert to radians

# Set thickness range for the organic layer (e.g., Alq3) in nanometers
thicknesses = np.linspace(25, 150, 10)  # Example thickness values, from 50nm to 200nm

# Loop over different thickness values of the organic layer
for thickness in thicknesses:
    # Initialize the outcoupling efficiency array for this thickness
    outcoupling_efficiency_2D = np.zeros((len(wavelengths), len(theta_list_deg)))

    # Create the n_list for the current wavelength range and thickness
    n_air = 1.0  # Refractive index of air
    n_list = [n_air]  # Start with air
    for wavelength in wavelengths:
        n_list.append(n_alq3(wavelength))  # Alq3 layer refractive index
    n_list.append(n_silver(wavelengths[-1]))  # Silver refractive index at last wavelength

    # Set the layer thickness values
    d_alq3 = thickness * 1e-9  # Convert nm to meters
    d_silver = 50e-9  # 50 nm for Silver
    d_list = [np.inf, d_alq3, d_silver, np.inf]  # Air, Alq3, Silver, and back air

    # Loop over each wavelength and angle to compute the outcoupling efficiency
    for i, wavelength in enumerate(wavelengths):
        R_TE, T_TE, R_TM, T_TM = [], [], [], []
        
        for theta in theta_list_rad:
            # Ensure angle is valid for first layer (Alq3)
            if theta < np.arcsin(n_air / np.abs(n_list[1])):
                # TE polarization (s-polarized light)
                coh_TE = coh_tmm('s', n_list, d_list, wavelength * 1e-9, theta)
                R_TE.append(coh_TE['R'])
                T_TE.append(coh_TE['T'])

                # TM polarization (p-polarized light)
                coh_TM = coh_tmm('p', n_list, d_list, wavelength * 1e-9, theta)
                R_TM.append(coh_TM['R'])
                T_TM.append(coh_TM['T'])
            else:
                R_TE.append(1.0)  # Total reflection
                T_TE.append(0.0)  # No transmission
                R_TM.append(1.0)  # Total reflection
                T_TM.append(0.0)  # No transmission

        # Combine TE and TM transmission efficiencies to compute the outcoupling efficiency
        outcoupling_efficiency = np.array(T_TE) + np.array(T_TM)
        outcoupling_efficiency_2D[i, :] = outcoupling_efficiency

    # Plotting the outcoupling efficiency for this thickness as a heatmap
    plt.figure(figsize=(10, 6))
    plt.contourf(theta_list_deg, wavelengths, outcoupling_efficiency_2D, levels=50, cmap='viridis')
    plt.colorbar(label='Outcoupling Efficiency')
    plt.xlabel('Angle of Incidence (degrees)')
    plt.ylabel('Wavelength (nm)')
    plt.title(f'Outcoupling Efficiency (Alq3 Thickness = {thickness:.1f} nm)')

    # Save the plot as an image file
    plt.savefig(f'outcoupling_efficiency_thickness_{thickness:.1f}nm.png')
    plt.close()  # Close the figure to free up memory

