import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
import warnings
warnings.filterwarnings('ignore')

class OLEDOutcouplingTMM:
    def __init__(self):
        """Initialize OLED outcoupling efficiency calculator using TMM"""
        self.c = 2.998e8  # speed of light in m/s
        self.h = 6.626e-34  # Planck constant
        self.eps0 = 8.854e-12  # vacuum permittivity
        
    def define_oled_structure(self):
        """Define typical white TADF OLED layer structure"""
        # Layer thicknesses in nm
        self.thicknesses = {
            'cathode': 100,      # Al cathode
            'ETL': 30,           # Electron transport layer
            'EML': 20,           # Emissive layer with TADF
            'HTL': 40,           # Hole transport layer  
            'anode': 150,        # ITO anode
            'substrate': 500000  # Glass substrate (0.5mm)
        }
        
        # Complex refractive indices (wavelength dependent)
        self.materials = {
            'air': lambda wl: 1.0 + 0j,
            'glass': lambda wl: 1.52 + 0j,
            'ITO': lambda wl: 1.8 + 0.01j,  # Simplified ITO
            'HTL': lambda wl: 1.7 + 0.001j,  # Typical organic HTL
            'EML': lambda wl: 1.8 + 0.002j,  # TADF host + dopant
            'ETL': lambda wl: 1.8 + 0.001j,  # Typical organic ETL
            'Al': lambda wl: self.aluminum_index(wl)
        }
        
    def aluminum_index(self, wavelength_nm):
        """Aluminum complex refractive index using Drude model"""
        wl = wavelength_nm * 1e-9  # convert to meters
        # Simplified Drude model for Al
        wp = 2.24e16  # plasma frequency (rad/s)
        gamma = 1.22e14  # damping frequency (rad/s)
        omega = 2 * np.pi * self.c / wl
        
        eps_inf = 1.0
        epsilon = eps_inf - wp**2 / (omega**2 + 1j * gamma * omega)
        n_complex = np.sqrt(epsilon)
        return n_complex
        
    def transfer_matrix_layer(self, n, thickness, kz, polarization='TE'):
        """Calculate transfer matrix for a single layer"""
        if polarization == 'TE':
            # TE polarization
            M = np.array([
                [np.cos(kz * thickness), 1j * np.sin(kz * thickness) / kz],
                [1j * kz * np.sin(kz * thickness), np.cos(kz * thickness)]
            ], dtype=complex)
        else:  # TM polarization
            # TM polarization
            M = np.array([
                [np.cos(kz * thickness), 1j * np.sin(kz * thickness) / (n**2 * kz)],
                [1j * n**2 * kz * np.sin(kz * thickness), np.cos(kz * thickness)]
            ], dtype=complex)
        return M
        
    def interface_matrix(self, n1, n2, kz1, kz2, polarization='TE'):
        """Calculate interface matrix between two media"""
        if polarization == 'TE':
            # TE interface matrix
            I = 0.5 * np.array([
                [1 + kz2/kz1, 1 - kz2/kz1],
                [1 - kz2/kz1, 1 + kz2/kz1]
            ], dtype=complex)
        else:  # TM polarization
            # TM interface matrix
            I = 0.5 * np.array([
                [1 + n1**2*kz2/(n2**2*kz1), 1 - n1**2*kz2/(n2**2*kz1)],
                [1 - n1**2*kz2/(n2**2*kz1), 1 + n1**2*kz2/(n2**2*kz1)]
            ], dtype=complex)
        return I
        
    def calculate_kz(self, n, k0, k_parallel):
        """Calculate z-component of wavevector"""
        kz_squared = (k0 * n)**2 - k_parallel**2
        if np.real(kz_squared) >= 0:
            kz = np.sqrt(kz_squared)
        else:
            kz = 1j * np.sqrt(-kz_squared)
        # Choose branch with positive real part or negative imaginary part
        if np.real(kz) < 0:
            kz = -kz
        return kz
        
    def total_transfer_matrix(self, wavelength_nm, k_parallel, polarization='TE'):
        """Calculate total transfer matrix through all layers"""
        k0 = 2 * np.pi / (wavelength_nm * 1e-9)  # free space wavevector
        
        # Get refractive indices
        layers = ['air', 'glass', 'ITO', 'HTL', 'EML', 'ETL', 'Al']
        n_values = [self.materials[layer](wavelength_nm) for layer in layers]
        
        # Calculate kz for each layer
        kz_values = [self.calculate_kz(n, k0, k_parallel) for n in n_values]
        
        # Start with identity matrix
        M_total = np.eye(2, dtype=complex)
        
        # Glass substrate
        M_glass = self.transfer_matrix_layer(n_values[1], 
                                           self.thicknesses['substrate'] * 1e-9, 
                                           kz_values[1], polarization)
        I_glass = self.interface_matrix(n_values[0], n_values[1], 
                                       kz_values[0], kz_values[1], polarization)
        M_total = M_total @ I_glass @ M_glass
        
        # ITO anode  
        M_ITO = self.transfer_matrix_layer(n_values[2], 
                                         self.thicknesses['anode'] * 1e-9, 
                                         kz_values[2], polarization)
        I_ITO = self.interface_matrix(n_values[1], n_values[2], 
                                     kz_values[1], kz_values[2], polarization)
        M_total = M_total @ I_ITO @ M_ITO
        
        # HTL
        M_HTL = self.transfer_matrix_layer(n_values[3], 
                                         self.thicknesses['HTL'] * 1e-9, 
                                         kz_values[3], polarization)
        I_HTL = self.interface_matrix(n_values[2], n_values[3], 
                                     kz_values[2], kz_values[3], polarization)
        M_total = M_total @ I_HTL @ M_HTL
        
        # EML
        M_EML = self.transfer_matrix_layer(n_values[4], 
                                         self.thicknesses['EML'] * 1e-9, 
                                         kz_values[4], polarization)
        I_EML = self.interface_matrix(n_values[3], n_values[4], 
                                     kz_values[3], kz_values[4], polarization)
        M_total = M_total @ I_EML @ M_EML
        
        # ETL
        M_ETL = self.transfer_matrix_layer(n_values[5], 
                                         self.thicknesses['ETL'] * 1e-9, 
                                         kz_values[5], polarization)
        I_ETL = self.interface_matrix(n_values[4], n_values[5], 
                                     kz_values[4], kz_values[5], polarization)
        M_total = M_total @ I_ETL @ M_ETL
        
        # Final interface to Al cathode
        I_Al = self.interface_matrix(n_values[5], n_values[6], 
                                    kz_values[5], kz_values[6], polarization)
        M_total = M_total @ I_Al
        
        return M_total, kz_values[0]  # Return matrix and kz in air
        
    def outcoupling_efficiency_angle_wavelength(self, wavelength_nm, angle_deg):
        """Calculate outcoupling efficiency for given wavelength and angle"""
        k0 = 2 * np.pi / (wavelength_nm * 1e-9)
        n_air = 1.0
        
        # Convert angle to k_parallel
        k_parallel = k0 * n_air * np.sin(np.radians(angle_deg))
        
        # Calculate for both polarizations
        efficiencies = []
        for pol in ['TE', 'TM']:
            try:
                M_total, kz_air = self.total_transfer_matrix(wavelength_nm, k_parallel, pol)
                
                # Calculate transmission coefficient
                # For emission from EML, we need to consider the source term
                # Simplified approach: |t|^2 where t = 1/M_total[0,0]
                if abs(M_total[0, 0]) > 1e-10:
                    t = 1.0 / M_total[0, 0]
                    transmission = abs(t)**2 * np.real(kz_air) / np.real(k0)
                else:
                    transmission = 0.0
                    
                efficiencies.append(transmission)
            except:
                efficiencies.append(0.0)
                
        # Average over polarizations for unpolarized emission
        return np.mean(efficiencies)
        
    def plot_efficiency_vs_angle_wavelength(self):
        """Plot outcoupling efficiency as function of angle and wavelength"""
        # Define wavelength and angle ranges
        wavelengths = np.linspace(400, 700, 50)  # nm
        angles = np.linspace(0, 90, 46)  # degrees
        
        # Calculate efficiency matrix
        efficiency_matrix = np.zeros((len(wavelengths), len(angles)))
        
        for i, wl in enumerate(wavelengths):
            for j, angle in enumerate(angles):
                efficiency_matrix[i, j] = self.outcoupling_efficiency_angle_wavelength(wl, angle)
        
        # Create plots
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # 2D contour plot
        X, Y = np.meshgrid(angles, wavelengths)
        contour = ax1.contourf(X, Y, efficiency_matrix, levels=20, cmap='viridis')
        ax1.set_xlabel('Angle (degrees)')
        ax1.set_ylabel('Wavelength (nm)')
        ax1.set_title('Outcoupling Efficiency')
        plt.colorbar(contour, ax=ax1, label='Efficiency')
        
        # Efficiency vs angle at 550nm
        wl_idx = np.argmin(abs(wavelengths - 550))
        ax2.plot(angles, efficiency_matrix[wl_idx, :], 'b-', linewidth=2)
        ax2.set_xlabel('Angle (degrees)')
        ax2.set_ylabel('Outcoupling Efficiency')
        ax2.set_title('Efficiency vs Angle (550nm)')
        ax2.grid(True, alpha=0.3)
        
        # Efficiency vs wavelength at normal incidence
        ax3.plot(wavelengths, efficiency_matrix[:, 0], 'r-', linewidth=2)
        ax3.set_xlabel('Wavelength (nm)')
        ax3.set_ylabel('Outcoupling Efficiency')
        ax3.set_title('Efficiency vs Wavelength (0°)')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return wavelengths, angles, efficiency_matrix

# Example usage
if __name__ == "__main__":
    # Initialize calculator
    oled = OLEDOutcouplingTMM()
    
    # Define OLED structure
    oled.define_oled_structure()
    
    # Calculate and plot efficiency
    wavelengths, angles, efficiency = oled.plot_efficiency_vs_angle_wavelength()
    
    # Print some results
    print("OLED Outcoupling Efficiency Analysis")
    print("=" * 40)
    print(f"Maximum efficiency: {np.max(efficiency):.4f}")
    print(f"Efficiency at 550nm, 0°: {efficiency[np.argmin(abs(wavelengths-550)), 0]:.4f}")
    print(f"Average efficiency: {np.mean(efficiency):.4f}")
