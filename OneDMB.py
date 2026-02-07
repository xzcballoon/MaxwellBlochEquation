import numpy as np
import scipy.constants as const
import matplotlib.pyplot as plt

import matplotlib.animation as animation #to draw animations

###to solve 1 dimensional Maxwell Bloch equations




##solve 1D Maxwell's equations coupled with Bloch equations for a two-level medium
##FDTD method for Maxwell's equations
##Runge-Kutta 4th order method for Bloch equations
class OneDimensionMaxwellBloch:
    # Initialize the parameters for the simulation
    def __init__(self,
                 E0: np.ndarray,  # Initial electric field (V/m)
                 H0: np.ndarray,  # Initial magnetic field (A/m)
                 rho0: list, # Initial density matrix elements, rho0[i] is a 2D array for the i-th spatial point
                 mu: list,   # list of dipole moment operators
                 H_free: np.ndarray, # Free Hamiltonian of the two-level system
                 dz=1e-6,          # Spatial step size (m)
                 dt=1e-15,         # Time step size (s)
                 t_start=0,        # Start time (s)
                 t_end=1e-12,      # End time (s)
                z_start=0,        # Start position (m)
                z_end=1e-3,       # End position (m)
                t = None,
                z = None,
                boundary_conditions=('periodic', 'periodic')
    ):
        self.E0 = E0 # Intial electric field. It is a 1D array with shape (Nz, )
        self.H0 = H0 # Initial magnetic field. It is a 1D array with shape (Nz, )
        self.rho0 = rho0 # Initial density matrix elements.
        self.mu = mu # Dipole moment matrix elements. It is a 1D array with shape (Nz, )
        #the initial condition is the field at time t=0 for electric field and t = dt/2 for magnetic field
        self.dz = dz # Spatial step size
        self.dt = dt # Time step size
        self.t_start = t_start # Start time
        self.t_end = t_end # End time
        self.z_start = z_start # Start position
        self.z_end = z_end # End position
        
        self.n0 = 0.0 #number density of medium
        self.mu_two_level = 1e-29 # Dipole moment of the two-level system (C*m)
        self.gamma_0_two_level = 0.042e12
        self.gamma_1_two_level = 6.2e8
        self.gap_two_level = 1.5e15 # energy gap in rad *s^{-1}
        
        
        #generate spatial grid
        if z is None:
            self.z = np.arange(z_start, z_end, dz)
        else:
            self.z = z
        self.Nz = len(self.z) # Number of spatial points
        #generate time grid
        if t is None:
            self.t = np.arange(t_start, t_end, dt)
        else:
            self.t = t 
        self.Nt = len(self.t) # Number of time points
        ##boundary conditions
        self.boundary_conditions = boundary_conditions
        
        self.H_free = H_free # Free Hamiltonian of the two-level system
        
    # Update the electric and magnetic fields using FDTD method
    def update_fields_vacuum(self, E: np.ndarray, H: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # Update magnetic field H
        #argument Pprime is the time derivative of polarization P, i.e., dP/dt
        
        
        H_new = np.copy(H)
        for i in range(1, self.Nz):
            H_new[i] = H[i] - (self.dt / (const.mu_0 * self.dz)) * (E[i] - E[i - 1])
        
        # Apply boundary conditions for H
        if self.boundary_conditions[0] == 'periodic':
            H_new[0] = H_new[-2]
        elif self.boundary_conditions[0] == 'absorbing':
            H_new[0] = H[0]  # Simple absorbing boundary condition
        
        # Update electric field E
        E_new = np.copy(E)
        for i in range(self.Nz - 1):
            E_new[i] = E[i] - (self.dt / (const.epsilon_0 * self.dz)) * (H_new[i + 1] - H_new[i])
        
        # Apply boundary conditions for E
        if self.boundary_conditions[1] == 'periodic':
            E_new[-1] = E_new[1]
        elif self.boundary_conditions[1] == 'absorbing':
            E_new[-1] = E[-1]  # Simple absorbing boundary condition
        
        return E_new, H_new
    
    def update_fields_two_level(self, E:np.ndarray, H: np.ndarray, a:np.ndarray,b:np.ndarray,u:np.ndarray, t_now=0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray,np.ndarray,np.ndarray]:
        '''
        Calculate the new electric and magnetic fields by solving Maxwell's equations with the polarization term from the two-level medium.
        Argus:
        E: initial electric field, 1D array with shape (Nz, )
        H: initial magnetic field, 1D array with shape (Nz, )
        a: initial real part of the off-diagonal element of the density matrix, 1D array with shape (Nz, )
        b: initial imaginary part of the off-diagonal element of the density matrix, 1D array with shape (Nz, )
        u: initial population inversion, 1D array with shape (Nz, )
        t_now: current time, used for calculating the polarization term. It is a scalar.
        
            The density matrix elements are related to a, b, u as follows:
            rho_12 = a + 1j * b
            rho_21 = a - 1j * b
            rho_11 - rho_22 = u
            
        
        Returns:
        E_new: updated electric field, 1D array with shape (Nz, )
        H_new: updated magnetic field, 1D array with shape (Nz, )
        a_new: updated real part of the off-diagonal element of the density matrix, 1D array with shape (Nz, )
        b_new: updated imaginary part of the off-diagonal element of the density matrix, 1D array with shape (Nz, )
        u_new: updated population inversion, 1D array with shape (Nz, )
        '''
        
        ##begin calculation###
        
        #first, update the magnetic field H using the vacuum update method
        H_new = np.copy(H)
        for i in range(1, self.Nz):
            H_new[i] = H[i] - (self.dt / (const.mu_0 * self.dz)) * (E[i] - E[i - 1])
        #boundary condition for H
        if self.boundary_conditions[0] == 'periodic':
            H_new[0] = H_new[-2]
        elif self.boundary_conditions[0] == 'absorbing':
            H_new[0] = H[0]  # Simple absorbing boundary condition
        
        #next, update the density matrix elements a, b, u using the coupled Maxwell-Bloch equations
        
        #define some temporary variables to accerelate the calculation
        #avoiding repeated calculation of the same terms
        Gamma0 = self.gamma_0_two_level
        Gamma1 = self.gamma_1_two_level
        
        
        
        
        Cpn = np.exp((Gamma1-Gamma0)*t_now) # population decay term
        
        temp_1 = 2.0*b + Cpn * E *u * self.dt * self.mu_two_level #a array that will be used a lot in the calculation of a_new and b_new
        temp_2 = 4.0+ self.dt**2 *(2.0*E*E*self.mu_two_level**2 + self.gap_two_level) #a denominator term that will be used in the calculation of a_new and b_new
        
        #update a, b, u using the Maxwell-Bloch equations
        a_new = a + 2.0*self.dt *self.gap_two_level * temp_1 / temp_2
        b_new = -b + 4*temp_1/temp_2
        u_new = u - 4.0*E*self.dt*self.mu_two_level * temp_1 / (Cpn*temp_2)
        
        #next, update the electric field E using the polarization term from the two-level medium
        An = np.exp(-Gamma1*(t_now + self.dt/2)) # coherence decay term
        
        
        E_new = np.copy(E)
        for i in range(self.Nz - 1):
            #the polarization term is calculated using the updated density matrix elements a_new, b_new, u_new
            #not that the electric field is always real, so as a, b, u, the polarization term is also real
            
            E_new[i] = E[i] - (self.dt / (const.epsilon_0 * self.dz)) * (H_new[i + 1] - H_new[i]) - self.dt * An * self.n0 *(b_new[i]*self.gamma_0_two_level - a_new[i]*Gamma1)
        
        # Apply boundary conditions for E
        if self.boundary_conditions[1] == 'periodic':
            E_new[-1] = E_new[1]
        elif self.boundary_conditions[1] == 'absorbing':
            E_new[-1] = E[-1]  # Simple absorbing boundary condition
            
            
        #also applie boundary conditions for a, b, u like electric field, assuming the medium is also periodic or absorbing at the boundaries
        if self.boundary_conditions[0] == 'periodic':
            a_new[0] = a_new[-2]
            b_new[0] = b_new[-2]
            u_new[0] = u_new[-2]
        elif self.boundary_conditions[0] == 'absorbing':
            a_new[0] = a[0]
            b_new[0] = b[0]
            u_new[0] = u[0]
        #TODO check the stability of the update method, and adjust the time step if necessary
        
        return E_new, H_new, a_new, b_new, u_new

    
    def calculate_polarization(self, rho: list) -> np.ndarray:
        #trace of the dipole moment matrix multiplied by the density matrix
        P = np.zeros(self.Nz)
        
        for i in range(self.Nz):
            P[i] = np.trace(self.mu[i] @ rho[i])
        return P
    
    def update_density_matrix(self, rho: list, E: np.ndarray) -> list:
        # Update the density matrix using the Runge-Kutta 4th order method
        
        
        
        def bloch_equations(rho_i, E_i):
            # Define the Bloch equations for a two-level system
            H_int = -self.mu[i] * E_i  # Interaction Hamiltonian
            H_total = self.H_free + H_int
            drho_dt = -1j / const.hbar * (H_total @ rho_i - rho_i @ H_total)
            return drho_dt
        rho_new = []
        for i in range(self.Nz):
            k1 = bloch_equations(rho[i], E[i])
            k2 = bloch_equations(rho[i] + 0.5 * self.dt * k1, E[i])
            k3 = bloch_equations(rho[i] + 0.5 * self.dt * k2, E[i])
            k4 = bloch_equations(rho[i] + self.dt * k3, E[i])
            rho_i_new = rho[i] + (self.dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
            rho_new.append(rho_i_new)
        return rho_new
    # Run the simulation
    def run_simulation_vacuum(self) -> tuple[np.ndarray, np.ndarray, list]:
        # Initialize fields and density matrix
        E = np.copy(self.E0)
        H = np.copy(self.H0)
        rho = self.rho0
        
        # Store results
        E_history = np.zeros((self.Nt, self.Nz))
        H_history = np.zeros((self.Nt, self.Nz))
        rho_history = []
        
        for n in range(self.Nt):
            # Store current state
            E_history[n, :] = E
            H_history[n, :] = H
            rho_history.append([rho_i.copy() for rho_i in rho])
            
            # Calculate polarization
            P = self.calculate_polarization(rho)
            
            # Update fields
            E, H = self.update_fields_vacuum(E, H, P)
            
            # Update density matrix
            rho = self.update_density_matrix(rho, E)
        
        return E_history, H_history, rho_history
    
    def run_simulation_two_level(self) -> tuple[np.ndarray, np.ndarray, list]:
        # Initialize fields and density matrix
        E = np.copy(self.E0)
        H = np.copy(self.H0)
        rho = self.rho0
        
        #check whether the density matrix is 2*2 for each spatial point
        for rho_i in rho:
            if rho_i.shape != (2, 2):
                raise ValueError("Density matrix must be 2x2 for each spatial point.")
        
        # Store results
        E_history = np.zeros((self.Nt, self.Nz))
        H_history = np.zeros((self.Nt, self.Nz))
        a_history = np.zeros((self.Nt, self.Nz))
        b_history = np.zeros((self.Nt, self.Nz))
        u_history = np.zeros((self.Nt, self.Nz))
        
        for n in range(self.Nt):
            # Store current state
            E_history[n, :] = E
            H_history[n, :] = H
            a_history[n, :] = [np.real(rho_i[0, 1]) for rho_i in rho]
            b_history[n, :] = [np.imag(rho_i[0, 1]) for rho_i in rho]
            u_history[n, :] = [rho_i[0, 0] - rho_i[1, 1] for rho_i in rho]
            
            # Update fields and density matrix using the coupled Maxwell-Bloch equations
            E, H, a_new, b_new, u_new = self.update_fields_two_level(E, H, a_history[n], b_history[n], u_history[n], t_now=self.t[n])
            
            # Update density matrix elements a, b, u for the next iteration
            a_history[n] = a_new
            b_history[n] = b_new
            u_history[n] = u_new
            
            # Reconstruct the density matrix from a, b, u for the next iteration
            rho = []
            for i in range(self.Nz):
                rho_i = np.array([[ (u_new[i] + 1) / 2 , a_new[i] + 1j * b_new[i]],
                                  [a_new[i] - 1j * b_new[i], (1 - u_new[i]) / 2 ]], dtype=complex)
                rho.append(rho_i)
        
        return E_history, H_history, a_history, b_history, u_history
    
    def animate_vacuum_fields(self, E_history: np.ndarray, H_history: np.ndarray, file_name="fields"): #draw the time evolution of the fields
        fig, ax = plt.subplots(2, 1, figsize=(10, 8))
        
        line1, = ax[0].plot(self.z, E_history[0, :], color='b')
        ax[0].set_title('Electric Field (E)')
        ax[0].set_xlabel('Position (m)')
        ax[0].set_ylabel('E (V/m)')
        ax[0].set_ylim(np.min(E_history), np.max(E_history))
        
        line2, = ax[1].plot(self.z, H_history[0, :], color='r')
        ax[1].set_title('Magnetic Field (H)')
        ax[1].set_xlabel('Position (m)')
        ax[1].set_ylabel('H (A/m)')
        ax[1].set_ylim(np.min(H_history), np.max(H_history))
        
        def update(frame):
            line1.set_ydata(E_history[frame, :])
            line2.set_ydata(H_history[frame, :])
            return line1, line2
        
        ani = animation.FuncAnimation(fig, update, frames=self.Nt, blit=True, interval=50)
        plt.tight_layout()
        #save the animation as gif file
        ani.save(f"{file_name}.gif", writer='pillow')
        
    

class OneDimensionMaxwellBloch_MultipleFrequencies:
    def __init__(self,
                 E0s: list,  # List of initial electric fields for different frequencies (each is a 1D array). They are envelopes. 
                    H0s: list,  # List of initial magnetic fields for different frequencies (each is a 1D array)
                    rho0: list, # Initial density matrix elements, rho0[i] is a 2D array for the i-th spatial point
                    mus: list,   # list of dipole moment operators. mus[i] corresponds to the i-th frequency. It is independent of spatial position and time.
                    central_frequencies: list, # central_frequencies[i] corresponds to the i-th frequency of the field, which is E0s[i] and H0s[i]. 
                    H_free: np.ndarray, # Free Hamiltonian of the two-level system
                    dz=1e-6,          # Spatial step size (m)
                    dt=1e-15,         # Time step size (s)
                    t_start=0,        # Start time (s)
                    t_end=1e-12,      # End time (s)
                    z_start=0,        # Start position (m)
                    z_end=1e-3,       # End position (m)
                    boundary_conditions=('periodic', 'periodic')
    ):
        self.E0s = E0s # List of initial electric fields for different frequencies
        self.H0s = H0s # List of initial magnetic fields for different frequencies
        self.rho0 = rho0 # Initial density matrix elements.
        self.mus = mus # List of dipole moment matrix elements for different frequencies
        self.H_free = H_free # Free Hamiltonian of the two-level system
        self.dz = dz # Spatial step size
        self.dt = dt # Time step size
        self.dtz = dt / dz # ratio of time step to spatial step
        self.t_start = t_start # Start time
        self.t_end = t_end # End time
        self.z_start = z_start # Start position
        self.z_end = z_end # End position
        
        self.central_frequencies = central_frequencies
        self.central_wavevectors = [2 * np.pi * f / const.c for f in central_frequencies]
        
        #generate spatial grid
        self.z = np.arange(z_start, z_end, dz)
        self.Nz = len(self.z) # Number of spatial points
        #generate time grid
        self.t = np.arange(t_start, t_end, dt)
        self.Nt = len(self.t) # Number of time points
        ##boundary conditions
        self.boundary_conditions = boundary_conditions
        
    def update_fields_envelope(self, E_envelopes: list, H_envelopes: list, P_envelope: np.ndarray, central_frequencies: list, central_wavevector: list) -> tuple[list, list]:
        # Update magnetic field envelopes H
        H_new_envelopes = []
        for idx, H in enumerate(H_envelopes):
            H_new = np.zeros_like(H, dtype=complex)
            for i in range(1, self.Nz):
                #formula: dH/dt  - 1j*central_frequencies[idx] * H = - (1/mu0) dE/dz - 1j*central_wavevector[idx] * (1/mu0) * E 
                #dicretized form:
                # (H_new[i] - H[i]) / dt - 1j*central_frequencies[idx] * (H_new[i] + H[i]) / 2 = - (1/mu0) * (E[i] - E[i - 1]) / dz - 1j*central_wavevector[idx] * (1/mu0) * (E[i] + E[i - 1]) / 2
                # Rearranging gives:
                
                
                #incorrect version:
                # H_new[i] = (
                #     H[i]*(2j - central_frequencies[idx] * self.dt) + self.dtz*((2j+central_wavevector[idx]*self.dz)*E_envelopes[idx][i-1] + (-2j + central_wavevector[idx]*self.dz)*E_envelopes[idx][i])/const.mu_0
                #     ) / (2j + central_frequencies[idx] * self.dt)             
                
                
                H_new[i] = (H[i] - (self.dt / (const.mu_0 * self.dz)) * (E_envelopes[idx][i] - E_envelopes[idx][i - 1]) - 1j * central_wavevector[idx] * (self.dt / const.mu_0) * (E_envelopes[idx][i] + E_envelopes[idx][i - 1]) / 2) / (1 + 1j * central_frequencies[idx] * self.dt / 2)
                
                
                # Apply boundary conditions for H
            if self.boundary_conditions[0] == 'periodic':
                H_new[0] = H_new[-2]
            elif self.boundary_conditions[0] == 'absorbing':
                H_new[0] = H[0]  # Simple absorbing boundary condition
            H_new_envelopes.append(H_new)
        
        # Update electric field envelopes E
        E_new_envelopes = []
        for idx, E in enumerate(E_envelopes):
            E_new = np.zeros_like(E, dtype=complex)
            for i in range(self.Nz - 1):
                #formula: dE/dt - 1j*central_frequencies[idx] * E = - (1/epsilon0) dH/dz - 1j*central_wavevector[idx] * (1/epsilon0) * H  - (1/epsilon0) * P
                #dicretized form:
                # (E_new[i] - E[i]) / dt - 1j*central_frequencies[idx] * (E_new[i] + E[i]) / 2 = - (1/epsilon0) * (H_new[i + 1] - H_new[i]) / dz - 1j*central_wavevector[idx] * (1/epsilon0) * (H_new[i + 1] + H_new[i]) / 2 - (1/epsilon0) * P_envelope[i]
                # Rearranging gives:
                
                #H_new = H_new_envelopes[idx]
                
                
                #incorrect version:
                # E_new[i] = (E[i]*(
                #     2j - central_frequencies[idx] * self.dt) + (
                #         self.dtz/const.epsilon_0)*(
                #             (2j + central_wavevector[idx]*self.dz)*H_new_envelopes[idx][i-1] + (-2j + central_wavevector[idx]*self.dz)*H_new_envelopes[idx][i]
                #             ) - (2j * self.dt / const.epsilon_0) * P_envelope[i]) / (2j + central_frequencies[idx] * self.dt)


                
                E_new[i] = (E[i] - (self.dt / (const.epsilon_0 * self.dz)) * (H_new_envelopes[idx][i + 1] - H_new_envelopes[idx][i]) - 1j * central_wavevector[idx] * (self.dt / const.epsilon_0) * (H_new_envelopes[idx][i + 1] + H_new_envelopes[idx][i]) / 2 - (self.dt / const.epsilon_0) * P_envelope[i]) / (1 + 1j * central_frequencies[idx] * self.dt / 2)
                
            # Apply boundary conditions for E
            if self.boundary_conditions[1] == 'periodic':
                E_new[-1] = E_new[1]
            elif self.boundary_conditions[1] == 'absorbing':
                E_new[-1] = E[-1]  # Simple absorbing boundary condition
            E_new_envelopes.append(E_new)
        return E_new_envelopes, H_new_envelopes
    
    def calculate_polarization_envelope(self, rho: list, central_frequencies: list) -> np.ndarray:
        #trace of the dipole moment matrix multiplied by the density matrix
        P_envelope = np.zeros(self.Nz, dtype=complex)
        
        for i in range(self.Nz):
            for idx, mu in enumerate(self.mus):
                P_envelope[i] += np.trace(mu @ rho[i]) * np.exp(-1j * central_frequencies[idx] * self.t[0])  # assuming t=0 for envelope calculation
        return P_envelope
    
    
    #TODO add run_simulation and animate_fields methods for this class
    def update_density_matrix(self, rho: list, E_envelope: np.ndarray) -> list:
        #TODO similar to the previous class, but now E_envelope is the total electric field envelope (sum of all frequency components)
        
        return np.zeros_like(rho) #placeholder implementation
    
    def run_simulation(self) -> tuple[list, list, list]:
        # Initialize fields and density matrix
        E_envelopes = [np.copy(E0) for E0 in self.E0s]
        H_envelopes = [np.copy(H0) for H0 in self.H0s]
        rho = self.rho0
        
        # Store results
        E_history = [[] for _ in range(len(self.E0s))]
        H_history = [[] for _ in range(len(self.H0s))]
        rho_history = []
        
        for n in range(self.Nt):
            # Store current state
            for idx in range(len(self.E0s)):
                E_history[idx].append(E_envelopes[idx].copy())
                H_history[idx].append(H_envelopes[idx].copy())
            rho_history.append([rho_i.copy() for rho_i in rho])
            
            # Calculate polarization envelope
            P_envelope = self.calculate_polarization_envelope(rho, self.central_frequencies)
            
            # Update fields envelopes
            E_envelopes, H_envelopes = self.update_fields_envelope(E_envelopes, H_envelopes, P_envelope, self.central_frequencies, self.central_wavevectors)
            
            # Update density matrix
            # Here we use the total electric field envelope (sum of all frequency components) to update the density matrix
            total_E_envelope = np.sum(E_envelopes, axis=0)
            rho = self.update_density_matrix(rho, total_E_envelope)
        
        return E_history, H_history, rho_history

    def animate_fields(self, E_history: list, H_history: list, file_name="fields_multiple_frequencies"): #draw the time evolution of the fields
        #plot the intensity (magnitude squared) of each frequency component
        #each is saved as a separate gif file
        for idx in range(len(E_history)):
            fig, ax = plt.subplots(2, 1, figsize=(10, 8))
            
            line1, = ax[0].plot(self.z, np.abs(E_history[idx][0])**2, color='b')
            ax[0].set_title(f'Electric Field Intensity (Frequency {self.central_frequencies[idx]} Hz)')
            ax[0].set_xlabel('Position (m)')
            ax[0].set_ylabel('|E|^2 (V^2/m^2)')
            ax[0].set_ylim(0, np.max(np.abs(E_history[idx])**2))
            
            line2, = ax[1].plot(self.z, np.abs(H_history[idx][0])**2, color='r')
            ax[1].set_title(f'Magnetic Field Intensity (Frequency {self.central_frequencies[idx]} Hz)')
            ax[1].set_xlabel('Position (m)')
            ax[1].set_ylabel('|H|^2 (A^2/m^2)')
            ax[1].set_ylim(0, np.max(np.abs(H_history[idx])**2))
            
            
            
            
            def update(frame):
                line1.set_ydata(np.abs(E_history[idx][frame])**2)
                line2.set_ydata(np.abs(H_history[idx][frame])**2)
                #update the legends to show time
                
                ax[0].legend([f'Time: {self.t[frame]*1e12:.2f} ps'])
                ax[1].legend([f'Time: {self.t[frame]*1e12:.2f} ps'])
                
                
                return line1, line2
            
            ani = animation.FuncAnimation(fig, update, frames=self.Nt, blit=True, interval=50)
            plt.tight_layout()
            #save the animation as gif file
            ani.save(f"{file_name}_freq{self.central_frequencies[idx]}.gif", writer='pillow')
#example usage 

if __name__ == "__main__":
    #to simulate a Gaussian pulse propogating in a two-level medium
    #the two-level medium do not offer any polarization, i.e., mu = 0
    # Define simulation parameters
    dz = 1e-6  # Spatial step size (m)
    dt = dz / const.c   # Time step size (s), satisfying the CFL condition
    t_start = 0  # Start time (s)
    t_end = 1e-12  # End time (s)
    z_start = 0  # Start position (m)
    z_end = 1e-3  # End position (m)
    
    
    
    z = np.arange(z_start, z_end, dz)
    t = np.arange(t_start, t_end, dt)
    Nt = len(t)
    Nz = len(z)
    print(f"Number of spatial points: {Nz}")
    #The field initial condition: a linear TEM wave packet
    
    #wave packet: Gaussian envelope modulated by a carrier wave
    #central frequency: 3e14 Hz (wavelength: 1 um)
    f0 = 3e11  # Central frequency (Hz)
    omega0 = 2 * np.pi * f0  # Angular frequency (rad/s)
    k0 = omega0 / const.c  # Wave number (rad/m)
    #the carrier wave exp(1j*(k0*z - omega0*t)) is considered in the time evolution, here we only set the envelope as the initial condition
    
    #check whether dt*omega is much less than 1
    if dt * omega0 >= 0.1:
        raise ValueError("Time step dt is too large for the central frequency. Please reduce dt.")
    
    
    
    ##initial wave packet parameters
    pulse_duration = 100e-15  # Pulse duration (s)
    pulse_center = z_end / 4  # Center position of the pulse (m)
    
    ##Initial electric field at time t=0
    ##grid is z
    E0 = np.zeros(Nz, dtype=complex)
    for i in range(Nz):
        envelope = np.exp(-((z[i] - pulse_center) ** 2) / (2 * (const.c * pulse_duration / (2 * np.sqrt(2 * np.log(2)))) ** 2))
        E0[i] = envelope * 1e7  # Peak electric field amplitude (V/m)
    #Initial magnetic field at time t=dt/2
    ##grid is z+dz/2
    H0 = np.zeros(Nz, dtype=complex)
    for i in range(Nz):
        envelope = np.exp(-((z[i] - pulse_center) ** 2) / (2 * (const.c * pulse_duration / (2 * np.sqrt(2 * np.log(2)))) ** 2))
        H0[i] = envelope *1e7 * np.sqrt(const.epsilon_0/const.mu_0)
        

    
    
    
    
    #Initial density matrix elements: all atoms are in the ground state
    rho0 = []
    for i in range(Nz):
        rho_i = np.array([[1, 0],
                          [0, 0]], dtype=complex)  # All atoms in ground state
        rho0.append(rho_i)
    
    #Dipole moment matrix elements: zero for non-polarizable medium
    mu = []
    for i in range(Nz):
        mu_i = np.array([[0, 0],
                         [0, 0]], dtype=complex)  # No dipole moment
        mu.append(mu_i)
    #Free Hamiltonian of the two-level system
    H_free = np.array([[0, 0],
                          [0, 1.6e-19]], dtype=complex)  # Energy difference of 1 eV
    
    ##run the simulation, use OneDimensionMaxwellBloch_MultipleFrequencies class
    ##only one frequency component
    central_frequencies = [f0]
    model = OneDimensionMaxwellBloch_MultipleFrequencies(
        E0s=[E0],
        H0s=[H0],
        rho0=rho0,
        mus=[mu[0]],  # all mus are the same in this example
        central_frequencies=central_frequencies,
        H_free=H_free,
        dz=dz,
        dt=dt,
        t_start=t_start,
        t_end=t_end,
        z_start=z_start,
        z_end=z_end,
        boundary_conditions=('periodic', 'periodic')
    )
    
    #run the simulation
    E_history, H_history, rho_history = model.run_simulation()
    #animate the fields
    model.animate_fields(E_history, H_history, file_name="gaussian_pulse_propagation")
        
    
    
    #do another simulation with full wavepacket (not envelope)
    model_full = OneDimensionMaxwellBloch(
        E0=E0 * np.cos(k0*z),  # Full initial electric field
        H0=H0 * np.cos(k0*z),  # Full initial magnetic field
        rho0=rho0,
        mu=mu,
        H_free=H_free,
        dz=dz,
        dt=dt,
        t_start=t_start,
        t_end=t_end,
        z_start=z_start,
        z_end=z_end,
        boundary_conditions=('periodic', 'periodic')
    )
    
    model_full.animate_vacuum_fields(*model_full.run_simulation_vacuum()[:2], file_name="gaussian_pulse_propagation_full_wave")