#*****************************************************************
# Date:         February, 2026
# Author:       Opeoluwa Adebayo, ooadebayo@mun.ca
# Institution:  Memorial University of Newfoundland, St. John's, NL
#*****************************************************************



"""
This script models the response of the three-tank system to different faulty conditions. The faults considered include
multiplicative faults in the states, additive actuator faults, and sensor faults.

Modify line 98 to simulate different additive faults by specifying the fault index, magnitude and time of injection.
Modify line 114 to simulate different multiplicative faults by specifying the fault severity levels for each multiplicative fault.
"""


###**** Necessary Imports ****###
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import cont2discrete, square

from dynamics import three_tank_model, multiplicative_faults_matrix




###**** Discretized State Space Model ****###
states = ["h1", "h2", "h3"],
inputs = ["Q1", "Q2"]
outputs = ["h1", "h2", "h3"]

ts = 10                                                                     # Sampling time in seconds
N = 1000                                                                    # Number of time steps
Ts = N * ts                                                                 # Total simulation time
t = np.arange(0, Ts, ts)                                                    # Time vector

# Nominal (fault-free) model
A, B, C, D = three_tank_model()

# Model dimensions
num_states = A.shape[0]
num_inputs = B.shape[1]
num_outputs = C.shape[0]

# Discretize the  fault-free continuous state space model using zero order hold method
Ad_free, Bd_free, C_free, D_free, dt = cont2discrete(system=(A, B, C, D), dt=ts)




###*** Generated Square Input in Q1 and Q2 ***###
period = 4000                                                              # Period of the square wave in seconds
freq = 1 / period                                                          # Frequency of the square wave in Hz


# Square wave (±0) for Q1 and Q2 
Q1 = 0.5*(1-square(2 * np.pi * freq * t))
Q2 = 0.5*(1-square(2 * np.pi * freq * t))

# Nominal inputs (without faults)
u_free = np.vstack((Q1, Q2))

# Plot of the generated signals
fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)

# plot of generated Q1
axes[0].plot(t, Q1)
axes[0].set_ylabel("Q1")
axes[0].grid(True, alpha=0.2)

# plot of generated Q2
axes[1].plot(t, Q2)
axes[1].set_ylabel("Q2")
axes[1].grid(True, alpha=0.2)

axes[-1].set_xlabel('Time (s)')
fig.suptitle("Generated  Square Wave Input Signals")
fig.tight_layout()



###***** Fault Modeling and Simulation *****###
# Additive fault signals
num_sensor_faults = 3
num_actuator_faults = 2

f = np.zeros((num_sensor_faults + num_actuator_faults, N))

# Fault indexing  (f1, f2 and f3 are sensor faults in h1, h2 and h3 respectively, 
# while f4 and f5 are actuator faults in pump 1 and pump 2 respectively)
IDX_H1 = 0                                                              # Index for fault in sensor measuring h1
IDX_H2 = 1                                                              # Index for fault in sensor measuring h2
IDX_H3 = 2                                                              # Index for fault in sensor measuring h3
IDX_P1 = 3                                                              # Index for fault in pump 1 
IDX_P2 = 4                                                              # Index for fault in pump 2

# Fault injection
f[IDX_H2, t >= 0] = 0.2                                                # Specify the specific fault index, magnitude and time of injection



# Fault Matrix (to represent the additive faults in the system)
# Sensor faults: Ef = 0, Ff= I
# Actuator faults: Ef = B, Ff = D
Ef = np.hstack((np.zeros((num_states, num_sensor_faults)), B))        # Ef = [0, B] where 0 is a 3 by 3 zero matrix and B is the 3 by 2 input matrix
Ff = np.hstack((np.eye(num_sensor_faults), D))                        # Ff = [I, D] where I is a 3 by 3 identity matrix and D is the 3 by 2   

# Stack inputs and faults together for simulation
u_faulty = np.vstack((u_free, f))


# Multiplicative faults in the states
theta_Ai = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]                           # Fault severity levels for multiplicative faults (Modify as needed)
Ai = multiplicative_faults_matrix()                                 # Get the multiplicative faults matrix

# delta_A = SUM of theta_Ai * Ai for i=1:6
dA_F = np.zeros((3, 3))
for idx in range(len(theta_Ai)):
    dA_F += theta_Ai[idx]* Ai[idx]


# State Space Model with Faults
A_line = A + dA_F                                                  # To account for multiplicative faults                    
B_line = np.hstack((B, Ef))                                        # To account for additive faults
C_line = C
D_line = np.hstack((D, Ff))                                         # To account for additive faults  

# Discretize the faulty model using zero order hold method
Ad_faulty, Bd_faulty, C_faulty, D_faulty, dt = cont2discrete(system=(A_line, B_line, C_line, D_line), dt=ts)   


# System response (Fault-free)
x_free = np.zeros((num_states, N+1))
y_free = np.zeros((num_states, N))

# System response (Faulty)
x_faulty = np.zeros((num_states, N+1))
y_faulty = np.zeros((num_states, N))


# Simulate the system response for both fault-free and faulty models
for k in range(N):
    # Fault free model
    y_free[:,k] = C_free @ x_free[:,k] + D_free @ u_free[:, k]
    x_free[:, k+1] = Ad_free @ x_free[:, k] + Bd_free @ u_free[:, k]

    # Faulty model
    y_faulty[:,k] = C_faulty @ x_faulty[:,k] + D_faulty @ u_faulty[:, k]
    x_faulty[:, k+1] = Ad_faulty @ x_faulty[:, k] + Bd_faulty @ u_faulty[:, k]


# Plot of the dynamic response
fig, axes = plt.subplots(3, 1, figsize=(9, 6), sharex=True)

# h1 response
axes[0].plot(t, y_free[0, :], color='blue', label="Nominal")
axes[0].plot(t, y_faulty[0, :], color="red", linestyle='--', label="Faulty")
axes[0].set_ylabel(r'$h_1$ (cm)')
axes[0].set_title(r'$h_1$ vs t')
axes[0].grid(True, alpha=0.2)
axes[0].legend()

# h2 response
axes[1].plot(t, y_free[1, :], color='blue', label="Nominal")
axes[1].plot(t, y_faulty[1, :], color="red", linestyle='--', label="Faulty")
axes[1].set_ylabel(r'$h_2$ (cm)', weight='demibold')
axes[1].set_title(r'$h_2$ vs t')
axes[1].grid(True, alpha=0.2)
axes[1].legend()

# h3 response
axes[2].plot(t, y_free[2, :], color='blue', label="Nominal")
axes[2].plot(t, y_faulty[2, :], color="red", linestyle='--', label="Faulty")
axes[2].set_ylabel(r'$h_3$ (cm)', weight='demibold')
axes[2].set_title(r'$h_3$ vs t')
axes[2].grid(True, alpha=0.2)
axes[2].legend()

axes[-1].set_xlabel('Time (s)')
fig.suptitle("Plot of the System Behaviour in Nominal and Faulty Condition")
fig.tight_layout()

plt.show() 
