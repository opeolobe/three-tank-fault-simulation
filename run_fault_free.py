#*****************************************************************
# Date:         February, 2026
# Author:       Opeoluwa Adebayo, ooadebayo@mun.ca
# Institution:  Memorial University of Newfoundland, St. John's, NL
#*****************************************************************



"""
This script models the nominal dynamics of the three tank system and the response of the system to different inputs. This allows us to analyze the behavior of the system under normal operating conditions and to establish a baseline for fault detection and diagnosis.
"""


###**** Necessary Imports ****###
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import cont2discrete, square

from dynamics import three_tank_model




###**** Discretized State Space Model ****###
states = ["h1", "h2", "h3"],
inputs = ["Q1", "Q2"]
outputs = ["h1", "h2", "h3"]

A, B, C, D = three_tank_model()

# Model dimensions
num_states = A.shape[0]
num_inputs = B.shape[1]
num_outputs = C.shape[0]

# Discretize the continuous state space model using zero order hold method
ts = 10                                                         # Sampling time in seconds
Ad, Bd, C, D, dt = cont2discrete(system=(A, B, C, D), dt=ts)

print("\nDiscretized State Space Model:")
print("A =", Ad, "\n")
print("B =", Bd, "\n")
print("C =", C, "\n")
print("D =", D, "\n")


###**** Simulate the System Responseto a Unit Step in Input Q1 ****###
N = 300                                                        # Number of time steps                                  
Ts = N * ts                                                    # Total simulation time
t = np.arange(0, Ts, ts)                                       # Time vector

# Input signal
Q1 = 1.0 * np.ones((1, N))                                     # Unit step input for Q1
Q2 = 0.0 * np.ones((1, N))                                     # No input for Q2
u = np.zeros((num_inputs, N))                                  # Input matrix (dimensions: 2 inputs by N time steps)
u[0, :] = Q1
u[1, :] = Q2

# Response to step change in Q1
y_sim_Q1 = np.zeros((num_outputs, N))                          # Output matrix (dimensions: 3 outputs by N time steps)
x = np.zeros((num_states, N+1))                                # States (dimensions: 3 states by N+1 time steps to include initial state)

for k in range(N):
    # Update the state
    x[:, k+1] = Ad @ x[:, k] + Bd @ u[:, k]
    # Compute the output
    y_sim_Q1[:, k] = C @ x[:, k] + D @ u[:, k]

# Plot the response to step change in Q1
fig, axes = plt.subplots(num_outputs, 1, figsize=(7, 5), sharex=True)

# h1 response
axes[0].plot(t, y_sim_Q1[0, :])
axes[0].set_ylabel(r'$h_1$ (cm)')
axes[0].set_title(r'$h_1$ vs t')
axes[0].grid(True, alpha=0.2)

# h2 response
axes[1].plot(t, y_sim_Q1[1, :])
axes[1].set_ylabel(r'$h_2$ (cm)')
axes[1].set_title(r'$h_2$ vs t')
axes[1].grid(True, alpha=0.2)

# h3 response
axes[2].plot(t, y_sim_Q1[2, :])
axes[2].set_ylabel(r'$h_3$ (cm)')
axes[2].set_title(r'$h_3$ vs t')
axes[2].grid(True, alpha=0.2)

axes[-1].set_xlabel('Time (s)')
fig.suptitle("Dynamic Response of the System to a Unit Step Change in Q1")
fig.tight_layout()


###**** Simulate the System Responseto a Unit Step in Input Q2 ****###
Q1 = 0.0 * np.ones((1, N))                                     # No input for Q1
Q2 = 1.0 * np.ones((1, N))                                     # Unit step input for Q2
u = np.zeros((num_inputs, N))                                  # Input matrix (dimensions: 2 inputs by N time steps)
u[0, :] = Q1
u[1, :] = Q2

# Response to step change in Q2
y_sim_Q2 = np.zeros((num_outputs, N))                          # Output matrix (dimensions: 3 outputs by N time steps)
x = np.zeros((num_states, N+1))                                # States (dimensions: 3 states by N+1 time steps to include initial state)

for k in range(N):
    # Update the state
    x[:, k+1] = Ad @ x[:, k] + Bd @ u[:, k]
    # Compute the output
    y_sim_Q2[:, k] = C @ x[:, k] + D @ u[:, k]

# Plot the response to step change in Q2
fig, axes = plt.subplots(num_outputs, 1, figsize=(7, 5), sharex=True)

# h1 response
axes[0].plot(t, y_sim_Q2[0, :])
axes[0].set_ylabel(r'$h_1$ (cm)')
axes[0].set_title(r'$h_1$ vs t')
axes[0].grid(True, alpha=0.2)

# h2 response
axes[1].plot(t, y_sim_Q2[1, :])
axes[1].set_ylabel(r'$h_2$ (cm)')
axes[1].set_title(r'$h_2$ vs t')
axes[1].grid(True, alpha=0.2)

# h3 response
axes[2].plot(t, y_sim_Q2[2, :])
axes[2].set_ylabel(r'$h_3$ (cm)')
axes[2].set_title(r'$h_3$ vs t')
axes[2].grid(True, alpha=0.2)

axes[-1].set_xlabel('Time (s)')
fig.suptitle("Dynamic Response of the System to a Unit Step Change in Q2")
fig.tight_layout()


###*** Dynamic Response of the System to a Square Input in Q1 and Q2 ***###
period = 4000                                                 # Period of the square wave in seconds
freq = 1 / period                                             # Frequency of the square wave in Hz

N = 1000                                                      # Number of time steps
Ts = N * ts                                                   # Total simulation time
t = np.arange(0, Ts, ts)                                      # Time vector

# Square wave (±0) for Q1 and Q2 
Q1 = 0.5*(1-square(2 * np.pi * freq * t))
Q2 = 0.5*(1-square(2 * np.pi * freq * t))

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


u = np.zeros((num_inputs, N))                                  # Input matrix (dimensions: 2 inputs by N time steps)
u[0, :] = Q1
u[1, :] = Q2

# Response to square wave input in Q1 and Q2
y_sim_square = np.zeros((num_outputs, N))                      # Output matrix (dimensions: 3 outputs by N time steps)
x = np.zeros((num_states, N+1))                                # States (dimensions: 3 states by N+1 time steps to include initial state)

for k in range(N):
    # Update the state
    x[:, k+1] = Ad @ x[:, k] + Bd @ u[:, k]
    # Compute the output
    y_sim_square[:, k] = C @ x[:, k] + D @ u[:, k]  


# Plot the response to square wave input in Q1 and Q2
fig, axes = plt.subplots(num_outputs, 1, figsize=(8, 6), sharex=True)   

# h1 response
axes[0].plot(t, y_sim_square[0, :])
axes[0].set_ylabel(r'$h_1$ (cm)')
axes[0].set_title(r'$h_1$ vs t')
axes[0].grid(True, alpha=0.2)

# h2 response
axes[1].plot(t, y_sim_square[1, :])
axes[1].set_ylabel(r'$h_2$ (cm)')
axes[1].set_title(r'$h_2$ vs t')
axes[1].grid(True, alpha=0.2)

# h3 response
axes[2].plot(t, y_sim_square[2, :])
axes[2].set_ylabel(r'$h_3$ (cm)')
axes[2].set_title(r'$h_3$ vs t')
axes[2].grid(True, alpha=0.2)

axes[-1].set_xlabel('Time (s)')
fig.suptitle("Dynamic Response of the System to Square Wave Inputs in Q1 and Q2")
fig.tight_layout()  

plt.show()

