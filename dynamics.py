""" 
Continuous state space model of the three tank system.
"""

###**** Necessary Imports ***###
import numpy as np



###**** Three Tank Model ***###
def three_tank_model():
    """ Continuous state space model of the three tank system."""
    A = np.array([
        [-0.0085, 0, 0.0085],
        [0, -0.0195, 0.0084],
        [0.0085, 0.0084, -0.0169]
        ])

    B = np.array([
        [0.0065, 0],
        [0, 0.0065],
        [0, 0]
        ])

    C = np.eye(3)                                   # 3 by 3 identity matrix  

    D = np.zeros((3, 2))                            # We have 3 outputs and 2 inputs, so D is a 3 by 2 zero matrix

    # Model uncertainty caused by linearization (Optional)
    delta_t = 1.3620 * np.eye(3)                   # For 3 states, we have a 3 by 3 identity matrix scaled by the uncertainty factor

    H = np.array([
        [-0.0085, 0, 0.0085],
        [0, -0.0195, 0.0084],
        [0.0085, 0.0084, -0.0169]
        ])

    delta_A = delta_t @ H

    # A += delta_A


    return A, B, C, D





def multiplicative_faults_matrix():
    """ Matrix to represent multiplicative faults in the states."""
    A1 = np.array([
    [-0.0214, 0, 0],
    [0, 0, 0],
    [0, 0, 0]
    ])

    A2 = np.array([
        [0, 0, 0],
        [0, -0.0371, 0],
        [0, 0, 0]
    ])

    A3 = np.array([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, -0.0262]
    ])


    A4 = np.array([
        [-0.0085, 0, 0.0085],
        [0, 0, 0],
        [0.0085, 0, -0.0085]
    ])


    A5 = np.array([
        [0, 0, 0],
        [0, -0.0111, 0],
        [0, 0, 0]
    ])


    A6 = np.array([
        [0, 0, 0],
        [0, -0.0084, 0.0084],
        [0, 0.0084, -0.0084]
    ])


    return [A1, A2, A3, A4, A5, A6]