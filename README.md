# Three-Tank System Fault Simulation

## Overview
This project simulates a three-tank system using a state-space model to study **fault behavior in dynamic systems**.  
Both **additive** and **multiplicative** faults are considered.

The implementation follows standard fault modeling approaches as described in:

> Ding, S. (2008). *Model-Based Fault Diagnosis Techniques: : design schemes, algorithms, and tools*. Springer

## System Description
The system consists of three interconnected tanks with fluid levels:

- **h1**: Tank 1 level  
- **h2**: Tank 2 level  
- **h3**: Tank 3 level  

### Inputs
- **Q1**: Inflow to Tank 1  
- **Q2**: Inflow to Tank 2  

### Outputs
- Measured tank levels

---

## Model
The system is described in discrete-time:

$$
x_{k+1} = A x_k + B u_k
$$

$$
y_k = C x_k + D u_k
$$

where:
- $x_k$: tank levels  
- $u_k = [Q_1, Q_2]$: inputs  

---

## Fault Modeling

### Additive Faults

Additive faults are external signals:

$$
x_{k+1} = A x_k + B u_k + E f_k
$$

$$
y_k = C x_k + D u_k + F f_k
$$

These represent:
- sensor bias  
- actuator disturbances  

---

### Multiplicative Faults

Multiplicative faults modify system parameters:

$$
A_f = A + \Delta A
$$

$$
x_{k+1} = A_f x_k + B u_k
$$


## Augmented Input Representation
To allow simulation using standard tools (e.g., `lsim`, `dlsim`), the system is rewritten using an augmented input:

$$
\tilde{u}_k =
\begin{bmatrix}
u_k \\
f_k
\end{bmatrix}
$$

$$
x_{k+1} = A_f x_k + \begin{bmatrix} B & E \end{bmatrix} \tilde{u}_k
$$

$$
y_k = C x_k + \begin{bmatrix} D & F \end{bmatrix} \tilde{u}_k
$$

---


## File Descriptions

- **dynamics.py**
  - Contains the state-space model of the three-tank system
  - Defines matrices:
    - \(A, B, C, D\)
  - Represents the physical system dynamics
  - Includes **multiplicative fault matrices**

- **run_fault_free.py**
  - Simulates the system under normal (fault-free) conditions
  - Used as a baseline for comparison

- **run_faulty.py**
  - Simulates the system with faults
  - Includes:
    - Definition of fault vector \(f\)
    - Construction of augmented input
     - Multiplicative fault effects
    - Fault injection into the system

---

## How to Run

### Step 1: Install dependencies

```bash
pip install numpy scipy matplotlib
```

### Step 2: Run simulations

**Fault-free case:**
```bash
python run_fault_free.py
```

**Faulty case:**
```bash
python run_faulty.py
```

The scripts will generate plots showing the system response.