import numpy as np
import matplotlib.pyplot as plt
from google.colab import files  # Import here

# Define the exact solutions for comparison
def exact_solution(t_values):
    """Returns the exact solutions for x and v at time t for a simple harmonic oscillator."""
    x_exact = np.sin(t_values)
    v_exact = np.cos(t_values)
    return x_exact, v_exact

# Define the modified Euler method (Heun's Method)
def modified_euler_method(x0, v0, h, tmax):
    """Solve the simple harmonic oscillator using the Modified Euler method."""
    t_values = np.arange(0, tmax + h, h)  # Time values
    x_values = np.zeros(len(t_values))     # Array to store position values (x)
    v_values = np.zeros(len(t_values))     # Array to store velocity values (v)

    # Initial conditions
    x_values[0] = x0
    v_values[0] = v0

    # Modified Euler Method loop
    for i in range(1, len(t_values)):
        # Initial estimates for position and velocity using Euler method
        xinit = x_values[i-1] + h * v_values[i-1]
        vinit = v_values[i-1] - h * x_values[i-1]

        # Corrected estimates for position and velocity
        x_values[i] = x_values[i-1] + 0.5 * h * (v_values[i-1] + vinit)
        v_values[i] = v_values[i-1] - 0.5 * h * (x_values[i-1] + xinit)

    return t_values, x_values, v_values

# Plot the solutions
def plot_solutions(h_values, tmax):
    """Plot numerical and exact solutions for different step sizes h."""
    plt.figure(figsize=(12, 8))

    for h in h_values:
        # Solve using the Modified Euler Method
        t_values, x_values, v_values = modified_euler_method(0, 1, h, tmax)

        # Plot the numerical solution for each h
        plt.plot(t_values, x_values, label=f'Modified Euler h = {h}')

    # Exact solution
    t_values_exact = np.linspace(0, tmax, 1000)
    x_exact, _ = exact_solution(t_values_exact)
    plt.plot(t_values_exact, x_exact, 'k--', label='Exact Solution (sin(t))', lw=2)

    # Graph formatting
    plt.title('Numerical vs Exact Solutions for Different Step Sizes using Modified Euler')
    plt.xlabel('Time (t)')
    plt.ylabel('Position x(t)')
    plt.legend()
    plt.xlim(0, tmax)
    plt.ylim(-1.5, 1.5)
    filename = f'modified_euler_solutions.png'
    plt.savefig(filename, dpi=300)
    print(f"Saved plot as {filename}") # Text confirm
    files.download(filename)
    plt.show()

# Plot the error
def plot_error(h_values, tmax):
    """Plot the error between the numerical and exact solution."""
    plt.figure(figsize=(12, 8))

    for h in h_values:
        # Solve using the Modified Euler Method
        t_values, x_values, v_values = modified_euler_method(0, 1, h, tmax)

        # Calculate the exact solution
        x_exact, _ = exact_solution(t_values)

        # Calculate the error
        error = x_values - x_exact

        # Plot the error for each h
        plt.plot(t_values, error, label=f'Error for h = {h}')

    # Graph formatting
    plt.title('Error between Numerical and Exact Solutions for Different Step Sizes using Modified Euler')
    plt.xlabel('Time (t)')
    plt.ylabel('Error in x(t)')
    plt.axhline(color='black')  # Add a horizontal line at y = 0
    plt.legend()
    plt.xlim(0, tmax)
    filename = f'modified_euler_errors.png'
    plt.savefig(filename, dpi=300)
    print(f"Saved plot as {filename}") # Text confirm
    files.download(filename)
    plt.show()

# Parameters
tmax = 5 * 2 * np.pi  # Maximum time covering 5 cycles (5 * 2π for sine wave)
h_values = [0.03, 0.015, 0.005, 0.001]  # Different step sizes to compare

# Plot the numerical solutions and compare with exact solution
plot_solutions(h_values, tmax)

# Plot the error between numerical and exact solutions
plot_error(h_values, tmax)

