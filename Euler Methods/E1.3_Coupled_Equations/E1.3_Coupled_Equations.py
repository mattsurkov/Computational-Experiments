import numpy as np
import matplotlib.pyplot as plt
from google.colab import files  # Import here

# Define the differential equation functions
def dxdt(y):
    """Returns the derivative dx/dt = y."""
    return y

def dydt(x):
    """Returns the derivative dy/dt = -x."""
    return -x

def euler_method_coupled(x0, y0, h, tmax):
    """Solves the coupled differential equations using Euler's method."""
    t_values = np.arange(0, tmax + h, h)  # Time values
    x_values = np.zeros(len(t_values))     # Array to store x values
    y_values = np.zeros(len(t_values))     # Array to store y values
    x_values[0] = x0                       # Initial condition for x
    y_values[0] = y0                       # Initial condition for y

    # Apply Euler's method
    for n in range(1, len(t_values)):
        x = x_values[n - 1]
        y = y_values[n - 1]

        dx = dxdt(y)                       # Compute dx/dt
        dy = dydt(x)                       # Compute dy/dt

        # Update x and y using Euler's method
        x_values[n] = x + h * dx
        y_values[n] = y + h * dy

    return t_values, x_values, y_values

def exact_solution(t_values):
    """Returns the exact solution for comparison."""
    x_exact = np.sin(t_values)
    y_exact = np.cos(t_values)
    return x_exact, y_exact

def plot_solutions(h_values, tmax):
    """Plots numerical and exact solutions for different time steps h."""
    plt.figure(figsize=(12, 8))

    # Plot for each h value
    for h in h_values:
        t_values, x_values, y_values = euler_method_coupled(0, 1, h, tmax)
        plt.plot(t_values, x_values, label=f'Euler h = {h}')

    # Exact solution
    t_values_exact = np.linspace(0, tmax, 1000)
    x_exact, y_exact = exact_solution(t_values_exact)
    plt.plot(t_values_exact, x_exact, 'k--', label='Exact Solution (sin(t))', lw=2)

    # Graph formatting
    plt.title('Numerical vs Exact Solutions of Coupled Equations for Various Time Steps h')
    plt.xlabel('Time (t)')
    plt.ylabel('x(t)')
    plt.legend()
    plt.xlim(0, tmax)
    plt.axhline()
    plt.ylim(-2, 2)
    filename = f'coupled_equations_solutions.png'
    plt.savefig(filename, dpi=300)
    print(f"Saved plot as {filename}") # Text confirm
    files.download(filename)
    plt.show()

def plot_error(h_values, tmax):
    """Plots the error between the numerical and exact solutions for different time steps h."""
    plt.figure(figsize=(12, 8))

    # Loop over different step sizes h
    for h in h_values:
        t_values, x_values, y_values = euler_method_coupled(0, 1, h, tmax)
        x_exact, y_exact = exact_solution(t_values)

        # Calculate the error between numerical and exact solution for x(t)
        error = x_values - x_exact

        # Plot the error
        plt.plot(t_values, error, label=f'h = {h}')

    # Graph formatting
    plt.title('Error between Numerical and Exact Solutions for Various h')
    plt.xlabel('Time (t)')
    plt.ylabel('Error x(t)')
    plt.axhline(color='black')  # Add a horizontal line at y = 0
    plt.legend()
    plt.xlim(0, tmax)
    filename = f'coupled_equations_errors.png'
    plt.savefig(filename, dpi=300)
    print(f"Saved plot as {filename}") # Text confirm
    files.download(filename)
    plt.show()
    
# Parameters
tmax = 5 * 2 * np.pi  # Time covering at least 5 cycles (5 * 2π)
h_values = [0.03, 0.01, 0.005]  # Different step sizes to compare

# Plot solutions
plot_solutions(h_values, tmax)
plot_error(h_values, tmax)
