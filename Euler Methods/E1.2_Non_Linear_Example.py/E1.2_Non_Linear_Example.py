import numpy as np
import matplotlib.pyplot as plt
from google.colab import files  # Import here

def f(x, t):
    """Define the differential equation function."""
    return t - x**2

def euler_method(x0, h, tmax):
    """Implement the Euler method for solving the differential equation."""
    t_values = np.arange(0, tmax + h, h)  # Time values from 0 to tmax
    x_values = np.zeros(len(t_values))     # Array to store x values
    x_values[0] = x0                        # Set the initial condition

    # Apply Euler's method
    for n in range(1, len(t_values)):
        t = t_values[n - 1]
        x = x_values[n - 1]
        x_values[n] = x + h * f(x, t)      # Update x using the Euler method

    return t_values, x_values

def plot_results(initial_conditions, h, tmax):
    """Plot results for different initial conditions and the curve x = sqrt(t)."""
    plt.figure(figsize=(12, 8))
    t_values = np.arange(0, tmax + h, h)  # Time values for the curve x = sqrt(t)

    # Plot each initial condition's result
    for x0 in initial_conditions:
        t_values, x_values = euler_method(x0, h, tmax)
        plt.plot(t_values, x_values, label=f'x0 = {x0}')

    # Plot the theoretical line x = sqrt(t)
    plt.plot(t_values, np.sqrt(t_values), 'k--', label='x = sqrt(t)', linewidth=2)  # Theoretical curve

    # Plot details
    plt.title(f'Numerical Solution of dx/dt = t - x^2 (tmax = {tmax}), h = {h}')
    plt.xlabel('Time (t)')
    plt.ylabel('x(t)')
    plt.axhline(0, color='black', lw=0.5, ls='--')  # Add horizontal line at y=0
    plt.axvline(0, color='black', lw=0.5, ls='--')  # Add vertical line at x=0
    plt.legend()
    plt.xlim(0, tmax)
    plt.ylim(-3, 7)  # Adjust y limits to include more values of y

    # Save the plot
    filename = f'euler_solution_tmax_{tmax}.png'
    plt.savefig(filename, dpi=300)
    print(f"Saved plot as {filename}") # Text confirm
    files.download(filename)
    plt.show()


# Parameters
h = 0.5                                                    # Step size
tmax_values = [15, 30]                                # Different maximum times
initial_conditions = [4, 2, 1, 0, -0.7, -0.73, -1.5, -2]  # More initial values for x

# Plot and save results for different tmax values
for tmax in tmax_values:
    plot_results(initial_conditions, h, tmax)

# Parameters for second graph

h = 0.25                                                    # Step size
tmax_values = [15, 50]                                # Different maximum times
initial_conditions = [4, 2, 1, 0, -0.7, -0.73, -1.5, -2]  # More initial values for x

# Plot and save results for different tmax values
for tmax in tmax_values:
    plot_results(initial_conditions, h, tmax)
