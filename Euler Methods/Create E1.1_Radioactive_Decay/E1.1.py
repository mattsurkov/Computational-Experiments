import numpy as np
import matplotlib.pyplot as plt

# Variables
dt = 0.4       # Time step
tmax = 10      # Maximum time
tau = 2.0      # Decay constant
t0 = 0         # Start time
N0 = 10        # Initial quantity

# Arrays to store calculated N and t values for Euler's method
t_values = [t0]
N_values = [N0]

# Arrays to store exact N values and error values
N_exact_values = [N0]
error_values = []

# Start loop for Euler's method
t = t0
N = N0
while abs(t - tmax) >= dt / 2:
    # Euler's method to calculate new N value
    N = N - (N / tau) * dt

    # Update time
    t = t + dt

    # Append to Euler's method arrays
    t_values.append(t)
    N_values.append(N)

    # Append exact solution values
    N_exact = N0 * np.exp(-t / tau)
    N_exact_values.append(N_exact)

    # Calculate and store the difference (error) between exact and Euler's method
    error = abs(N_exact - N)
    error_values.append(error)

# Plot both the calculated (Euler's method) and exact values
plt.plot(t_values, N_values, label="Euler's Method", linestyle='--', marker='o')
plt.plot(t_values, N_exact_values, label="Exact Values", linestyle='-', color='red')

# Labels and title
plt.xlabel('Time (t)')
plt.ylabel('Quantity (N)')
plt.title('Radioactive Decay: Euler vs Exact Values')
plt.legend()
plt.savefig("radioactive_decay_euler_vs_exact.png", dpi=300)

# Show plot of N(t)
plt.show()

# Plot error vs time
plt.plot(t_values[1:], error_values, label="Error (Exact - Euler's Method)", linestyle='-', color='blue')

# Labels and title for error plot
plt.xlabel('Time (t)')
plt.ylabel('Error (|Exact - Euler|)')
plt.title('Error between Exact Values and Euler\'s Method')
plt.legend()
plt.savefig("radioactive_decay_error.png", dpi=300)
# Show error plot
plt.show()

from google.colab import files
files.download("radioactive_decay_euler_vs_exact.png")
files.download("radioactive_decay_error.png")
