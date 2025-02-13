import numpy as np
import matplotlib.pyplot as plt

# Function to define the differential equation dx/dt = t - x^2
def diff_func(x, t):
    return t - x**2

# Euler method to solve dx/dt = f(x, t)
def euler_method(x0, tmax, h):
    t_values = np.arange(0, tmax + h, h)
    x_values = np.zeros(len(t_values))
    x_values[0] = x0
    for i in range(1, len(t_values)):
        t = t_values[i-1]
        x = x_values[i-1]
        x_values[i] = x + h * diff_func(x, t)  # Euler step: x = x + h*f(x, t)
    return t_values, x_values

# Plotting function for different initial values and parameters
def plot_graphs(initial_conditions, tmax_list, h):
    plt.figure(figsize=(10, 6))

    # Plot for each initial condition x0
    for x0 in initial_conditions:
        for tmax in tmax_list:
            t_values, x_values = euler_method(x0, tmax, h)
            plt.plot(t_values, x_values, label=f"x0={x0}, tmax={tmax}, h={h}")

    plt.title("Euler Method for dx/dt = t - x^2")
    plt.xlabel("t")
    plt.ylabel("x(t)")
    plt.legend(loc="best")
    plt.show()

# Parameters
step_size = 0.05  # Step size h
longer_step_size = 0.01  # Smaller step size for additional experimentation
initial_conditions = [1]  # Initial values of x0
tmax_list = [9]  # Maximum run time for initial comparison
long_tmax_list = [50]  # Longer time periods for later experimentation

# Plot the graphs for x0 and tmax
plot_graphs(initial_conditions, tmax_list, step_size)

# Run for larger times and observe trends
"plot_graphs([1], long_tmax_list, step_size)"

# Run again with smaller step size h=0.01
"plot_graphs(initial_conditions, tmax_list, longer_step_size)"
