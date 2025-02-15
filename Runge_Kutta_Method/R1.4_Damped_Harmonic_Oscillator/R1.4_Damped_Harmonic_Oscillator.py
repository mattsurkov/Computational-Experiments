import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import os
from google.colab import files


def damped_pendulum(t, y, b, omega0=1):
    """
    Define the derivatives for the damped harmonic oscillator.
    :param t: Time variable (not used in this case)
    :param y: Tuple containing the position (x) and velocity (v)
    :param b: Damping constant
    :param omega0: Natural frequency (default = 1)
    :return: Array of the derivatives [dx/dt, dv/dt]
    """
    x, v = y
    dxdt = v
    dvdt = -b * v - (omega0 ** 2) * x
    dydt = np.array([dxdt, dvdt])
    return dydt

def plot_results(t, x, v, b_label, axs, row):
    """
    Plots the time dependency of position and velocity, and phase space, for a given damping case.
    :param t: Array of time values
    :param x: Array of position values
    :param v: Array of velocity values
    :param b_label: Label for the damping coefficient to display in the title
    :param axs: Axes array for subplots
    :param row: Which row (i.e., under-damped, critically-damped, or over-damped) to plot on
    """
    # Plot position and velocity as a function of time
    axs[row, 0].plot(t, x, label=r"$x(t)$", color='blue')
    axs[row, 0].plot(t, v, label=r"$v(t)$", color='orange')
    axs[row, 0].set_title(f"{b_label} Time Dependency")
    axs[row, 0].set_xlabel("Time (s)")
    axs[row, 0].set_ylabel("Value")
    axs[row, 0].legend(loc='best')

    # Plot phase space (position vs velocity)
    axs[row, 1].plot(x, v, 'k')
    axs[row, 1].set_title(f"{b_label} Phase Space Plot")
    axs[row, 1].set_xlabel(r"$x$")
    axs[row, 1].set_ylabel(r"$v$")
    axs[row, 1].axis('equal')  # Ensure equal scaling on both axes for square shape

def main():
    # Define initial parameters
    x0 = 0  # Initial position
    v0 = 1  # Initial velocity
    y0 = (x0, v0)  # Initial state as a tuple
    t0 = 0  # Initial time
    tf = 30  # Final time (increased to observe damping behavior)
    n = 1001  # Number of points at which output will be evaluated

    # Define natural frequency
    omega0 = 1  # Natural frequency

    # Define the damping constants for under-damped, critically-damped, and over-damped cases
    b_values = [(0.3, 'Under-damped, b : 0.3'), (2.0, 'Critically-damped, b: 2.0'), (3.0, 'Over-damped, b: 3.0')]

    # Create an array of time steps
    t = np.linspace(t0, tf, n)

    # Create subplots for time dependency and phase space
    fig, axs = plt.subplots(3, 2, figsize=(12, 10))  # Three rows for each damping case, two columns (time/phase)

    # Loop through the damping constants and simulate the system
    for i, (b, label) in enumerate(b_values):
        # Solve the damped oscillator using solve_ivp
        result = integrate.solve_ivp(fun=lambda t, y: damped_pendulum(t, y, b, omega0),
                                     t_span=(t0, tf),  # Initial and final times
                                     y0=y0,  # Initial state
                                     method="RK45",  # Integration method
                                     t_eval=t)  # Time points for result to be defined at

        # Read the solution (position and velocity) and time from the result array
        x, v = result.y  # Position and velocity
        t = result.t  # Time values

        # Plot results for each damping scenario
        plot_results(t, x, v, label, axs, i)

    plt.tight_layout()  # Adjust subplots to fit into the figure area
    filename = f'damped_pendulum_x_and_v.png'
    plt.savefig(filename, dpi=300)
    print(f"Saved plot as {filename}")  # Text confirm
    files.download(filename)
    plt.show()  # Display the plots

if __name__ == '__main__':
    main()

