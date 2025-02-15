import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from google.colab import files

# Define the driven pendulum function
def driven_pendulum(t, y, b, omega0, A, omega):
    x, v = y
    dxdt = v
    dvdt = -b * v - (omega0 ** 2) * x - A * np.sin(omega * t)
    return np.array([dxdt, dvdt])

def loop_through(omega, b, A, tf, y0):
    """
    Plots the response for different driving frequencies.
    """
    # Loop through different driving frequencies (100%, 90%, 50% of omega)
    for omegad in (omega, 0.9 * omega, 0.5 * omega):
        # Define the anonymous function, including the changing omegad
        lfun = lambda t, y: driven_pendulum(t, y, b, omega0, A, omegad)

        # Call the solver for this definition of lfun
        t_eval = np.linspace(0, tf, 1000)  # Define time points for evaluation
        result = integrate.solve_ivp(
            fun=lfun,
            t_span=(0, tf),
            y0=y0,
            method="RK45",
            t_eval=t_eval
        )

        # Store result of this run in variables t, x, v
        t = result.t
        x, v = result.y

        # Plot the result x(t) for this run, label it with omegad as well
        plt.plot(t, x, label='$x(t): \omega_d = {:.2f}$'.format(omegad))

    # Finish plotting
    plt.legend()  # Make the plot labels visible
    plt.title('Driven Harmonic Oscillator')
    plt.xlabel('Time (s)')
    plt.ylabel('Displacement x(t)')
    filename = f'driven_harmonic_oscillator2.png'
    plt.savefig(filename, dpi=300)
    print(f"Saved plot as {filename}") # Text confirm
    files.download(filename)
    plt.show()

# Main function to run the code
if __name__ == '__main__':
    # Define parameters
    A = 1.5       # Amplitude of the driving force
    b = 0.05     # Damping coefficient
    omega0 = 1  # Natural frequency
    omega = 1.2 # Driving frequency
    tf = 50     # Final time for integration
    y0 = [0, 0] # Initial conditions: [position, velocity]

    # Call the loop_through function
    loop_through(omega, b, A, tf, y0)
