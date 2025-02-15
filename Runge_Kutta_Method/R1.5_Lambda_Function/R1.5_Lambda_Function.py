import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from google.colab import files

# Damped pendulum function definition
def damped_pendulum(t, y, b, omega0):
    x, v = y
    dxdt = v
    dvdt = -b * v - (omega0**2) * x
    return np.array([dxdt, dvdt])

def main():
    # Define initial parameters
    b = 0.1      # Damping coefficient
    omega0 = 1   # Resonant frequency
    x0 = 0       # Initial position
    v0 = 1       # Initial velocity
    y0 = (x0, v0)  # Initial state
    t0 = 0       # Initial time
    tf = 25      # Final time

    # Create an array of time points
    n = 1001  # Number of points for evaluation
    t = np.linspace(t0, tf, n)

    # Define the lambda function for the ODE
    lfun = lambda t, y: damped_pendulum(t, y, b, omega0)

    # Call the solver with the lambda function
    result = integrate.solve_ivp(fun=lfun,
                                 t_span=(t0, tf),
                                 y0=y0,
                                 method="RK45",
                                 t_eval=t)

    # Read the solution and time from the result
    x, v = result.y
    t = result.t

    # Plot position and velocity as a function of time
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.plot(t, x, label='Position $x(t)$', color='b')
    plt.plot(t, v, label='Velocity $v(t)$', color='r')
    plt.title('Damped Harmonic Oscillator: Time Dependence')
    plt.xlabel('Time $t$')
    plt.ylabel('Position and Velocity')
    plt.legend()


    # Phase space plot
    plt.subplot(2, 1, 2)
    plt.plot(x, v, 'k')
    plt.title('Phase Space Plot')
    plt.xlabel('Position $x$')
    plt.ylabel('Velocity $v$')
    plt.axis('equal')


    # Show plots
    plt.tight_layout()
    filename = f'damped_oscillator_lambda.png'
    plt.savefig(filename, dpi=300)
    print(f"Saved plot as {filename}") # Text confirm
    files.download(filename)
    plt.show()


if __name__ == '__main__':
    main()
