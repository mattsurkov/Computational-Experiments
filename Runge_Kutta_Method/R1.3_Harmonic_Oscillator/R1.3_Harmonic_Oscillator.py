import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from pathlib import Path
from google.colab import files

def simple_pendulum(t, y):
   
   # Define the two derivatives
   # note that dydt is now an array that holds both of the derivatives
   # :param t: an array of times (floats) - not used in this template
   # :param y: a tuple containing the values x and v
   # :return: the derivative of x and v
    
    x, v = y  # extracts the x and v values from the tuple
    dydt = np.array([v, -x])  # generates an array with the rates of change: dxdt = v, dvdt = -x
    return dydt  # returns the array

def phase_space(x, v):
    
   # Generates a phase space plot for the Simple Harmonic Oscillator.
   # :param x: array of position values
   # :param v: array of velocity values

    plt.plot(x, v, 'k')  # Plots v against x in black ('k')
    plt.axis('equal')  # Sets the axes to equal sizes
    plt.xlabel(r"$x$")  # Labels the x-axis
    plt.ylabel(r"$v$")  # Labels the y-axis
    filename = f'pendulum_phase_space.png'
    plt.savefig(filename, dpi=300)
    print(f"Saved plot as {filename}") # Text confirm
    files.download(filename)
    plt.show()


def main():

    # define the initial parameters
    x0 = 0  # initial position
    v0 = 1  # initial velocity
    y0 = (x0, v0)  # initial state
    t0 = 0  # initial time

    # define the final time and the number of time steps
    tf = 10*np.pi  # final time
    n = 1001  # Number of points at which output will be evaluated
    # Note: this does not mean the integrator will take only n steps
    # Scipy will take more steps if required to control the error in the solution

    # creates an array of the time steps
    t = np.linspace(t0, tf, n)  # Points at which output will be evaluated

    # Calls the method integrate.solve_ivp()
    result = integrate.solve_ivp(fun=simple_pendulum,  # The function defining the derivative
                                 t_span=(t0, tf),  # Initial and final times
                                 y0=y0,  # Initial state
                                 method="RK45",  # Integration method
                                 t_eval=t)  # Time points for result to be defined at

    # Read the solution and time from the result array returned by Scipy
    x, v = result.y
    t = result.t

    # plot position ad velocity as a function of time.
    plt.plot(t, x, label=r"$x(t)$")
    plt.plot(t, v, label=r"$v(t)$")
    plt.title("Simple Harmonic Oscillator - Pendulum")
    plt.xlabel("Time")
    plt.ylabel("Position (x) and Velocity (v)")
    plt.legend(loc=1)

    filename = f'pendulum.png'
    plt.savefig(filename, dpi=300)
    print(f"Saved plot as {filename}") # Text confirm
    files.download(filename)
    plt.show()

    phase_space(x, v)

if __name__ == '__main__':
    main()
    
