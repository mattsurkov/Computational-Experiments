import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from google.colab import files


# Define the nonlinear derivative function with parameters a and b
def nonlinear1(t, y, a, b):
    """
    Calculates the derivative value for the differential equation given in experiment R1, with parameters a and b.
    :param t: a float for the time variable
    :param y: a float for the dependent variable
    :param a: parameter for the equation
    :param b: parameter for the equation
    :return dydt a float for the derivative of the differential equation:
    """

    # Calculate the derivative explicitly with parameters a and b
    dydt = -a * y**3 + b * np.sin(t)

    # Return the value
    return dydt

# Wrapper function to call the ODE solver
def solve_ode(a, b, t0, tf, n, y0):
    """
    Solves the ODE for given parameters a, b, initial conditions, and time steps.
    :param a: parameter a in the ODE
    :param b: parameter b in the ODE
    :param t0: initial time
    :param tf: final time
    :param n: number of time steps
    :param y0: initial condition
    :return: t (time points), y (solution)
    """
    t = np.linspace(t0, tf, n)  # Create time array

    # Define the lambda function to pass a and b to the ODE solver
    ode_func = lambda t, y: nonlinear1(t, y, a, b)

    # Solve the ODE
    result = integrate.solve_ivp(fun=ode_func,  # The function defining the derivative
                                 t_span=(t0, tf),  # Initial and final times
                                 y0=y0,  # Initial state
                                 method="RK45",  # Integration method
                                 t_eval=t)  # Time points for result to be reported

    return result.t, result.y[0]  # Return time and solution

def main():
    """
    Main function to solve the ODE for different values of a and b, and plot the results on the same graph.
    """
    # Define the initial variables
    t0 = 0  # Initial time
    tf = 20  # Final time
    n = 101  # Number of time steps
    y0 = np.array([0])  # Initial state at t = 0

    # Different values of a and b to iterate over
    a_values = [2, 0.5, 2]  # Example values for a
    b_values = [2, 3.5, 0.75]  # Example values for b

    # Colors for plotting
    colors = ['b', 'g', 'r']

    # Create the plot
    plt.figure()

    # Solve the ODE for each pair of (a, b)
    for i, (a, b) in enumerate(zip(a_values, b_values)):
        t, y = solve_ode(a, b, t0, tf, n, y0)  # Solve the ODE
        plt.plot(t, y, f'{colors[i]}.', label=f'a={a}, b={b}')  # Plot with different color and label

    # Add labels, title, and legend
    plt.xlabel('Time (t)')  # Label for x-axis
    plt.ylabel('y(t)')  # Label for y-axis
    plt.title('Solution of ODE for Different Values of a and b')  # Title of the plot
    plt.legend()  # Displaying the legend
    filename = f'ODE.png'
    plt.savefig(filename, dpi=300)
    print(f"Saved plot as {filename}") # Text confirm
    files.download(filename)
    plt.show()


# Ensure that the main function is called when the script is executed
if __name__ == '__main__':
    main()


