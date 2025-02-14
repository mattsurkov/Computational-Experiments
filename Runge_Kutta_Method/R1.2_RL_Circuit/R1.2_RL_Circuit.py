import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from google.colab import files

def differential_rl(t, i, v, r, l):
    """
    Calculates the rate of change of current in the RL circuit:
    dI/dt = (V - RI) / L
    """
    return (v - r * i) / l

def exact_solution_rl(v, r, l, t):
    """
    Exact solution for the RL circuit:
    I(t) = (V/R)(1 - exp(-Rt/L))
    """
    return (v / r) * (1 - np.exp(-r * t / l))

def solve_rl_circuit(v, r, l, i0, t0, tf, n):
    """
    Solves the RL circuit ODE numerically using solve_ivp.
    """
    t = np.linspace(t0, tf, n)
    result = integrate.solve_ivp(fun=lambda t, i: differential_rl(t, i, v, r, l),
                                 t_span=(t0, tf), y0=[i0], method='RK45', t_eval=t)
    return result.t, result.y[0]

def compare_solutions(v, r, l, i0, t0, tf, step_sizes):
    """
    Compares numerical and exact solutions for different time steps and plots the results.
    """
    for n in step_sizes:
        # Solve the ODE with current step size
        t_numeric, i_numeric = solve_rl_circuit(v, r, l, i0, t0, tf, n)

        # Exact solution
        t_exact = np.linspace(t0, tf, n)
        i_exact = exact_solution_rl(v, r, l, t_exact)

        # Plot the solutions
        plt.plot(t_numeric, i_numeric, label=f'Numerical (n={n})')
        plt.plot(t_exact, i_exact, 'k--', label=f'Exact (n={n})')

    # Add labels and title
    plt.xlabel('Time (s)')
    plt.ylabel('Current (I)')
    plt.title(f'RL Circuit with V={v}V, R={r}Ω, L={l}H')
    plt.legend()
    filename = f'RL_circuit_{v}, {r}, {l}.png'
    plt.savefig(filename, dpi=300)
    print(f"Saved plot as {filename}") # Text confirm
    files.download(filename)
    plt.show()


def main():
    """
    Main function to compare RL circuit solutions for different step sizes and parameters.
    """
    # Initial physical parameters
    V = 16     # Voltage in Volts
    R = 50    # Resistance in Ohms
    L = 10    # Inductance in Henrys
    I0 = 0     # Initial current in Amps
    t0 = 0     # Initial time in seconds
    tf = 2.5    # Final time in seconds

    # List of time step sizes (varying n)
    step_sizes = [4, 8, 50]  # Smaller step size means larger n

    # Compare solutions for the default RL parameters
    print("Comparison of solutions for different time steps:")
    compare_solutions(V, R, L, I0, t0, tf, step_sizes)

    # Modify physical parameters to test their effect on the result
    print("Effect of different physical parameters:")

    # Case 1: Change resistance
    R_new = 100  # Double the resistance
    compare_solutions(V, R_new, L, I0, t0, tf, step_sizes)

    # Case 2: Change inductance
    L_new = 50  # Halve the inductance
    compare_solutions(V, R, L_new, I0, t0, tf, step_sizes)

    # Case 3: Change voltage
    V_new = 5  # Halve the voltage
    compare_solutions(V_new, R, L, I0, t0, tf, step_sizes)

# Ensure main function runs
if __name__ == '__main__':
    main()

