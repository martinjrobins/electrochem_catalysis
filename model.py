import pybamm
import numpy as np
import matplotlib.pylab as plt
import pints


class CatalyticModel(pints.ForwardModel):
    def __init__(self, param=None):
        # Set fixed parameters here
        if param is None:
            param = pybamm.ParameterValues({
                "Far-field concentration of S(soln) [mol cm-3]": 1e-6,
                "Diffusion Constant [cm2 s-1]": 7.2e-6,
                "Faraday Constant [C mol-1]": 96485.3328959,
                "Gas constant [J K-1 mol-1]": 8.314459848,
                "Electrode Area [cm2]": 0.07,
                "Temperature [K]": 273.0,
                "Voltage frequency [rad s-1]": 9.0152,
                "Voltage start [V]": 0.2,
                "Voltage reverse [V]": -0.2,
                "Voltage amplitude [V]": 0.05,
                "Scan Rate [V s-1]": 0.05,
                "Electrode Coverage [mol cm2]": 6.5e-12,
            })

        # Create dimensional fixed parameters
        D = pybamm.Parameter("Diffusion Constant [cm2 s-1]")
        F = pybamm.Parameter("Faraday Constant [C mol-1]")
        R = pybamm.Parameter("Gas constant [J K-1 mol-1]")
        a = pybamm.Parameter("Electrode Area [cm2]")
        T = pybamm.Parameter("Temperature [K]")
        omega_d = pybamm.Parameter("Voltage frequency [rad s-1]")
        E_start_d = pybamm.Parameter("Voltage start [V]")
        E_reverse_d = pybamm.Parameter("Voltage reverse [V]")
        deltaE_d = pybamm.Parameter("Voltage amplitude [V]")
        v = pybamm.Parameter("Scan Rate [V s-1]")
        Gamma = pybamm.Parameter("Electrode Coverage [mol cm2]")

        # Create dimensional input parameters
        E0_d = pybamm.InputParameter("Reversible Potential [V]")
        k0_d = pybamm.InputParameter("Redox Rate [s-1]")
        kcat_d = pybamm.InputParameter("Catalytic Rate [cm3 mol-l s-1]")
        alpha = pybamm.InputParameter("Symmetry factor [non-dim]")
        Cdl_d = pybamm.InputParameter("Capacitance [F]")
        Ru_d = pybamm.InputParameter("Uncompensated Resistance [Ohm]")

        # Create scaling factors for non-dimensionalisation
        E_0 = R * T / F
        T_0 = E_0 / v
        I_0 = F * a * Gamma / T
        L_0 = pybamm.sqrt(D * T_0)

        # Non-dimensionalise parameters
        E0 = E0_d / E_0
        k0 = k0_d * T_0
        kcat = kcat_d * Gamma * L_0 / D
        Cdl = Cdl_d * a * E_0 / (I_0 * T_0)
        Ru = Ru_d * I_0 / E_0

        omega = 2 * np.pi * omega_d * T_0
        E_start = E_start_d / E_0
        E_reverse = E_reverse_d / E_0
        t_reverse = E_start - E_reverse
        deltaE = deltaE_d / E_0

        # Input voltage protocol
        Edc_forward = -pybamm.t
        Edc_backwards = pybamm.t - 2*t_reverse
        Eapp = E_start + \
            (pybamm.t <= t_reverse) * Edc_forward + \
            (pybamm.t > t_reverse) * Edc_backwards + \
            deltaE * pybamm.sin(omega * pybamm.t)

        # create PyBaMM model object
        model = pybamm.BaseModel()

        # Create state variables for model
        theta = pybamm.Variable("O(surf) [non-dim]")
        c = pybamm.Variable("S(soln) [non-dim]", domain="solution")
        i = pybamm.Variable("Current [non-dim]")

        # Effective potential
        Eeff = Eapp - i * Ru

        # Faridaic current (Butler Volmer)
        i_f = k0 * ((1 - theta) * pybamm.exp((1-alpha) * (Eeff - E0))
                    - theta * pybamm.exp(-alpha * (Eeff - E0))
                    )

        c_at_electrode = pybamm.BoundaryValue(c, "left")

        # Catalytic current
        i_cat = kcat * c_at_electrode * (1 - theta)

        # ODE equations
        model.rhs = {
            theta: i_f,
            i: 1/(Cdl * Ru) * (i_f + Cdl * Eapp.diff(pybamm.t) - i - i_cat),
            c: pybamm.div(pybamm.grad(c)),
        }

        # algebraic equations (none)
        model.algebraic = {
        }

        # Boundary and initial conditions
        model.boundary_conditions = {
            c: {
                "right": (pybamm.Scalar(1), "Dirichlet"),
                "left": (i_cat, "Neumann"),
            }
        }

        model.initial_conditions = {
            theta: pybamm.Scalar(1),
            i: pybamm.Scalar(0),
            c: pybamm.Scalar(1),
        }

        # set spatial variables and solution domain geometry
        x = pybamm.SpatialVariable('x', domain="solution")
        model.default_geometry = pybamm.Geometry({
            "solution": {
                "primary":
                    {x: {
                        "min": pybamm.Scalar(0),
                        "max": pybamm.Scalar(20)
                    }}
            }
        })
        model.default_var_pts = {
            x: 100
        }

        # Using Finite Volume discretisation on an expanding 1D grid
        model.default_submesh_types = {
            "solution": pybamm.MeshGenerator(
                pybamm.Exponential1DSubMesh, {'side': 'left'}
            )
        }
        model.default_spatial_methods = {
            "solution": pybamm.FiniteVolume()
        }

        # model variables
        model.variables = {
            "Current [non-dim]": i,
            "O(surf) [non-dim]": theta,
            "S(soln) at electrode [non-dim]": c_at_electrode,
            "Applied Voltage [non-dim]": Eapp,
        }

        # --------------------------------

        # Set model parameters
        param.process_model(model)
        geometry = model.default_geometry
        param.process_geometry(geometry)

        # Create mesh and discretise model
        mesh = pybamm.Mesh(
            geometry, model.default_submesh_types, model.default_var_pts)
        disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
        disc.process_model(model)

        # Create solver
        solver = pybamm.CasadiSolver(mode="fast")

        # Store discretised model and solver
        self._model = model
        self._solver = solver
        self._omega_d = param.process_symbol(omega_d).evaluate()
        self._I_0 = param.process_symbol(I_0).evaluate()
        self._T_0 = param.process_symbol(T_0).evaluate()

    def simulate(self, parameters, times):
        input_parameters = {
            "Redox Rate [s-1]": parameters[0],
            "Catalytic Rate [cm3 mol-l s-1]": parameters[1],
            "Reversible Potential [V]": parameters[2],
            "Symmetry factor [non-dim]": parameters[3],
            "Uncompensated Resistance [Ohm]": parameters[4],
            "Capacitance [F]": parameters[5],
        }

        try:
            solution = self._solver.solve(
                self._model, times, inputs=input_parameters)
        except pybamm.SolverError:
            solution = np.zeros_like(times)
        return (
            solution["Current [non-dim]"](times),
            solution["O(surf) [non-dim]"](times),
            solution["S(soln) at electrode [non-dim]"](times),
            solution["Applied Voltage [non-dim]"](times),
        )

    def n_parameters(self):
        return 6


if __name__ == '__main__':
    model = CatalyticModel()
    x = [1.0, 100.0, 0.0, 0.5, 8.0, 20.0e-12]

    n = 2000
    t_eval = np.linspace(0, 35, n)
    current, theta, s_soln, Eapp = model.simulate(x, t_eval)
    plt.figure()
    plt.plot(t_eval, current)
    plt.ylabel("current [non-dim]")
    plt.xlabel("time [non-dim]")

    plt.figure()
    plt.plot(t_eval, s_soln)
    plt.ylabel("S(soln) at electrode [non-dim]")
    plt.xlabel("time [non-dim]")

    plt.figure()
    plt.plot(t_eval, theta)
    plt.ylabel("O(surf) [non-dim]")
    plt.xlabel("time [non-dim]")

    plt.figure()
    plt.plot(t_eval, Eapp)
    plt.ylabel("Applied Voltage [non-dim]")
    plt.xlabel("time [non-dim]")

    plt.show()
