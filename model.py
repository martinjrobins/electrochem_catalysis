import pybamm
import numpy as np
import matplotlib.pylab as plt
import pints
import Sollplotter_mod as mod
import scipy as sci
import scipy

from jax.config import config
config.update("jax_enable_x64", True)



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
                "Voltage start [V]": 0.4,
                "Voltage reverse [V]": -0.4,
                "Voltage amplitude [V]": 0.0, #0.05,
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
            theta: i_f + i_cat,
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
            i: Cdl * Eapp.diff(pybamm.t),
            c: pybamm.Scalar(1),
        }

        # set spatial variables and solution domain geometry
        x = pybamm.SpatialVariable('x', domain="solution")
        model.geometry = pybamm.Geometry({
            "solution": {
                "primary":
                    {x: {
                        "min": pybamm.Scalar(0),
                        "max": pybamm.Scalar(20)
                    }}
            }
        })
        model.var_pts = {
            x: 100
        }

        # Using Finite Volume discretisation on an expanding 1D grid
        model.submesh_types = {
            "solution": pybamm.MeshGenerator(
                pybamm.Exponential1DSubMesh, {'side': 'left'}
            )
        }
        model.spatial_methods = {
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
        geometry = model.geometry
        param.process_geometry(geometry)

        # Create mesh and discretise model
        mesh = pybamm.Mesh(
            geometry, model.submesh_types, model.var_pts)
        disc = pybamm.Discretisation(mesh, model.spatial_methods)
        disc.process_model(model)

        # Create solver
        #solver = pybamm.CasadiSolver(mode="fast", rtol=1e-6, atol=1e-8,
        #                             extra_options_setup={
        #                                 "linear_multistep_method": "adams",
        #                                 "print_stats": True,
        #                             })
                                     #extra_options_setup={"max_num_steps": 10})
                                     #extra_options_call={"verbose": True})
        solver = pybamm.ScipySolver(rtol=1e-6, atol=1e-6)
        #solver = pybamm.JaxSolver(rtol=1e-6, atol=1e-6)
        #model.convert_to_format = 'jax'
        #for eq in model.rhs.values():
        #    for node in eq.pre_order():
        #        if isinstance(node, pybamm.Matrix):
        #            print('found')
        #            if scipy.sparse.issparse(node.entries):
        #                node._entries = node.entries.toarray()
        #solver = pybamm.ScikitsOdeSolver(rtol=1e-6, atol=1e-8,
        #                                 extra_options={'mxsteps':1000})

        # Store discretised model and solver
        self._model = model
        self._param = param
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
            pybamm.set_logging_level('DEBUG')
            solution = self._solver.solve(
                self._model, times, inputs=input_parameters)
        except pybamm.SolverError as e:
            print(e)
            solution = np.zeros_like(times)
        return (
            solution["Current [non-dim]"](times),
            solution["O(surf) [non-dim]"](times),
            solution["S(soln) at electrode [non-dim]"](times),
            solution["Applied Voltage [non-dim]"](times),
            input_parameters
        )

    def n_parameters(self):
        return 6


if __name__ == '__main__':
    model = CatalyticModel()
    print(model._param)
    x = [1e6, 1e10, 0.0, 0.5, 8.0, 20.0e-12]

    n = 2000
    Estart = 0.4
    Eend = -0.4
    scanrate = 0.05
    res = 2**16 # Needs to be higgh enough or fft doesn't go to high enough frequency

    # constants
    Rg = 8.314459848  # Gas constant
    F = 96485.3328959  # faradays constants
    T = 273    # temperature

    # calculate time scale
    Tmax = abs(Estart - Eend)/scanrate
    Tdim = np.linspace(0,Tmax,res)
    TnonDim = (F*scanrate/(Rg*T))*Tdim

    if Estart < Eend:
        Etdc = Estart + scanrate*Tdim   # non-normalised scanrate
    else:
        Etdc = Estart - scanrate*Tdim

    current, theta, s_soln, Eapp, input_parameters = model.simulate(x, TnonDim)

    # Parameters (Use volts)
    E0 = 0
    Kcat1 = x[1]
    dE = 0.0 # 0.05 #Sine wave ampliture
    freq = 9.0152

    # odd stuff
    trun = 8 # DON"T INCREASE PAST THIS VALUE ITS PRETTY UNSTABLE

    # Nondimensionalising parameters
    omega = 2*np.pi*freq*Rg*T/(F*scanrate)
    # potentials
    dE = F/(Rg*T)*dE
    Etdcnondim = F/(Rg*T)*Etdc
    Eadjust = (F/(Rg*T))*(Etdc - E0)/2

    Kcat1 =((Rg*T)/(F*scanrate))*Kcat1

    # calculate IDCNorm
    IDCnorm = np.ones(res)*(-Kcat1/2)

    for N in range(0,trun+1):
        # case for when g is just tanh
        if N == 0:
            g1 = np.tanh(Eadjust)
        else:
            g1 = mod.Tanh_nthdev(2*N,Eadjust)
        g2 = mod.Tanh_nthdev(2*N+1,Eadjust)
        """NEED TO DOUBLE CHECK IF ITS 2 TIMES N FACTORIAL OR 2N factorial"""
        holder = ((dE/4)**(2*N))*(g2- 2*Kcat1*g1)/((2*sci.special.factorial(N))**2)
        IDCnorm  -= holder

    # Plot odd harmonics
    Ioddnorm= np.zeros(res)
    for N in range(0,trun+1):
        g2 = mod.Tanh_nthdev(2*N+1,Eadjust)*((dE/4)**(2*N+1))
        holder = 0
        for i in range(0,N+1):
            scalar = ((-1)^i)/(sci.special.factorial(N-i)*sci.special.factorial(N+i+1))
            AC = Kcat1*np.sin((2*i+1)*omega*TnonDim) + (2*i+1)*omega*np.cos((2*i+1)*omega*TnonDim)
            holder += scalar*AC
        Ioddnorm += g2*holder

    # Plot even harmonics
    Ievennorm= np.zeros(res)
    for N in range(0,trun+1):
        g2 = mod.Tanh_nthdev(2*N+2,Eadjust)*((dE/4)**(2*N+2))
        holder = 0
        for i in range(0,N+1):
            scalar = ((-1)^i)/(sci.special.factorial(N-i)*sci.special.factorial(N+i+2))
            AC = -Kcat1*np.cos((2*i+2)*omega*TnonDim) + (2*i+2)*omega*np.sin((2*i+2)*omega*TnonDim)
            holder += scalar*AC
        Ievennorm += g2*holder

    # extract non dim harmonics These are all in nondimensional currents
    bandwidth = np.array([0,4,0,4,0,4,0,4,0])
    Harmonicsodd = mod.harm_gen(Ioddnorm ,Tdim,[freq],bandwidth, 1)

    bandwidth = np.array([0,0,4,0,4,0,4,0,4])
    Harmonicseven = mod.harm_gen(Ievennorm ,Tdim,[freq],bandwidth, 1)
    #Harmonicseven = mod.harm_gen(Ievennorm,TnonDim,[omega],bandwidth, 1)


    print(input_parameters)
    plt.figure()
    plt.plot(TnonDim, current)
    plt.plot(TnonDim, IDCnorm+Ievennorm+Ioddnorm)
    plt.ylabel("current [non-dim]")
    plt.xlabel("time [non-dim]")
    plt.savefig("current_nondim.pdf")
    np.savetxt("current_nondim.dat", np.transpose(np.vstack((TnonDim, current))))

    plt.cla()
    plt.plot(Etdc, current)
    plt.ylabel("current [non-dim]")
    plt.xlabel("Edc [V]")
    plt.savefig("current_nondim_pybamm.pdf")

    plt.cla()
    plt.plot(Etdc, IDCnorm+Ievennorm+Ioddnorm)
    plt.ylabel("current [non-dim]")
    plt.xlabel("Edc [V]")
    plt.savefig("current_nondim_jie.pdf")


    plt.cla()
    plt.plot(TnonDim, s_soln)
    plt.ylabel("S(soln) at electrode [non-dim]")
    plt.xlabel("time [non-dim]")
    plt.savefig("s_soln_nondim.pdf")
    np.savetxt("s_soln_nondim.dat", np.transpose(np.vstack((TnonDim, s_soln))))
    print(s_soln)

    plt.cla()
    plt.plot(TnonDim, theta)
    plt.ylabel("O(surf) [non-dim]")
    plt.xlabel("time [non-dim]")
    plt.savefig("o_surf_nondim.pdf")
    np.savetxt("o_surf_nondim.dat", np.transpose(np.vstack((TnonDim, theta))))

    plt.cla()
    plt.plot(TnonDim, Eapp)
    plt.ylabel("Applied Voltage [non-dim]")
    plt.xlabel("time [non-dim]")
    plt.savefig("e_app_nondim.pdf")
    np.savetxt("e_app_nondim.dat", np.transpose(np.vstack((TnonDim, Eapp))))

