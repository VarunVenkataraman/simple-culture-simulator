"""
SBML to Diffrax bridge module.

Converts SBML models to JAX-based ODE systems compatible with diffrax.
"""

import jax
import jax.numpy as jnp
import diffrax
from sbmltoodejax.parse import ParseSBMLFile
from sbmltoodejax.modulegeneration import GenerateModel
from pathlib import Path
import importlib.util
import sys

from continuous_logic import augmented_vector_field


def load_sbml_model(sbml_path):
    """
    Load and parse SBML model using sbmltoodejax.

    Generates a Python module with JAX functions and imports it.

    Args:
        sbml_path: Path to SBML XML file

    Returns:
        Generated model module
    """
    # Parse SBML file
    sbml_path = Path(sbml_path)
    modelData = ParseSBMLFile(str(sbml_path))

    # Generate Python code from parsed model
    output_dir = sbml_path.parent
    model_name = sbml_path.stem
    model_file = output_dir / f"{model_name}.py"

    GenerateModel(
        modelData,
        str(model_file),
        deltaT=0.1
    )

    # Import the generated module
    spec = importlib.util.spec_from_file_location(model_name, str(model_file))
    model_module = importlib.util.module_from_spec(spec)
    sys.modules[model_name] = model_module
    spec.loader.exec_module(model_module)

    return model_module


def get_sbml_vector_field(model_module):
    """
    Extract the vector field (ODE right-hand side) from SBML model.

    Args:
        model_module: Generated model module from sbmltoodejax

    Returns:
        Function (t, y, args) -> dy/dt and initial conditions
    """
    # Get model data
    y0 = model_module.y0
    w0 = model_module.w0
    c = model_module.c

    # Instantiate the rate function
    rate_func = model_module.RateofSpeciesChange()

    def sbml_vector_field(t, y, args):
        """
        Compute dy/dt from SBML model.

        Args:
            t: Time
            y: State vector (order from SBML)
            args: Dictionary of parameters

        Returns:
            dy/dt vector
        """
        # Call the rate function: __call__(self, y, t, w, c)
        dy = rate_func(y, t, w0, c)
        return dy

    return sbml_vector_field, y0


def create_simulator(sbml_path, t_end=100.0, dt=0.1, solver=None):
    """
    Create a diffrax-based simulator from an SBML model.

    Args:
        sbml_path: Path to SBML XML file
        t_end: End time for simulation
        dt: Output time step
        solver: Diffrax solver (default: Tsit5)

    Returns:
        Simulation function that takes (threshold, feed_rate) and returns loss
    """
    # Load SBML model
    model_module = load_sbml_model(sbml_path)

    # Get base vector field from SBML
    sbml_vf, sbml_y0 = get_sbml_vector_field(model_module)

    # Augment with continuous passaging logic
    vector_field = augmented_vector_field(sbml_vf)

    # Default solver
    if solver is None:
        solver = diffrax.Tsit5()

    def simulate(threshold, feed_rate, y0_override=None):
        """
        Run forward simulation with given parameters.

        Args:
            threshold: Passaging threshold (0.0 - 1.0)
            feed_rate: Nutrient feed rate
            y0_override: Initial state [S, D, N, Y] (optional)

        Returns:
            Final state [S_final, D_final, N_final, Y_final]
        """
        # Default initial conditions
        # Note: sbml_y0 order is [N, S, D] based on the generated model
        if y0_override is None:
            N0 = float(sbml_y0[0])  # Nutrients
            S0 = float(sbml_y0[1])  # Stem cells
            D0 = float(sbml_y0[2])  # Differentiated cells
            Y0 = 0.0                # Initial yield (zero)
            # Our augmented system uses [N, S, D, Y] order
            y0 = jnp.array([N0, S0, D0, Y0])
        else:
            y0 = y0_override

        # Package parameters
        args = {
            'threshold': threshold,
            'feed_rate': feed_rate,
        }

        # Define ODE term
        ode_term = diffrax.ODETerm(vector_field)

        # Time span
        t0 = 0.0
        t1 = t_end
        saveat = diffrax.SaveAt(ts=jnp.arange(t0, t1, dt))

        # Solve ODE
        solution = diffrax.diffeqsolve(
            ode_term,
            solver,
            t0=t0,
            t1=t1,
            dt0=dt,
            y0=y0,
            args=args,
            saveat=saveat,
            max_steps=16**4,  # Required for gradient computation
        )

        # Return final state
        return solution.ys[-1]

    return simulate, model_module


def create_loss_function(simulator, penalty_weight=1.0):
    """
    Create a loss function from a simulator.

    Args:
        simulator: Simulation function from create_simulator
        penalty_weight: Weight for differentiation penalty

    Returns:
        Loss function (threshold, feed_rate) -> scalar loss
    """
    def loss_fn(threshold, feed_rate):
        """
        Compute loss for given control parameters.

        Loss = -(Total Yield) + (Penalty * Differentiation)
        """
        final_state = simulator(threshold, feed_rate)
        S_final, D_final, N_final, Y_final = final_state

        # Loss: maximize yield, minimize differentiation
        loss = -Y_final + penalty_weight * D_final
        return loss

    return loss_fn


def create_gradient_function(loss_fn):
    """
    Create gradient function using JAX autodiff.

    Args:
        loss_fn: Loss function (threshold, feed_rate) -> loss

    Returns:
        Gradient function (threshold, feed_rate) -> dict of gradients
    """
    # Create gradient function for both parameters
    grad_fn = jax.grad(lambda params: loss_fn(params['threshold'], params['feed_rate']))

    def compute_gradients(threshold, feed_rate):
        """
        Compute gradients of loss w.r.t. control parameters.

        Returns:
            Dictionary {'threshold': grad_threshold, 'feed_rate': grad_feed_rate}
        """
        params = {'threshold': threshold, 'feed_rate': feed_rate}
        grads = grad_fn(params)
        return grads

    return compute_gradients


def create_batched_simulator(simulator):
    """
    Create a vectorized simulator for parameter sweeps.

    Args:
        simulator: Single simulation function

    Returns:
        Batched simulator that accepts arrays of parameters
    """
    return jax.vmap(simulator, in_axes=(0, 0))
