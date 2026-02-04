"""
Continuous logic module for differentiable passaging operations.

Uses sigmoid relaxation to avoid discrete if/else branching,
ensuring compatibility with jax.grad().
"""

import jax
import jax.numpy as jnp


def sigmoid_harvest(S, threshold, sharpness=10.0):
    """
    Smooth approximation to step function for harvesting.

    Args:
        S: Stem cell concentration (confluency)
        threshold: Target confluency for passaging (0.0 - 1.0)
        sharpness: Controls how sharp the transition is (higher = sharper)

    Returns:
        Activation value in [0, 1] indicating harvest strength
    """
    return jax.nn.sigmoid(sharpness * (S - threshold))


def relaxed_passage_rate(S, threshold, base_harvest_fraction=0.5, sharpness=10.0):
    """
    Calculate smooth passaging rate based on confluency.

    Instead of:
        if S > threshold:
            remove base_harvest_fraction * S

    We use:
        harvest_rate = sigmoid(S - threshold) * base_harvest_fraction * S

    Args:
        S: Current stem cell concentration
        threshold: Passaging threshold
        base_harvest_fraction: Fraction of cells to harvest when activated
        sharpness: Sigmoid sharpness parameter

    Returns:
        Continuous harvest rate
    """
    activation = sigmoid_harvest(S, threshold, sharpness)
    return activation * base_harvest_fraction * S


def nutrient_replenishment(N, feed_rate, N_max=1.0):
    """
    Continuous nutrient replenishment (feeding).

    Args:
        N: Current nutrient concentration
        feed_rate: Rate of nutrient addition
        N_max: Maximum nutrient concentration

    Returns:
        Rate of nutrient replenishment
    """
    # Replenish towards N_max with rate proportional to deficit
    return feed_rate * (N_max - N)


def augmented_vector_field(sbml_vector_field):
    """
    Wraps an SBML-derived vector field with continuous passaging logic.

    Args:
        sbml_vector_field: Function (t, y, args) -> dy/dt from sbmltoodejax

    Returns:
        Augmented vector field function compatible with diffrax
    """
    def vector_field(t, y, args):
        """
        Augmented dynamics with passaging and feeding.

        State vector y = [N, S, D, Y] where:
            N: Nutrients
            S: Stem cells
            D: Differentiated cells
            Y: Cumulative yield (harvested cells)

        Args:
            threshold: Passaging threshold (from args)
            feed_rate: Nutrient replenishment rate (from args)
        """
        # Extract state variables (order: N, S, D, Y)
        N, S, D, Y = y

        # Extract control parameters
        threshold = args.get('threshold', 0.7)
        feed_rate = args.get('feed_rate', 0.1)

        # Get biological dynamics from SBML (order: N, S, D)
        y_sbml = jnp.array([N, S, D])
        dy_sbml = sbml_vector_field(t, y_sbml, args)
        dN_bio, dS_bio, dD_bio = dy_sbml

        # Compute passaging (harvest) rate
        harvest_rate = relaxed_passage_rate(S, threshold)

        # Compute nutrient replenishment
        feed_rate_actual = nutrient_replenishment(N, feed_rate)

        # Augmented dynamics
        dN = dN_bio + feed_rate_actual  # Nutrients: biology + feeding
        dS = dS_bio - harvest_rate      # Stem cells: biology - harvest
        dD = dD_bio                      # Differentiated cells: biology only
        dY = harvest_rate                # Yield: accumulate harvested cells

        return jnp.array([dN, dS, dD, dY])

    return vector_field


def compute_loss(final_state, penalty_weight=1.0):
    """
    Loss function for optimization.

    Loss = -(Total Yield) + (Penalty * Differentiation)

    Args:
        final_state: Final state [N, S, D, Y]
        penalty_weight: Weight for differentiation penalty

    Returns:
        Scalar loss value (higher is worse)
    """
    N_final, S_final, D_final, Y_final = final_state

    # We want to maximize yield (Y) and minimize differentiation (D)
    # Since we minimize loss: loss = -yield + penalty * differentiation
    loss = -Y_final + penalty_weight * D_final

    return loss
