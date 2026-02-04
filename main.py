"""
Main entry point for iPSC culture simulator.

Tests forward simulation and gradient computation.
"""

import jax.numpy as jnp
from src.tesseract_interface import iPSCSimulator


def compute_loss_and_grad(threshold=0.7, feed_rate=0.1):
    """
    Run simulation and compute loss with gradients.

    Args:
        threshold: Passaging threshold (0.0 - 1.0)
        feed_rate: Nutrient replenishment rate

    Returns:
        Tuple of (loss, gradients)
    """
    # Initialize simulator
    simulator = iPSCSimulator()

    # Compute loss and gradients
    loss = float(simulator.compute_loss(threshold, feed_rate))
    grads = simulator.compute_gradients(threshold, feed_rate)

    # Convert gradients to regular Python floats
    grads_dict = {
        'threshold': float(grads['threshold']),
        'feed_rate': float(grads['feed_rate'])
    }

    return loss, grads_dict


def main():
    """
    Main test function.
    """
    print("=" * 60)
    print("iPSC Culture Simulator - Gradient Verification")
    print("=" * 60)

    # Initialize simulator
    print("\n[1/4] Initializing simulator...")
    simulator = iPSCSimulator()
    print(" Simulator initialized successfully")

    # Test parameters
    threshold = 0.7
    feed_rate = 0.1

    print(f"\n[2/4] Running forward simulation...")
    print(f"  Parameters:")
    print(f"    - threshold: {threshold}")
    print(f"    - feed_rate: {feed_rate}")

    # Run simulation
    final_state = simulator.simulate(threshold, feed_rate)
    N_final, S_final, D_final, Y_final = final_state

    print(f"  Final state:")
    print(f"    - Nutrients (N): {float(N_final):.4f}")
    print(f"    - Stem cells (S): {float(S_final):.4f}")
    print(f"    - Differentiated cells (D): {float(D_final):.4f}")
    print(f"    - Yield (Y): {float(Y_final):.4f}")
    print(" Forward simulation completed")

    # Compute loss
    print(f"\n[3/4] Computing loss...")
    loss = simulator.compute_loss(threshold, feed_rate)
    print(f"  Loss: {float(loss):.6f}")
    print(" Loss computed")

    # Compute gradients
    print(f"\n[4/4] Computing gradients...")
    grads = simulator.compute_gradients(threshold, feed_rate)

    grad_threshold = float(grads['threshold'])
    grad_feed_rate = float(grads['feed_rate'])

    print(f"  Gradients:")
    print(f"    - loss/threshold: {grad_threshold:.6f}")
    print(f"    - loss/feed_rate: {grad_feed_rate:.6f}")

    # Verify gradients are valid
    print("\n" + "=" * 60)
    print("Gradient Verification")
    print("=" * 60)

    all_passed = True

    # Check for zero gradients
    if grad_threshold == 0.0:
        print(" FAIL: Gradient w.r.t. threshold is zero")
        all_passed = False
    else:
        print(f" PASS: Gradient w.r.t. threshold is non-zero ({grad_threshold:.6f})")

    if grad_feed_rate == 0.0:
        print(" FAIL: Gradient w.r.t. feed_rate is zero")
        all_passed = False
    else:
        print(f" PASS: Gradient w.r.t. feed_rate is non-zero ({grad_feed_rate:.6f})")

    # Check for NaN gradients
    if jnp.isnan(grad_threshold):
        print(" FAIL: Gradient w.r.t. threshold is NaN")
        all_passed = False
    else:
        print(" PASS: Gradient w.r.t. threshold is not NaN")

    if jnp.isnan(grad_feed_rate):
        print(" FAIL: Gradient w.r.t. feed_rate is NaN")
        all_passed = False
    else:
        print(" PASS: Gradient w.r.t. feed_rate is not NaN")

    print("\n" + "=" * 60)
    if all_passed:
        print(" ALL TESTS PASSED")
        print("The simulator is fully differentiable!")
    else:
        print(" SOME TESTS FAILED")
        print("Please check the implementation.")
    print("=" * 60)

    return all_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
