"""
Tesseract interface module (placeholder for future Docker integration).

This module will eventually expose the simulator via Tesseract API.
For now, it provides a simple Python interface.
"""

from pathlib import Path
from bridge import create_simulator, create_loss_function, create_gradient_function


class iPSCSimulator:
    """
    iPSC culture simulator with differentiable passaging.

    This will be wrapped in a Tesseract container in the future.
    """

    def __init__(self, sbml_path=None):
        """
        Initialize the simulator.

        Args:
            sbml_path: Path to SBML model file
        """
        if sbml_path is None:
            # Default to the generic stem cell model
            sbml_path = Path(__file__).parent / "models" / "generic_stem_cell.xml"

        self.sbml_path = Path(sbml_path)

        # Create simulator
        self.simulator, self.modelData = create_simulator(self.sbml_path)

        # Create loss function
        self.loss_fn = create_loss_function(self.simulator)

        # Create gradient function
        self.grad_fn = create_gradient_function(self.loss_fn)

    def simulate(self, threshold, feed_rate):
        """
        Run forward simulation.

        Args:
            threshold: Passaging threshold (0.0 - 1.0)
            feed_rate: Nutrient feed rate

        Returns:
            Final state [S, D, N, Y]
        """
        return self.simulator(threshold, feed_rate)

    def compute_loss(self, threshold, feed_rate):
        """
        Compute loss for given parameters.

        Args:
            threshold: Passaging threshold
            feed_rate: Nutrient feed rate

        Returns:
            Scalar loss value
        """
        return self.loss_fn(threshold, feed_rate)

    def compute_gradients(self, threshold, feed_rate):
        """
        Compute gradients of loss w.r.t. parameters.

        Args:
            threshold: Passaging threshold
            feed_rate: Nutrient feed rate

        Returns:
            Dictionary of gradients
        """
        return self.grad_fn(threshold, feed_rate)

    def run_optimization_step(self, threshold, feed_rate):
        """
        Run one forward pass with gradient computation.

        Args:
            threshold: Passaging threshold
            feed_rate: Nutrient feed rate

        Returns:
            Tuple of (loss, gradients, final_state)
        """
        loss = self.compute_loss(threshold, feed_rate)
        grads = self.compute_gradients(threshold, feed_rate)
        final_state = self.simulate(threshold, feed_rate)

        return loss, grads, final_state
