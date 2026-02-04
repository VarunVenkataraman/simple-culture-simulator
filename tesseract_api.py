"""
Tesseract API for iPSC culture simulator.

Exposes the differentiable simulator through Tesseract's standard interface.
"""

from pydantic import BaseModel, Field
from typing import Dict
from tesseract_interface import iPSCSimulator


# Initialize the simulator globally
_simulator = None


def _get_simulator():
    """Get or create simulator instance."""
    global _simulator
    if _simulator is None:
        _simulator = iPSCSimulator()
    return _simulator


# Define input schema using Pydantic
class InputSchema(BaseModel):
    threshold: float = Field(
        default=0.7,
        description="Target confluency for passaging (0.0 - 1.0)",
        ge=0.0,
        le=1.0
    )
    feed_rate: float = Field(
        default=0.1,
        description="Continuous nutrient replenishment rate",
        ge=0.0,
        le=1.0
    )


# Define output schema using Pydantic
class GradientSchema(BaseModel):
    threshold: float = Field(description="∂loss/∂threshold")
    feed_rate: float = Field(description="∂loss/∂feed_rate")


class FinalStateSchema(BaseModel):
    model_config = {"populate_by_name": True}

    nutrients: float = Field(description="Final nutrient concentration (N)")
    stem_cells: float = Field(description="Final stem cell concentration (S)")
    differentiated_cells: float = Field(description="Final differentiated cell concentration (D)")
    yield_value: float = Field(alias="yield", description="Cumulative harvested cells (Y)")


class OutputSchema(BaseModel):
    loss: float = Field(description="Optimization objective: -(yield) + penalty*(differentiation)")
    gradients: GradientSchema = Field(description="Partial derivatives of loss w.r.t. inputs")
    final_state: FinalStateSchema = Field(description="Final simulation state")


# Main Tesseract entry point
def apply(inputs: InputSchema) -> OutputSchema:
    """
    Run iPSC culture simulation with given parameters.

    This is the main entry point that Tesseract will call.

    Args:
        inputs: Input parameters (threshold, feed_rate)

    Returns:
        Simulation results including loss, gradients, and final state
    """
    # Get simulator instance
    simulator = _get_simulator()

    # Extract inputs
    threshold = float(inputs.threshold)
    feed_rate = float(inputs.feed_rate)

    # Run simulation
    final_state = simulator.simulate(threshold, feed_rate)

    # Compute loss
    loss = float(simulator.compute_loss(threshold, feed_rate))

    # Compute gradients
    grads = simulator.compute_gradients(threshold, feed_rate)

    # Build output schema
    return OutputSchema(
        loss=loss,
        gradients=GradientSchema(
            threshold=float(grads['threshold']),
            feed_rate=float(grads['feed_rate'])
        ),
        final_state=FinalStateSchema(
            nutrients=float(final_state[0]),
            stem_cells=float(final_state[1]),
            differentiated_cells=float(final_state[2]),
            yield_value=float(final_state[3])
        )
    )
