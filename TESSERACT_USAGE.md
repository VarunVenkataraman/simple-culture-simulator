# Tesseract iPSC Simulator - Testing and Client Usage Guide

## Table of Contents
1. [Understanding the Architecture](#understanding-the-architecture)
2. [Testing the Docker Container](#testing-the-docker-container)
3. [Client-Side Usage](#client-side-usage)
4. [Developer Workflow](#developer-workflow)
5. [Integration Examples](#integration-examples)

---

## Understanding the Architecture

**Developer Role (Building the Container):**
- `tesseract_api.py` defines the **interface** of your simulator
- It uses Pydantic schemas to define inputs/outputs
- The `apply()` function is the entry point that Tesseract will call
- **You don't import tesseract_core here** - this file gets packaged INTO the container
- Think of it as: "This is what my simulator does"

**Client Role (Using the Container):**
- Clients import `from tesseract_core import Tesseract`
- They use this to **load and interact** with your pre-built container
- The tesseract_core SDK handles Docker/REST communication
- Think of it as: "I want to use someone's simulator"

**The Flow:**
```
Developer Side:                    Client Side:
┌─────────────────────┐           ┌──────────────────────┐
│ tesseract_api.py    │           │ from tesseract_core  │
│ (defines interface) │           │ import Tesseract     │
│                     │           │                      │
│ def apply(inputs):  │  ──build─>│ Tesseract.from_image │
│   return results    │           │   ('your-image')     │
└─────────────────────┘           └──────────────────────┘
        ↓                                    ↓
  tesseract build .                  tess.apply(inputs)
        ↓                                    ↓
   Docker Image                         JSON Results
```

### How Containerization Works

1. **Build Time:**
   - `tesseract build .` reads `tesseract_config.yaml`
   - Packages your Python files into a Docker image
   - Installs dependencies from `tesseract_requirements.txt`
   - Creates entry points for all Tesseract commands

2. **Runtime:**
   - Client loads the image using tesseract_core SDK
   - Container starts with your environment
   - Client calls `apply()` with JSON inputs
   - Your code runs inside container, returns JSON outputs

---

## Testing the Docker Container

### 1. Verify Container Was Built Successfully

```bash
# Check that the image exists
docker images | grep ipsc-culture-simulator

# Should show:
# ipsc-culture-simulator  1.0.0   <image-id>   <time>   <size>
# ipsc-culture-simulator  latest  <image-id>   <time>   <size>
```

### 2. Test CLI Interface - Basic Test

```bash
# Activate your virtual environment
source .venv/bin/activate

# Test with default-ish parameters
tesseract run ipsc-culture-simulator:latest apply '{"inputs": {"threshold": 0.7, "feed_rate": 0.1}}'
```

**Expected Output:**
```json
{
  "loss": -0.019038721919059753,
  "gradients": {
    "threshold": 0.6144416928291321,
    "feed_rate": -1.4371790885925293
  },
  "final_state": {
    "nutrients": 0.1851663440465927,
    "stem_cells": 0.05436909198760986,
    "differentiated_cells": 0.5287790894508362,
    "yield_value": 0.07340781390666962
  }
}
```

### 3. Test with Different Parameters

```bash
# Low threshold, high feed rate
tesseract run ipsc-culture-simulator:latest apply '{"inputs": {"threshold": 0.3, "feed_rate": 0.5}}'

# High threshold, low feed rate
tesseract run ipsc-culture-simulator:latest apply '{"inputs": {"threshold": 0.9, "feed_rate": 0.05}}'

# Edge cases
tesseract run ipsc-culture-simulator:latest apply '{"inputs": {"threshold": 0.0, "feed_rate": 0.0}}'
tesseract run ipsc-culture-simulator:latest apply '{"inputs": {"threshold": 1.0, "feed_rate": 1.0}}'
```

### 4. Verify Gradients are Non-Zero and Non-NaN

**What to check:**
-  Gradients should be real numbers (not NaN or Inf)
-  At least one gradient should be non-zero
-  Gradients should change when you change input parameters

**Test Script:**
```bash
# Save to test_gradients.sh
echo "Testing gradient computation..."

result1=$(tesseract run ipsc-culture-simulator:latest apply '{"inputs": {"threshold": 0.5, "feed_rate": 0.1}}' 2>/dev/null)
result2=$(tesseract run ipsc-culture-simulator:latest apply '{"inputs": {"threshold": 0.7, "feed_rate": 0.1}}' 2>/dev/null)

echo "Result 1: $result1"
echo "Result 2: $result2"

# Gradients should be different for different inputs
if [ "$result1" != "$result2" ]; then
    echo "✓ Gradients change with input parameters"
else
    echo "✗ Gradients are identical (potential issue)"
fi
```

### 5. Validate Output Schema

Every output should have:
- `loss` (float): The optimization objective
- `gradients` (object):
  - `threshold` (float): ∂loss/∂threshold
  - `feed_rate` (float): ∂loss/∂feed_rate
- `final_state` (object):
  - `nutrients` (float): Final N
  - `stem_cells` (float): Final S
  - `differentiated_cells` (float): Final D
  - `yield_value` (float): Final Y (cumulative harvest)

---

## Client-Side Usage

### Method 1: Python SDK (Recommended for Integration)

**Install tesseract-core on client machine:**
```bash
pip install tesseract-core
```

**Basic Usage:**
```python
from tesseract_core import Tesseract

# Load the Tesseract container
with Tesseract.from_image('ipsc-culture-simulator:latest') as tess:
    # Run simulation
    result = tess.apply(inputs={
        'threshold': 0.7,
        'feed_rate': 0.1
    })

    # Access results
    print(f"Loss: {result['loss']}")
    print(f"Gradients: {result['gradients']}")
    print(f"Final State: {result['final_state']}")
```

**Optimization Loop Example:**
```python
from tesseract_core import Tesseract
import numpy as np

with Tesseract.from_image('ipsc-culture-simulator:latest') as tess:
    # Initial parameters
    threshold = 0.7
    feed_rate = 0.1
    learning_rate = 0.01

    # Gradient descent
    for i in range(10):
        result = tess.apply(inputs={
            'threshold': threshold,
            'feed_rate': feed_rate
        })

        loss = result['loss']
        grads = result['gradients']

        # Update parameters (gradient descent)
        threshold -= learning_rate * grads['threshold']
        feed_rate -= learning_rate * grads['feed_rate']

        # Clamp to valid ranges
        threshold = np.clip(threshold, 0.0, 1.0)
        feed_rate = np.clip(feed_rate, 0.0, 1.0)

        print(f"Iteration {i}: loss={loss:.4f}, threshold={threshold:.4f}, feed_rate={feed_rate:.4f}")
```

### Method 2: CLI (Recommended for Scripting)

**One-off runs:**
```bash
tesseract run ipsc-culture-simulator:latest apply '{"inputs": {"threshold": 0.7, "feed_rate": 0.1}}'
```

**Save output to file:**
```bash
tesseract run ipsc-culture-simulator:latest apply \
  '{"inputs": {"threshold": 0.7, "feed_rate": 0.1}}' \
  --output-file results.json
```

**Read inputs from file:**
```bash
# Create input file
echo '{"inputs": {"threshold": 0.7, "feed_rate": 0.1}}' > input.json

# Run with file input
tesseract run ipsc-culture-simulator:latest apply @input.json
```

**Batch processing with shell script:**
```bash
#!/bin/bash
# batch_run.sh

for threshold in 0.3 0.5 0.7 0.9; do
  for feed_rate in 0.05 0.1 0.2 0.5; do
    echo "Running threshold=$threshold, feed_rate=$feed_rate"
    tesseract run ipsc-culture-simulator:latest apply \
      "{\"inputs\": {\"threshold\": $threshold, \"feed_rate\": $feed_rate}}" \
      --output-file "results_t${threshold}_f${feed_rate}.json"
  done
done
```

### Method 3: REST API (Recommended for Web Services)

**Start the Tesseract server:**
```bash
tesseract serve -p 8080 ipsc-culture-simulator:latest
```

**Make HTTP requests:**
```bash
# Using curl
curl -X POST http://127.0.0.1:8080/apply \
  -H "Content-Type: application/json" \
  -d '{"inputs": {"threshold": 0.7, "feed_rate": 0.1}}'

# Using httpie (pip install httpie)
http POST http://127.0.0.1:8080/apply \
  inputs:='{"threshold": 0.7, "feed_rate": 0.1}'
```

**Python requests:**
```python
import requests

# Start server first: tesseract serve -p 8080 ipsc-culture-simulator:latest

response = requests.post(
    'http://127.0.0.1:8080/apply',
    json={'inputs': {'threshold': 0.7, 'feed_rate': 0.1}}
)

result = response.json()
print(result)
```

**Access OpenAPI documentation:**
```bash
# While server is running, visit:
open http://127.0.0.1:8080/docs
```

---

## Developer Workflow

### Modifying the Simulator

1. **Edit source files:**
   - `src/continuous_logic.py` - Modify passaging logic
   - `models/generic_stem_cell.xml` - Change SBML model
   - `tesseract_api.py` - Update input/output schemas

2. **Regenerate SBML model (if you edited the XML):**
   ```bash
   source .venv/bin/activate
   python -c "
   from sbmltoodejax.parse import ParseSBMLFile
   from sbmltoodejax.modulegeneration import GenerateModel
   from pathlib import Path

   sbml_path = Path('models/generic_stem_cell.xml')
   modelData = ParseSBMLFile(str(sbml_path))
   GenerateModel(modelData, str(sbml_path.parent / f'{sbml_path.stem}.py'), deltaT=0.1)
   "
   ```

3. **Test locally first:**
   ```bash
   python main.py
   ```

4. **Rebuild Docker container:**
   ```bash
   tesseract build .
   ```

5. **Test the new container:**
   ```bash
   tesseract run ipsc-culture-simulator:latest apply '{"inputs": {"threshold": 0.7, "feed_rate": 0.1}}'
   ```

### Debugging Issues

**Container won't build:**
- Check `tesseract_config.yaml` for syntax errors
- Verify all files in `package_data` exist
- Check `tesseract_requirements.txt` for invalid packages

**Runtime errors:**
- Check import paths (should not use `src.` prefix)
- Verify all dependencies are in `tesseract_requirements.txt`
- Check file paths are relative to `/tesseract/` inside container

**Gradients are NaN:**
- Check for division by zero in ODEs
- Verify `max_steps` is set in diffrax solver
- Ensure all operations are differentiable (no if/else, use sigmoid)

**Inspect running container:**
```bash
# Start container in interactive mode
docker run -it --entrypoint /bin/bash ipsc-culture-simulator:latest

# Inside container:
ls /tesseract/
python /tesseract/tesseract_api.py  # Won't work directly, but shows imports
```

### Version Management

**Update version:**
```yaml
# tesseract_config.yaml
name: "ipsc-culture-simulator"
version: "1.1.0"  # Increment version
```

**Build with new version:**
```bash
tesseract build .
# Creates: ipsc-culture-simulator:1.1.0
```

**Tag as latest:**
```bash
docker tag ipsc-culture-simulator:1.1.0 ipsc-culture-simulator:latest
```

---

## Integration Examples

### Example 1: Parameter Sweep Study

```python
from tesseract_core import Tesseract
import pandas as pd
import numpy as np

# Parameter ranges
thresholds = np.linspace(0.1, 0.9, 9)
feed_rates = np.linspace(0.05, 0.5, 10)

results = []

with Tesseract.from_image('ipsc-culture-simulator:latest') as tess:
    for threshold in thresholds:
        for feed_rate in feed_rates:
            result = tess.apply(inputs={
                'threshold': float(threshold),
                'feed_rate': float(feed_rate)
            })

            results.append({
                'threshold': threshold,
                'feed_rate': feed_rate,
                'loss': result['loss'],
                'final_yield': result['final_state']['yield_value'],
                'final_differentiation': result['final_state']['differentiated_cells']
            })

# Save to CSV
df = pd.DataFrame(results)
df.to_csv('parameter_sweep.csv', index=False)
print(f"Completed {len(results)} simulations")
```

### Example 2: Bayesian Optimization

```python
from tesseract_core import Tesseract
from skopt import gp_minimize
from skopt.space import Real

def objective(params):
    """Objective function for Bayesian optimization."""
    threshold, feed_rate = params

    with Tesseract.from_image('ipsc-culture-simulator:latest') as tess:
        result = tess.apply(inputs={
            'threshold': float(threshold),
            'feed_rate': float(feed_rate)
        })

    return result['loss']

# Define search space
space = [
    Real(0.0, 1.0, name='threshold'),
    Real(0.0, 1.0, name='feed_rate')
]

# Run Bayesian optimization
result = gp_minimize(
    objective,
    space,
    n_calls=50,
    random_state=42
)

print(f"Best parameters: threshold={result.x[0]:.4f}, feed_rate={result.x[1]:.4f}")
print(f"Best loss: {result.fun:.4f}")
```

### Example 3: Gradient-Based Optimization with JAX

```python
from tesseract_core import Tesseract
import jax.numpy as jnp
import optax

def optimize_culture(initial_params, n_steps=100):
    """Optimize culture parameters using gradients."""
    with Tesseract.from_image('ipsc-culture-simulator:latest') as tess:
        # Initialize optimizer (Adam)
        optimizer = optax.adam(learning_rate=0.01)
        params = jnp.array([initial_params['threshold'], initial_params['feed_rate']])
        opt_state = optimizer.init(params)

        history = []

        for step in range(n_steps):
            # Get loss and gradients from Tesseract
            result = tess.apply(inputs={
                'threshold': float(params[0]),
                'feed_rate': float(params[1])
            })

            loss = result['loss']
            grads = jnp.array([
                result['gradients']['threshold'],
                result['gradients']['feed_rate']
            ])

            # Update parameters
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)

            # Clamp to valid range
            params = jnp.clip(params, 0.0, 1.0)

            history.append({
                'step': step,
                'loss': loss,
                'threshold': float(params[0]),
                'feed_rate': float(params[1])
            })

            if step % 10 == 0:
                print(f"Step {step}: loss={loss:.4f}, params={params}")

        return params, history

# Run optimization
final_params, history = optimize_culture({
    'threshold': 0.5,
    'feed_rate': 0.2
})

print(f"\nOptimal parameters:")
print(f"  Threshold: {final_params[0]:.4f}")
print(f"  Feed rate: {final_params[1]:.4f}")
```

### Example 4: Multi-Objective Optimization

```python
from tesseract_core import Tesseract
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
import numpy as np

class iPSCOptimization(Problem):
    def __init__(self):
        super().__init__(
            n_var=2,  # threshold, feed_rate
            n_obj=2,  # minimize differentiation, maximize yield
            n_constr=0,
            xl=np.array([0.0, 0.0]),
            xu=np.array([1.0, 1.0])
        )

    def _evaluate(self, X, out, *args, **kwargs):
        # X is array of [threshold, feed_rate] pairs
        with Tesseract.from_image('ipsc-culture-simulator:latest') as tess:
            results = []
            for params in X:
                result = tess.apply(inputs={
                    'threshold': float(params[0]),
                    'feed_rate': float(params[1])
                })

                # Objectives: minimize differentiation, maximize yield (so negate it)
                obj1 = result['final_state']['differentiated_cells']
                obj2 = -result['final_state']['yield_value']
                results.append([obj1, obj2])

            out["F"] = np.array(results)

# Run multi-objective optimization
problem = iPSCOptimization()
algorithm = NSGA2(pop_size=20)

res = minimize(
    problem,
    algorithm,
    ('n_gen', 50),
    verbose=True
)

print(f"Found {len(res.X)} Pareto optimal solutions")
```

---

## Quick Reference

### Common Commands

```bash
# Build container
tesseract build .

# Run simulation
tesseract run ipsc-culture-simulator:latest apply '{"inputs": {...}}'

# Serve on port 8080
tesseract serve -p 8080 ipsc-culture-simulator:latest

# Check available commands
tesseract run ipsc-culture-simulator:latest --help

# List Docker images
docker images | grep ipsc-culture-simulator

# Remove old images
docker rmi ipsc-culture-simulator:old-version
```

### Input Schema

```json
{
  "inputs": {
    "threshold": 0.7,    // float, range [0.0, 1.0], default: 0.7
    "feed_rate": 0.1     // float, range [0.0, 1.0], default: 0.1
  }
}
```

### Output Schema

```json
{
  "loss": -0.019,                    // float, optimization objective
  "gradients": {
    "threshold": 0.614,              // float, ∂loss/∂threshold
    "feed_rate": -1.437              // float, ∂loss/∂feed_rate
  },
  "final_state": {
    "nutrients": 0.185,              // float, final N concentration
    "stem_cells": 0.054,             // float, final S concentration
    "differentiated_cells": 0.528,   // float, final D concentration
    "yield_value": 0.073             // float, cumulative harvested cells
  }
}
```

For questions or issues, check the [Tesseract documentation](https://docs.pasteurlabs.ai/projects/tesseract-core/stable/).
