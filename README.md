# Hermite Schrödinger Bridge 🧠🌉

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository implements the **Hermite Schrödinger Bridge Matching (HSBM)** algorithm for generative modeling through optimal transport.

---

## Overview 📜

The **Schrödinger Bridge problem** provides a powerful framework for generative modeling by finding the optimal transport map between probability distributions, typically with entropic regularization. This implementation leverages a novel **sum-of-squares (SOS) Hermite polynomial parameterization** for the adjusted Schrödinger potential. This key choice enables the closed-form computation of the crucial drift function, leading to significant efficiency gains.

---

## ✨ Key Features

*   **Hermite Polynomial Parameterization:** Employs multivariate Hermite polynomials to represent the adjusted Schrödinger potential, offering a flexible and expressive function class.
*   **Explicit Drift Formulation:** Derives an analytically tractable expression for the drift function, avoiding costly numerical approximations.
*   **Computational Efficiency:** Exploits the structure of diagonal and lower triangular matrices (resulting from the SOS formulation) to reduce complexity from O(MK³) to O(K²), where K is related to the polynomial degree.
*   **Direct Optimization:** Enables straightforward gradient-based learning of the model parameters without requiring complex iterative procedures like Sinkhorn iterations during training.

---

## 🚀 Getting Started

Follow these steps to get the HSBM code running on your local machine.

### Prerequisites

*   Python 3.7+
*   `NumPy`
*   `PyTorch`
*   `Matplotlib` (for visualization)
*   `TensorLy` (optional, used for potential tensor-train decomposition extensions/experiments)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/hermite-schrodinger-bridge.git
    cd hermite-schrodinger-bridge
    ```
    *(Replace `yourusername` with the actual GitHub username or organization)*

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Ensure you have a `requirements.txt` file listing the packages above)*

### Usage Example

Here's a basic example of training the HSBM model to transport samples from a standard Gaussian distribution to a Gaussian mixture model:

```python
import numpy as np
from hsbm import HSBM # Assuming your main class is in hsbm.py

# --- 1. Define Source and Target Distributions ---
# Source: Standard Gaussian (2D)
source_samples = np.random.randn(5000, 2)

# Target: Example Gaussian Mixture Model (replace with your actual sampling)
# Example definition (replace with your actual parameters)
def sample_gaussian_mixture(centers, weights, covs, n_samples):
    # Simple GMM sampling logic (implement based on your needs)
    # This is just a placeholder
    components = np.random.choice(len(weights), size=n_samples, p=weights)
    samples = np.zeros((n_samples, centers.shape[1]))
    for i, comp_idx in enumerate(components):
        samples[i, :] = np.random.multivariate_normal(centers[comp_idx], covs[comp_idx])
    return samples

# Example GMM parameters (adjust these)
centers = np.array([[-2, -2], [2, 2], [0, 3]])
weights = np.array([0.3, 0.3, 0.4])
covs = [np.eye(2) * 0.5] * 3
target_samples = sample_gaussian_mixture(centers, weights, covs, 5000)

# --- 2. Initialize and Train the HSBM Model ---
model = HSBM(
    dim=2,          # Data dimension
    max_degree=6,   # Maximum degree of Hermite polynomials
    num_squares=10, # Number of squares in the SOS representation (L matrix columns)
    epsilon=0.5     # Diffusion coefficient (entropic regularization)
)

print("Starting training...")
model.train(
    source_samples,
    target_samples,
    num_iters=300,  # Number of training iterations
    lr=0.0005       # Learning rate
)
print("Training finished.")

# --- 3. Visualize Results ---
print("Generating and visualizing trajectories...")
model.visualize_trajectories(num_points=50) # Visualize trajectories of 50 points
print("Visualization complete.")
