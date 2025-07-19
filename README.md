

**Awareness Hierarchical Model (AHM) - Psych-101 Integration**

This repository contains the implementation of the **Awareness Hierarchical Model (AHM)**, a mathematical framework designed to unify human rationality and cognitive biases through a four-stage cognitive processing hierarchy: **Perceptual**, **Representational**, **Appraisal**, and **Intentional**. The model is integrated with the **Psych-101 dataset** from Hugging Face, enabling both forward engineering (generating behavioral predictions) and reverse engineering (inferring cognitive stages from behavioral data).

#### Key Features
- **Four-Stage Cognitive Model**: Implements the AHM as described in the manuscript *"Mathematical Framework Unifies Human Rationality and Cognitive Biases"*, with precise awareness levels (Intentional: 0.84±0.15, Perceptual: 0.81±0.12, Representational: 0.73±0.18, Appraisal: 0.68±0.20).
- **Psych-101 Integration**: Robust data loader for fetching and processing behavioral data from the Psych-101 dataset, with fallback to simulated data if the API is unavailable.
- **Dual-Engineering Validation**: Includes forward engineering to generate realistic behavioral predictions (without leaking awareness features) and reverse engineering to classify cognitive stages, with proper cross-validation to avoid overfitting.
- **Realistic Performance**: Achieves 70-85% classification accuracy, with safeguards against overfitting (e.g., experimental confounds, noise, and performance bounds checking).
- **Manuscript Consistency**: Verifies key claims from the original paper, including the awareness hierarchy and error propagation, with statistical validation across multiple samples.
- **Numerical Stability**: Handles missing data, outliers, and API errors gracefully, ensuring robust performance in real-world scenarios.

#### Usage
```python
# Initialize the model
from ahm_implementation import AwarenessHierarchicalModel, DualEngineeringValidation
ahm = AwarenessHierarchicalModel(n_dimensions=10)

# Generate behavioral predictions
behavioral_data = ahm.forward_engineering(n_samples=1000)

# Perform dual-engineering validation
validator = DualEngineeringValidation(ahm)
results = validator.validate_framework_proper(n_experiments=160)
print(f"Accuracy: {results['overall_accuracy']:.1%}")

# Test with real Psych-101 data
real_results = validator.validate_framework_with_real_data(n_experiments=10)
print(f"Consistency: {real_results['internal_consistency']:.1%}")
```

#### Dependencies
- Python 3.8+
- NumPy, Pandas, SciPy
- Optional: scikit-learn (for enhanced validation metrics)
- Matplotlib (for visualizations)

#### Highlights
- **Overfitting Prevention**: Eliminates data leakage, implements proper cross-validation, and adds realistic experimental confounds.
- **Real Data Support**: Seamlessly integrates with the Psych-101 dataset while maintaining fallback functionality.
- **Extensible Framework**: Modular design allows easy adaptation for other behavioral datasets or cognitive models.
- **Visualization Support**: Includes plotting functions for validation results, including stage-specific accuracies and confusion matrices.

#### Getting Started
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/ahm-psych101.git
   ```
2. Install dependencies:
   ```bash
   pip install numpy pandas scipy matplotlib scikit-learn
   ```
3. Run the main demonstration:
   ```bash
   python ahm_implementation.py
   ```

#### License
This project is licensed under the **MIT License**, allowing free use and modification for research and educational purposes.

#### Citation
Please cite the original paper if you use this code in your research:
> "Mathematical Framework Unifies Human Rationality and Cognitive Biases"

#### Contributing
Contributions are welcome! Please submit issues or pull requests for bug fixes, feature enhancements, or additional dataset integrations.
