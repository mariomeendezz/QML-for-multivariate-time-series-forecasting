# About the repository

This repository contains an experimental reproduction of selected results from the paper:

Quantum-enhanced architectures for multivariate time-series forecasting
Sandra Ranilla-Cortina, Diego A. Aranda, Jorge Ballesteros, Jesús Bonilla, Nerea Monrio & Elías F. Combarro 

The goal of this project is to reproduce, analyze, and extend the main experimental ideas presented in the paper using Python-based quantum computing and machine learning tools.

---


## Project Requirements

First, create a virtual environment:

```bash
python -m venv .venv
```

Then activate it:

```bash
# Windows
.venv\Scripts\activate
```

```bash
# macOS/Linux
source .venv/bin/activate
```

Finally, install all required dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

The recommended workflow is:

1. Create and activate a virtual environment.
2. Install the dependencies from `requirements.txt`.
3. Open the notebooks with Jupyter Notebook, JupyterLab, VS Code, or another compatible environment.
4. Run each notebook cell by cell in order.


## Sources

- Quantum-enhanced architectures for multivariate time-series forecasting - Sandra Ranilla-Cortina, Diego A. Aranda, Jorge Ballesteros, Jesús Bonilla, Nerea Monrio & Elías F. Combarro: https://link.springer.com/article/10.1007/s11227-026-08295-x 
- Qiskit documentation: https://quantum.cloud.ibm.com/docs
- PennyLane documentation: https://docs.pennylane.ai
- PyTorch documentation: https://pytorch.org/docs/stable/index.html
- A Practical Guide to Quantum Computing. Elías F. Combarro, Samuel González-Castillo. Packt (2025).
- A Practical Guide to Quantum Machine Learning and Quantum Optimization. Elías F. Combarro, Samuel González-Castillo. Packt (2023).

## License

This project is licensed under the MIT License.