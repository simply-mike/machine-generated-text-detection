# Source code

This folder contains reusable code for inference.

Current files:

```text
src/
├── __init__.py
├── inference.py       # Checkpoint loading and prediction wrapper
├── modeling.py        # Encoder + mean pooling + MLP classifier
└── preprocessing.py   # Text normalization shared with training
```

The training pipeline is still notebook-first. A future production-oriented refactor could add:

```text
src/
├── dataset.py
├── model.py
├── train.py
├── evaluate.py
├── inference.py
└── preprocessing.py
```

Important: keep `preprocessing.py` consistent with the preprocessing used during model training. Train/inference skew is a common source of degraded NLP model performance.
