# Comparative Analysis of Classification Models

> A machine learning project comparing the performance of different classification algorithms on the same dataset

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-green)

## Overview

This project provides a **side-by-side comparison** of three popular classification algorithms to help understand their strengths, weaknesses, and ideal use cases. By evaluating each model on the same dataset with identical preprocessing, we can make fair comparisons of their performance.

### Why Compare Models?

Different classification algorithms work better for different types of data:
- **Logistic Regression**: Best for linearly separable data, highly interpretable
- **Decision Tree**: Handles non-linear relationships, easy to visualize
- **Random Forest**: Robust to overfitting, works well with complex data

## Models Compared

| Model | Type | Strengths | Best For |
|-------|------|-----------|----------|
| **Logistic Regression** | Linear | Fast, interpretable, probabilistic outputs | Binary classification, baseline model |
| **Decision Tree** | Non-linear | Visual, handles mixed data types | Explainable AI, feature importance |
| **Random Forest** | Ensemble | High accuracy, reduces overfitting | Complex datasets, production systems |

## Evaluation Metrics

Each model is evaluated using multiple metrics to provide a comprehensive view:

| Metric | What It Measures | When It's Important |
|--------|------------------|---------------------|
| **Accuracy** | Overall correct predictions | Balanced datasets |
| **Precision** | True positives / Predicted positives | When false positives are costly |
| **Recall** | True positives / Actual positives | When false negatives are costly |
| **F1-Score** | Harmonic mean of precision & recall | Imbalanced datasets |
| **Confusion Matrix** | Detailed prediction breakdown | Understanding error types |

## Project Structure

```
comparative-analysis-of-classification-models/
├── Decision_tree.ipynb       # Decision Tree implementation & analysis
├── Logistic_regression.ipynb # Logistic Regression implementation & analysis
├── Random_forest.ipynb       # Random Forest implementation & analysis
├── dataset_assignment1.csv   # Dataset used for all models
└── README.md                 # Project documentation
```

## Tech Stack

| Technology | Purpose |
|------------|----------|
| **Python** | Programming language |
| **Pandas** | Data manipulation & analysis |
| **NumPy** | Numerical computations |
| **Scikit-learn** | ML algorithms & metrics |
| **Matplotlib** | Static visualizations |
| **Seaborn** | Statistical visualizations |
| **Jupyter Notebook** | Interactive development |

## Getting Started

### Prerequisites
- Python 3.x
- Jupyter Notebook or JupyterLab

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/magnus210/comparative-analysis-of-classification-models.git
   cd comparative-analysis-of-classification-models
   ```

2. **Install dependencies**
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn jupyter
   ```

3. **Launch Jupyter**
   ```bash
   jupyter notebook
   ```

4. **Open any notebook** to explore the model implementations

## How to Use

### Exploring Individual Models
1. Open `Logistic_regression.ipynb` for baseline linear classification
2. Open `Decision_tree.ipynb` for tree-based classification
3. Open `Random_forest.ipynb` for ensemble learning approach

### Comparing Results
Each notebook contains:
- Data loading and preprocessing
- Model training with hyperparameters
- Performance metrics calculation
- Visualizations (confusion matrices, ROC curves)

## Key Findings

| Model | Typical Strengths | Typical Weaknesses |
|-------|-------------------|--------------------|
| Logistic Regression | Fast training, good baseline | Struggles with non-linear data |
| Decision Tree | Interpretable, handles outliers | Prone to overfitting |
| Random Forest | High accuracy, robust | Slower, less interpretable |

## Learning Outcomes

This project demonstrates:
- ✅ How to implement classification algorithms from scratch
- ✅ Proper train/test splitting methodology
- ✅ Evaluation metrics interpretation
- ✅ Model comparison best practices
- ✅ Data visualization for ML results

## Future Improvements

- [ ] Add cross-validation for more robust evaluation
- [ ] Include hyperparameter tuning (GridSearchCV)
- [ ] Add more models (SVM, KNN, Naive Bayes)
- [ ] Create a summary comparison notebook
- [ ] Add ROC curve comparisons

## References

- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Classification Metrics Guide](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [Random Forest vs Decision Tree](https://en.wikipedia.org/wiki/Random_forest)

## License

This project is open source and available for educational purposes.
