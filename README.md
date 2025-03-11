# Gait Stability Assessment Using Nonlinear Analysis and Artificial Intelligence

This repository contains the code, and documentation for a research project focused on developing a methodology to assess gait stability by integrating nonlinear analysis techniques and artificial intelligence (AI).

## Project Overview

Gait stability is a critical factor in reducing fall risk. Current methodologies often lack a comprehensive approach to capturing the complexity of human movement. This project aims to close that gap by combining:

- **Nonlinear Analysis:** Techniques like Lyapunov exponents, recurrence quantification analysis, entropy measures, and fractal dynamics to characterize gait variability and stability.
- **Artificial Intelligence:** To identify key observation variables and predict fall risk.
- **Biomechanical Data:** A unique dataset with spatiotemporal information from full-body markers 

## Objectives

1. **Data Transformation:** Process and transform the dataset to extract stability indicators using nonlinear analysis techniques.
2. **Statistical Analysis:** Identify the most significant variables for classifying gait patterns between healthy individuals and prosthesis users.
3. **AI Model Development:** Build and compare machine learning models to predict fall risk based on the identified variables.
4. **Model Evaluation:** Validate the AI models’ performance and generalization capabilities with new data.

## Project Structure

```
├── data                    # Raw and processed datasets
├── notebooks               # Jupyter notebooks for data analysis and modeling
├── src                     # Source code for feature extraction, modeling, and evaluation
│   ├── preprocessing       # Data transformation scripts
│   ├── analysis            # Nonlinear analysis techniques implementation
│   ├── models              # AI models and training scripts
│   └── evaluation          # Model performance evaluation
├── docs                    # Project documentation
└── README.md               # Project overview
```

## Requirements

- Python 3.8+
- Libraries: NumPy, Pandas, SciPy, Scikit-learn, TensorFlow/PyTorch, Matplotlib, Seaborn
- Additional tools: Jupyter Notebook, SolidWorks, Gazebo (for physical simulation)
``



## Contact

For questions or collaboration opportunities, please reach out via dianacmartinez13@gmail.com

