# Copula-Based Feature Selection for Diabetes Risk Prediction 💡📈

[![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/ML-scikit--learn%2Fxgboost%2Ftensorflow-orange)](https://scikit-learn.org/stable/)
[![Copulas](https://img.shields.io/badge/Stats-Copula%20Modeling-6E40AA)](https://en.wikipedia.org/wiki/Copula_(probability_theory))

This repository contains the full implementation of our machine learning pipeline that introduces a **copula-based approach** for feature selection, tailored for diabetes risk prediction. We use the **Gumbel copula's upper-tail dependence coefficient** to rank features based on their extreme-value co-movement with the diabetes label. This novel technique is benchmarked against Mutual Information (MI), mRMR, ReliefF and L1EN feature selection methods.

All results, discussions, and implications are provided in our accompanying research paper. This repository focuses strictly on code and reproducibility.

Information on the datasets used are provided below.

## CDC Diabetes Health Indicators Dataset  

**Source:** [UCI Machine Learning Repository – Diabetes Health Indicators Dataset](https://archive.ics.uci.edu/ml/datasets/diabetes+health+indicators)  
**Original Data:** Behavioral Risk Factor Surveillance System (BRFSS) survey, U.S. Centers for Disease Control and Prevention (CDC)  
**License/Usage:** Data are derived from a public health survey and made publicly available for research purposes.  
**Note:** No modifications were made (except formatting for our experiments).


## Pima Indians Diabetes Dataset  

**Source:** [Kaggle – Pima Indians Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)  
**License:** [CC0: Public Domain](https://creativecommons.org/publicdomain/zero/1.0/)  
**Note:** No modifications were made (except formatting for our experiments).

---

## 📦 Requirements

- **Python** 3+
- Install required packages via:

```bash
!pip install numpy pandas scipy scikit-learn xgboost statsmodels joblib ucimlrepo seaborn
```

## 🚀 Getting Started
```bash
git clone https://github.com/agnivibes/copula-feature-selection.git
cd copula-feature-selection
```

## Run the full analysis:
```bash
python copula_feature_selection.py
```

## 🔬 Research Paper
Aich, A., Murshed, M., Mayeaux, A., Hewage, S. (2025). Can Copulas Be Used in Feature Selection? 
A Machine Learning Study on Diabetes Risk Prediction [Manuscript under review]

## 📊 Citation
If you use this code or method in your own work, please cite:

@article{Aich2025A2CopulaDiabetes,
  title   = {Can Copulas Be Used in Feature Selection? A Machine Learning Study on Diabetes Risk Prediction},
  author  = {Aich, Agnideep and Murshed, Md Monzur and Mayeaux, Amanda and Hewage, Sameera},
  journal = {},
  year    = {2025},
  note    = {Manuscript under review}
}

## 📬 Contact
For questions or collaborations, feel free to contact:

Agnideep Aich,
Department of Mathematics, University of Louisiana at Lafayette
📧 agnideep.aich1@louisiana.edu

## 📝 License

This project is licensed under the [MIT License](LICENSE).
