# Machine Learning Model for Continuum Suppression
In this project we have used machine learning models (Logistic Regression, SVM, DNN and BDT) to separate signal and background events.

Course: **IDC409 Project**  
Author: **Riddhi Panja (MS22237)**; **Pragnya Mishra (MS22108)**


# The Problem : Finding a Needle in a Haystack
In the Belle II experiment, electron-positron collisions are used to study rare B-meson decays. However, for every signal event, the accelerator produces many more unwanted background events.
Signal: These events are heavy, decay at rest, and produce a **"spherical"** particle topology. Type: 0 and 1
Background: These events are light, high-energy events that produce a **"jet-like"** topology. Type: 2, 3, 4 and 5
Objective is to train a model to recognize these different shapes and **suppress the background** to create a pure signal sample for physics analysis.


# Dataset and Setup
This project uses the following Python libraries. You can install them via pip:

```bash
pip install pandas numpy scikit-learn tensorflow matplotlib seaborn re lightgbm
```

# Steps to follow
1) Univariate analysis of the features can be performed using the following scripts:
   ```bash
   load_data.py
   feature_distribution_univariate.py
   feature_plots_univariate.py
   ```
2) Prerequisite
Before running any models, you must first process the raw data and generate the optimized feature set.
**Run this script once (before each model) to prepare all data:**
```bash
feature_reduc.py
```
3) Models: **run the following scripts for each model:**
   ```bash
   log_reg.py
   svm.py
   dnn.py
   bdt.py
   ``` 
