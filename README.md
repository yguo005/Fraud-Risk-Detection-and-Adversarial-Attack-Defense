# Fraud Risk Detection and Adversarial Attack Defense

**Authors:** Jing Xia, Siyuan Liu, Yunyu Guo, Yuqing Qiao

**Implementation Code:**
*   `CS5100 final proj.ipynb`https://colab.research.google.com/drive/18Sc759kyVsH2GvfdPjZjGCbBlAn0BTDf?usp=sharing
*   `ShadowModel - Privacy ML.ipynb`https://colab.research.google.com/drive/1rv5yAovyFdSuGjnoTb3Htxh1EneNeuK6?usp=sharing

## Introduction

This project aims to enhance fraud detection in financial transactions using machine learning models. Traditional rule-based systems often fail to identify sophisticated fraud patterns. This project focuses on using features like transaction amounts, categories, age, gender, merchants, and user behavior.

A key challenge is the high sensitivity of transaction data and privacy concerns from clients (banks, fintech companies, buyers). Even within a single institution, data access needs to be controlled. Therefore, this project also explores methods to protect data privacy while maintaining model performance, specifically by enabling the model to process encrypted data and return encrypted outputs, ensuring no sensitive information is revealed outside the client's environment.

**Goal:** Develop an ML model for detecting fraudulent transactions that can operate on encrypted data to ensure privacy.

## Related Works (Common Fraud Detection Methods)

*   Decision Tree
*   Random Forest
*   Support Vector Machine (SVM)
*   Neural Network
*   Naive Bayes

## Dataset: BankSim

*   **Source:** An agent-based simulator of bank payments based on aggregated transactional data from a bank in Spain.
*   **Purpose:** Generation of synthetic data for fraud detection research.
*   **Privacy Libraries Explored:**
    *   TenSEAL: [https://github.com/OpenMined/TenSEAL](https://github.com/OpenMined/TenSEAL)
    *   OpenFHE: [https://openfhe-development.readthedocs.io/en/latest/](https://openfhe-development.readthedocs.io/en/latest/)

## Exploratory Data Analysis (EDA)

*(See Colab notebooks for full details)*

**Data Features:**
*   **No NAs** in the dataset.
*   **Step:** Day of the transaction.
*   **Customer:** Unique customer ID (C + 10 digits, 4,109 unique).
*   **Age:** Categorical age intervals (0-6, U for Unknown).
    *   0: <18 yrs
    *   1: 19-25 yrs
    *   2: 26-35 yrs
    *   3: 36-45 yrs
    *   4: 46-55 yrs
    *   5: 56-65 yrs
    *   6: >65 yrs
    *   U: Unknown (only for 'Enterprise' gender).
*   **Gender:** F (Female), M (Male), E (Enterprise), U (Unknown).
    *   Unknown gender group (170 customers) primarily in age groups 1, 2, 3.
*   **Merchant:** Unique merchant ID (M + 9 digits, 50 unique).
*   **Category:** 15 unique transaction categories (e.g., transportation, food, health, tech).
*   **Amount:** Transaction value (52 values are 0, no negative values).
*   **Fraud:** Flag (0 for clean, 1 for fraudulent).
*   **zipcodeOri, zipMerchant:** Removed (constant value 28007, Ansonville, NC). Amount expressed in US dollars.

**EDA Insights & Visualizations:**
*   **Average Spending:**
    *   By Category: 'travel' highest, 'transportation' lowest.
    *   By Age: Group '0' (<18) highest, Group '7' (Unknown) lowest.
    *   By Gender: 'F' highest, 'E' lowest.
*   **Fraud Percentage vs. Spent Amount:**
    *   Close to no fraud for transactions < $500.
    *   Fraud jumps to ~90% or more for amounts > $500.
    *   100% fraud for transactions > $2500.
*   **Fraud in Spending Categories:**
    *   Most fraud: 'sportsandtoys', 'health', 'wellnessandbeauty', 'leisure'.
    *   >50% fraud: 'leisure', 'travel', 'sportsandtoys'.
    *   100% clean: 'transportation', 'food', 'contents'.
*   **Merchants and Fraud:** (Visualization shows fraud distribution across different merchants).

## Data Processing

*   **Dropped Columns:** `step`, `zipcodeOri`, `zipMerchant`.
*   **Handled Unknowns:**
    *   'U' in Age replaced with '7'. (Enterprise gender often has unknown age).
    *   'U' in Gender erased (affects customers in age intervals 1, 2, 3 with F=55%, M=44%, U=1% distribution).
*   Removed punctuations.
*   Encoded features for ML model training.

## Handling Imbalanced Data

*   **Issue:** Dataset is highly imbalanced (only 1.2% fraud out of 586,928 cases).
*   **Solution:** Synthetic Minority Oversampling Technique (SMOTE) to create new, synthetic samples for the minority (fraud) class.

## AI Methods Evaluated

Five classification models were selected:
1.  **Random Forest:** Robust ensemble for complex, high-dimensional data.
2.  **K Neighbors Classifier (KNN):** Simple instance-based learner.
3.  **Logistic Regression:** Linear model, interpretable and efficient.
4.  **XGBoost Classifier:** Gradient boosting, often state-of-the-art.
5.  **Multilayer Perceptron (MLP):** Deep learning model for non-linear patterns.

**Evaluation Metrics:** Precision, Recall, F1-score, Accuracy, ROC AUC, Precision-Recall AUC.

## Results

### Model Performance (Without Resampling/SMOTE)

| Model              | Precision | Recall | F1-score |
| :----------------- | :-------- | :----- | :------- |
| Random Forest      | 0.95      | 0.83   | 0.88     |
| KNN                | 0.93      | 0.79   | 0.85     |
| Logistic Regression| 0.58      | 0.93   | 0.62     |
| **XGBoost**        | **0.93**  | **0.86**| **0.89** |
| MLP                | 0.94      | 0.71   | 0.79     |

*   **ROC AUC:** XGBoost had the best ROC AUC (1.00), indicating excellent class discrimination. Random Forest (0.99), MLP (0.98), KNN (0.95).
*   **Precision-Recall AUC:** XGBoost (0.84) and Random Forest (0.86) had the best PR AUC, indicating better performance on the imbalanced data for distinguishing the minority (fraud) class.

**Conclusion (Without Resampling):** XGBoost and Random Forest show the best performance.

### Model Performance (After Resampling with SMOTE)

| Model              | Precision | Recall | F1-score |
| :----------------- | :-------- | :----- | :------- |
| Random Forest      | 0.84      | 0.86   | 0.85     |
| KNN                | 0.70      | 0.90   | 0.76     |
| Logistic Regression| 0.60      | 0.93   | 0.64     |
| **XGBoost**        | **0.85**  | **0.88**| **0.87** |
| MLP                | 0.60      | 0.89   | 0.65     |

*   **ROC AUC (After SMOTE):** XGBoost (0.99) best. Random Forest (0.98), MLP (0.98), KNN (0.92).
*   **Precision-Recall AUC (After SMOTE):** XGBoost (0.82) best. Random Forest (0.77).

**Conclusion (After Resampling):** XGBoost has the best overall performance based on Precision, Recall, and F1-score.

### Final Tuned Model: XGBoost

*   K-Fold cross-validation was used for hyperparameter tuning.
*   **Potential Hyperparameters explored:**
    *   `n_estimators`: 100, 200, 300
    *   `learning_rate`: 0.01, 0.1, 0.2
    *   `max_depth`: 3, 5, 7
*   **Best Hyperparameters found:**
    *   `n_estimators`: 200
    *   `learning_rate`: 0.2
    *   `max_depth`: 7
*   **Final XGBoost Results (with tuned parameters, likely on SMOTE data):**
    *   Precision: 0.93
    *   Recall: 0.86
    *   F1-score: 0.89
    *   ROC AUC: 0.99
    *   Precision-Recall AUC: 0.84

## Discussion of Results

*   **Impact of SMOTE:** Model performance (especially precision for some models) generally decreased after applying SMOTE. This might be because SMOTE, by interpolating, can generate synthetic samples that cause the model to overfit to these synthetic points or amplify noise if the minority class has noisy data points.
*   **Best Performing Models:** XGBoost and Random Forest consistently showed the most promising results, especially in handling imbalanced data.
    *   **XGBoost Mechanisms:** Allows weighted loss functions (penalizing misclassification of minority class more) and supports early stopping/fine-grained tuning.
    *   **Random Forest Mechanisms:** Creates bootstrapped samples (ensuring minority class representation in some trees) and can use balanced class weights.

## Defending Against Adversarial Attacks (Model Extraction)

*   **Threat:** Model extraction attacks, where an adversary queries the ML model to infer its parameters or replicate functionality.
*   **Defense:** Fully Homomorphic Encryption (FHE).
    *   **How FHE Works:** Allows ML models to process encrypted inputs without decryption, ensuring data and model parameters remain secure. Users encrypt inputs; the server processes encrypted data; results are returned encrypted. Only the user with the decryption key can interpret the output.
    *   **Impact on Attack Surface:** Encrypted responses are useless to attackers without the key, reducing model reconstruction risk.
    *   **Challenges:** Computational overhead and slower operations.
    *   **Future:** Advancements are making FHE more efficient and feasible, positioning it as a key technology for securing ML models and safeguarding user privacy.

## Conclusion

*   Five ML models were evaluated for fraud detection. XGBoost and Random Forest were most promising.
*   XGBoost achieved the best overall performance. After K-fold cross-validation and fine-tuning, its effectiveness was confirmed.
*   FHE is proposed as a robust defense against adversarial attacks like model extraction, enhancing data privacy.

## References

[1] Trenton's Blog on Fraud Detection: [https://trenton3983.github.io/posts/fraud-detection-python/](https://trenton3983.github.io/posts/fraud-detection-python/)
[2] Al-Hashedi, K. G., & Magalingam, P. (2021). Financial fraud detection applying data mining techniques: A comprehensive review from 2009 to 2019. *Computer Science Review, 40*, 100402. [https://doi.org/10.1016/j.cosrev.2021.100402](https://doi.org/10.1016/j.cosrev.2021.100402)
[3] BankSim Dataset on Kaggle: [https://www.kaggle.com/datasets/ealaxi/banksim1/data](https://www.kaggle.com/datasets/ealaxi/banksim1/data)
[4] TenSEAL FHE Library: [https://github.com/OpenMined/TenSEAL](https://github.com/OpenMined/TenSEAL)
[5] OpenFHE Development Documentation: [https://openfhe-development.readthedocs.io/en/latest/](https://openfhe-development.readthedocs.io/en/latest/)
