# Credit Card Fraud Detection Using Autoencoders

This is an unsupervised deep learning project that uses an autoencoder neural network to detect fraudulent credit card transactions. The model is trained only on normal transactions, and detects fraud based on reconstruction error.

---

## Dataset

- Source: [Kaggle – Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- Contains 284,807 transactions, of which 492 are fraudulent (Class = 1).
- Features are numerical results of PCA transformation for confidentiality.

---

## Approach

- This Autoencoder is trained on normal (non-fraudulent) transactions only. 
- Fraudulent transactions result in **higher reconstruction error**, and are flagged as anomalies.

---

## Model Architecture

- **Input Layer**: 30 features
- **Encoder**:
  - Dense(14, activation='relu')
  - Dense(7, activation='relu')
- **Decoder**:
  - Dense(14, activation='relu')
  - Dense(30, activation='sigmoid')

---

## Technologies Used

- Python
- TensorFlow / Keras
- Pandas, NumPy
- Scikit-learn
- Matplotlib


