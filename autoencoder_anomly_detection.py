# Import libraries
import pandas as pd                  #loading and handling data
import numpy as np                   #numerical computations
import matplotlib.pyplot as plt      #visual plotting the reconstruction error

#preprocessing, model evaluation, and data splitting
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

#building the neural network
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

# Step 1: Load and preprocess credit card dataset
df = pd.read_csv('creditcard.csv')

# Normalize 'Amount' column to mean 0, std 1
df['Amount'] = StandardScaler().fit_transform(df[['Amount']])

# Drop 'Time' column
df = df.drop(['Time'], axis=1)

# Separate features and target labels
X = df.drop('Class', axis=1)  # Features
y = df['Class']               # Labels: 0 (normal), 1 (fraud)

# Step 2: Train-test split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

# Filter the training data to only include normal transactions (Class == 0)
X_train = X_train[y_train == 0]

# Step 3: Build the autoencoder model
input_dim = X_train.shape[1] 
input_layer = Input(shape=(input_dim,))

# Encoder: compresses data 
encoded = Dense(14, activation='relu')(input_layer)
encoded = Dense(7, activation='relu')(encoded)

# Decoder: reconstructs compressed data
decoded = Dense(14, activation='relu')(encoded)
decoded = Dense(input_dim, activation='sigmoid')(decoded)

# Combine data into full autoencoder model
autoencoder = Model(inputs=input_layer, outputs=decoded)

# Compile the model with optimizer and loss function
autoencoder.compile(optimizer='adam', loss='mse')  # Use MSE to measure reconstruction error

# Step 4: Train the autoencoder
history = autoencoder.fit(
    X_train,                
    X_train,
    epochs=20,              # Number of training iterations
    batch_size=32,          # Samples per training step
    shuffle=True,           # Shuffle data in each epoch
    validation_split=0.1,   # Reserve 10% of training data for validation
    verbose=1               
)

# Step 5: Make predictions on test data
X_test_pred = autoencoder.predict(X_test)

# Calculate reconstruction error per transaction
recon_error = np.mean(np.square(X_test - X_test_pred), axis=1)

# Step 6: Set a threshold to classify anomalies using the 95th percentile of reconstruction error 
threshold = np.percentile(recon_error, 95)
print("Threshold:", threshold)

# Label normal (0) and fraud (1) if error > threshold 
y_pred = [1 if e > threshold else 0 for e in recon_error]

# Step 7: Evaluate the model
print(confusion_matrix(y_test, y_pred))

# Show precision
print(classification_report(y_test, y_pred, digits=4))

# Step 8: Plot reconstruction error distribution
plt.hist(recon_error, bins=50)  # Plot histogram of reconstruction errors
plt.axvline(threshold, color='red', linestyle='--', label='Threshold')  # threshold line
plt.xlim(0, np.max(recon_error) + 0.01) #fixed upper bound to better visualize data
plt.title("Reconstruction Error Distribution")
plt.xlabel("Reconstruction Error")
plt.ylabel("Frequency")
plt.legend()
plt.show()
