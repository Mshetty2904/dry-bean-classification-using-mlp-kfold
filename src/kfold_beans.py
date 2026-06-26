# ================================
# IMPORTS
# ================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# Scikit-Learn for preprocessing, splitting, and metrics
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import (
    confusion_matrix, 
    ConfusionMatrixDisplay,
    precision_score, 
    recall_score, 
    f1_score,
    precision_recall_fscore_support
)

# TensorFlow / Keras for the Neural Network
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD

# ================================
# LOAD AND PREPROCESS DATA
# ================================
# 1. Load Data
filename = "/content/Dry_Bean_Dataset.csv"
try:
    df = pd.read_csv(filename)
except FileNotFoundError:
    print(f"Error: {filename} not found. Please ensure it is in the same directory.")
    exit()

# Separate features (X) and target (y)
X = df.iloc[:, :-1].values
y_raw = df.iloc[:, -1].values

# 2. Encode Labels (String -> Int)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw)
labels = list(label_encoder.classes_)
num_classes = len(labels)

# ================================
# HYPERPARAMETERS & SETUP
# ================================
n_folds = 5
l_rate  = 0.05
n_epoch = 125
h1      = 64
h2      = 32
h3      = 16

# Set seeds for reproducibility
np.random.seed(1)
try:
    import tensorflow as tf
    tf.random.set_seed(1)
except ImportError:
    pass

# ================================
# MODEL DEFINITION
# ================================
def create_model(input_dim):
    """Builds and compiles the Keras MLP model."""
    model = Sequential([
        Dense(h1, activation='sigmoid', input_dim=input_dim),
        Dense(h2, activation='sigmoid'),
        Dense(h3, activation='sigmoid'),
        Dense(num_classes, activation='softmax') # Softmax for multi-class classification
    ])
    
    # Using SGD to match the manual backprop behavior of the original script
    optimizer = SGD(learning_rate=l_rate)
    
    # sparse_categorical_crossentropy automatically handles integer encoded labels
    model.compile(optimizer=optimizer, 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    return model

# ================================
# K-FOLD EVALUATION
# ================================
kf = KFold(n_splits=n_folds, shuffle=True, random_state=1)

scores = []
all_errors = []
all_accuracies = []
global_act = []
global_pred = []

print("Starting K-Fold training with Keras...")

fold_no = 1
for train_index, test_index in kf.split(X):
    print(f"Training Fold {fold_no}...")
    
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # Scale data inside the loop to prevent data leakage from test set
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize and train model
    model = create_model(input_dim=X_train.shape[1])
    
    # Train the model (batch_size=32 is standard, use 1 to match strict row-by-row SGD)
    history = model.fit(X_train_scaled, y_train, 
                        epochs=n_epoch, 
                        batch_size=32, 
                        verbose=0)
    
    # Store metrics per epoch for averaging later
    all_errors.append(history.history['loss'])
    all_accuracies.append([acc * 100 for acc in history.history['accuracy']])
    
    # Predictions
    # model.predict outputs probabilities; np.argmax grabs the highest probability class index
    y_pred_probs = model.predict(X_test_scaled, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Collect metrics
    fold_accuracy = np.mean(y_pred == y_test) * 100
    scores.append(fold_accuracy)
    
    global_act.extend(y_test)
    global_pred.extend(y_pred)
    
    fold_no += 1

# ================================
# GLOBAL METRICS
# ================================
print("\nScores per fold:", [f"{score:.2f}%" for score in scores])
print(f"Mean Accuracy: {np.mean(scores):.2f}%")

precision = precision_score(global_act, global_pred, average='macro', zero_division=0)
recall = recall_score(global_act, global_pred, average='macro', zero_division=0)
f1 = f1_score(global_act, global_pred, average='macro', zero_division=0)

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# ================================
# PLOTTING
# ================================
# 1 & 2. Average Loss and Accuracy over Epochs
avg_error = np.mean(all_errors, axis=0)
avg_acc = np.mean(all_accuracies, axis=0)

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(range(1, n_epoch+1), avg_error)
plt.xlabel("Epochs")
plt.ylabel("Cross-Entropy Loss") # Replaced SSE with Cross-Entropy
plt.title("Avg Loss vs Epoch")

plt.subplot(1, 3, 2)
plt.plot(range(1, n_epoch+1), avg_acc, color='orange')
plt.xlabel("Epochs")
plt.ylabel("Accuracy (%)")
plt.title("Avg Training Accuracy vs Epoch")

# 3. Confusion Matrix
cm = confusion_matrix(global_act, global_pred)
ax3 = plt.subplot(1, 3, 3)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap=plt.cm.Blues, ax=ax3, values_format='d', xticks_rotation=45)
ax3.set_title('Confusion Matrix (All Folds)')

plt.tight_layout()
plt.show()

# 4. Class Distribution
class_counts = Counter(y_raw)
plt.figure(figsize=(8,5))
plt.bar([str(k) for k in class_counts.keys()], class_counts.values())
plt.xlabel("Bean Type")
plt.ylabel("Count")
plt.title("Class Distribution (Actual Dataset)")
plt.xticks(rotation=45)
plt.show()

# 5. K-Fold Accuracy Curve
plt.figure(figsize=(7,5))
plt.plot(range(1, len(scores)+1), scores, marker='o')
plt.xlabel("Fold")
plt.ylabel("Accuracy (%)")
plt.title("K-Fold Accuracy")
plt.grid(True)
plt.show()

# 6. Predicted Counts per Class
pred_counts = Counter(global_pred)
plt.figure(figsize=(8,5))
plt.bar([labels[i] for i in pred_counts.keys()], pred_counts.values())
plt.xlabel("Bean Class")
plt.ylabel("Predicted Count")
plt.title("Predictions per Class")
plt.xticks(rotation=45)
plt.show()

# 7. Class-wise Metrics
prec, rec, f1_cls, _ = precision_recall_fscore_support(global_act, global_pred, zero_division=0)
x = range(num_classes)

plt.figure(figsize=(10,5))
plt.plot(x, prec, marker='o', label='Precision')
plt.plot(x, rec, marker='s', label='Recall')
plt.plot(x, f1_cls, marker='^', label='F1')
plt.xticks(x, labels, rotation=45)
plt.ylabel("Score")
plt.title("Class-wise Metrics")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
