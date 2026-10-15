import numpy as np
import os
import json
from datetime import datetime

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks, optimizers
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve, f1_score
)

print(f"TensorFlow: {tf.__version__}")
print(f"GPUs: {tf.config.list_physical_devices('GPU')}")

# Data
data = np.load("./augmented_dataset/lstm_dataset_v6.npz")
X_train, y_train = data["X_train"][..., np.newaxis], data["y_train"]
X_val, y_val = data["X_val"][..., np.newaxis], data["y_val"]
X_test, y_test = data["X_test"][..., np.newaxis], data["y_test"]

input_shape = X_train.shape[1:]  # (256, 256, 1)
print(f"Train: {X_train.shape[0]} ({np.mean(y_train):.1%} FRB)")
print(f"Val:   {X_val.shape[0]} ({np.mean(y_val):.1%} FRB)")
print(f"Test:  {X_test.shape[0]} ({np.mean(y_test):.1%} FRB)")

# Parameters
batch_size = 128
epochs = 150
learning_rate = 1e-3
early_stop_patience = 30
lr_patience = 10
save_dir = "./models/cnn_v1"
os.makedirs(save_dir, exist_ok=True)

# Model
model = keras.Sequential([
    layers.Input(shape=input_shape, name='input_data'),
    layers.Conv2D(4, (3, 3), activation='relu', padding='same', name='conv_0'),
    layers.MaxPooling2D((2, 2), name='pool_0'),
    layers.Conv2D(8, (3, 3), activation='relu', padding='same', name='conv_1'),
    layers.MaxPooling2D((2, 2), name='pool_1'),
    layers.Conv2D(8, (3, 3), activation='relu', padding='same', name='conv_2'),
    layers.MaxPooling2D((2, 2), name='pool_2'),
    layers.Flatten(name='flatten'),
    layers.Dense(32, activation='relu', name='dense_0'),
    layers.Dropout(0.3, name='drop_0'),
    layers.Dense(1, activation='sigmoid', name='output'),
])

model.summary()

for layer in model.layers:
    if 'flatten' in layer.name:
        flatten_dim = int(np.prod(layer.output_shape[1:]))
        print(f"\nFlatten dimension: {flatten_dim}")

# Class weights
n_pos = np.sum(y_train == 1)
n_neg = np.sum(y_train == 0)
class_weight = {0: len(y_train) / (2 * n_neg), 1: len(y_train) / (2 * n_pos)}
print(f"Class weights: {class_weight}")

# Compile
model.compile(
    optimizer=optimizers.Adam(learning_rate=learning_rate),
    loss='binary_crossentropy',
    metrics=['accuracy'],
)

# Train
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=epochs,
    batch_size=batch_size,
    class_weight=class_weight,
    callbacks=[
        callbacks.EarlyStopping(
            monitor='val_loss', patience=early_stop_patience,
            restore_best_weights=True, verbose=1
        ),
        callbacks.ModelCheckpoint(
            os.path.join(save_dir, 'best_model.keras'),
            monitor='val_loss', save_best_only=True, verbose=0
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5,
            patience=lr_patience, min_lr=1e-6, verbose=1
        ),
    ],
    verbose=1,
)

best_epoch = int(np.argmin(history.history['val_loss']))
print(f"\nBest epoch: {best_epoch + 1}")

# Evaluate
y_pred_prob = model.predict(X_test, batch_size=batch_size).flatten()
y_pred = (y_pred_prob >= 0.5).astype(int)

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
cm = confusion_matrix(y_test, y_pred)
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)
prec_vals, rec_vals, _ = precision_recall_curve(y_test, y_pred_prob)
pr_auc = auc(rec_vals, prec_vals)
f1 = f1_score(y_test, y_pred)

print(f"\nTest Loss:     {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
print(f"ROC AUC:       {roc_auc:.4f}")
print(f"PR AUC:        {pr_auc:.4f}")
print(f"F1 Score:      {f1:.4f}")
print(f"\n{classification_report(y_test, y_pred, target_names=['Noise', 'FRB'])}")
print(f"Confusion Matrix:  TN={cm[0,0]}  FP={cm[0,1]}  FN={cm[1,0]}  TP={cm[1,1]}")

print("\nRecall vs Threshold:")
for thresh in [0.5, 0.4, 0.35, 0.3, 0.25, 0.2]:
    yp = (y_pred_prob >= thresh).astype(int)
    cm_t = confusion_matrix(y_test, yp)
    rec = cm_t[1, 1] / (cm_t[1, 0] + cm_t[1, 1])
    prec = cm_t[1, 1] / (cm_t[0, 1] + cm_t[1, 1]) if (cm_t[0, 1] + cm_t[1, 1]) > 0 else 0
    print(f"  thresh={thresh:.2f}  recall={rec:.3f}  precision={prec:.3f}  "
          f"missed={cm_t[1,0]}  false_alarms={cm_t[0,1]}")

# Save
model.save(os.path.join(save_dir, 'model.keras'))
model.save(os.path.join(save_dir, 'model.h5'))

results = {
    'total_params': int(model.count_params()),
    'flatten_dim': flatten_dim,
    'best_epoch': best_epoch + 1,
    'test_loss': float(test_loss),
    'test_acc': float(test_acc),
    'roc_auc': float(roc_auc),
    'pr_auc': float(pr_auc),
    'f1_score': float(f1),
    'confusion_matrix': cm.tolist(),
    'timestamp': datetime.now().isoformat(),
}
with open(os.path.join(save_dir, 'results.json'), 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nSaved to {save_dir}/")
