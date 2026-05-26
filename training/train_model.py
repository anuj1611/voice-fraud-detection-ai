import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


REPORTS_DIR = Path("training/reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


X = np.load("training/X_features.npy")
y = np.load("training/y_labels.npy")

print("Dataset shape:", X.shape)
print("Labels shape:", y.shape)

# split (70/15/15)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)

print("Train shape:", X_train.shape)
print("Validation shape:", X_val.shape)
print("Test shape:", X_test.shape)


model = models.Sequential([
    layers.Input(shape=(128, 128, 1)),

    layers.Conv2D(32, (3,3), padding="same"),
    layers.BatchNormalization(),
    layers.Activation("relu"),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(64, (3,3), padding="same"),
    layers.BatchNormalization(),
    layers.Activation("relu"),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(128, (3,3), padding="same"),
    layers.BatchNormalization(),
    layers.Activation("relu"),
    layers.MaxPooling2D((2,2)),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()



early_stop = EarlyStopping(patience=3, restore_best_weights=True)
checkpoint = ModelCheckpoint(
    "saved_model/voice_ai_detector.h5",
    save_best_only=True
)



history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[early_stop, checkpoint]
)



test_loss, test_acc = model.evaluate(X_test, y_test)
y_pred_prob = model.predict(X_test).ravel()
y_pred = (y_pred_prob >= 0.5).astype(int)


def save_training_curves(training_history):
    epochs = range(1, len(training_history.history["accuracy"]) + 1)

    plt.figure(figsize=(10, 4.5))
    plt.plot(epochs, training_history.history["accuracy"], label="Train Accuracy", linewidth=2)
    plt.plot(epochs, training_history.history["val_accuracy"], label="Validation Accuracy", linewidth=2)
    plt.title("Training vs Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "accuracy_curve.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 4.5))
    plt.plot(epochs, training_history.history["loss"], label="Train Loss", linewidth=2)
    plt.plot(epochs, training_history.history["val_loss"], label="Validation Loss", linewidth=2)
    plt.title("Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "loss_curve.png", dpi=150)
    plt.close()


def save_evaluation_plots(y_true, y_prob, y_binary):
    cm = confusion_matrix(y_true, y_binary)
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["Human", "AI"],
        yticklabels=["Human", "AI"],
        ylabel="True Label",
        xlabel="Predicted Label",
        title="Confusion Matrix"
    )

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"), ha="center", va="center")

    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="#0ea5e9", linewidth=2, label=f"ROC AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.grid(alpha=0.3)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "roc_curve.png", dpi=150)
    plt.close()

    report_text = classification_report(
        y_true,
        y_binary,
        target_names=["Human", "AI"],
        digits=4
    )
    metrics_summary = [
        f"Test Loss: {test_loss:.4f}",
        f"Test Accuracy: {test_acc:.4f}",
        f"ROC AUC: {roc_auc:.4f}",
        "",
        "Classification Report:",
        report_text,
    ]
    (REPORTS_DIR / "metrics_summary.txt").write_text("\n".join(metrics_summary), encoding="utf-8")


save_training_curves(history)
save_evaluation_plots(y_test, y_pred_prob, y_pred)


final_train_acc = history.history['accuracy'][-1]
final_val_acc = history.history['val_accuracy'][-1]

print(f"\nFinal Training Accuracy: {final_train_acc:.4f}")
print(f"Final Validation Accuracy: {final_val_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
print(f"\nSaved reports to: {REPORTS_DIR.resolve()}")
