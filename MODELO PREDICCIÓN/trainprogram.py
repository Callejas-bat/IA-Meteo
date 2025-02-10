import tensorflow as tf
from tensorflow.keras import layers, regularizers, callbacks
from tensorflow.keras.layers import BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.utils import to_categorical
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize
from imblearn.over_sampling import SMOTE
import subprocess


def plot_history(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

def evaluate_model(model, x_test, y_test, label_encoder):
    # Evaluación en el conjunto de test
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

    # Predicción en el conjunto de test
    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)

    # Matriz de confusión
    cm = confusion_matrix(y_true, y_pred_classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
    disp.plot(cmap=plt.cm.Blues)
    plt.show()

    # Informe de clasificación
    report = classification_report(y_true, y_pred_classes, target_names=label_encoder.classes_)
    print(report)

    # ROC AUC
    y_test_binarized = label_binarize(y_true, classes=np.arange(len(label_encoder.classes_)))
    roc_auc = roc_auc_score(y_test_binarized, y_pred, multi_class="ovo")
    print(f"ROC AUC Score: {roc_auc:.4f}")

def main():

    tf.keras.backend.clear_session()

    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    data = pd.read_json("invgmodified.json")

    datat = data.T

    # Limpiar datos
    datat = datat.dropna()
    datat[['HUM', 'LUZ', 'PRES', 'TEMP', 'VEL']] = datat[['HUM', 'LUZ', 'PRES', 'TEMP', 'VEL']].apply(pd.to_numeric, errors='coerce')
    datat = datat.dropna()

    # Verificar datos
    print(datat.describe())
    print(datat.isnull().sum())

    x = datat[["HUM", "LUZ", "PRES", "TEMP", "VEL"]].values
    y = datat["WEATHER"].values

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    y_categorical = to_categorical(y_encoded)
    joblib.dump(label_encoder, 'label_encoder.pkl')

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    joblib.dump(scaler, 'scaler.pkl')

    x_scaled = np.array(x_scaled, dtype=np.float32)
    y_categorical = np.array(y_categorical, dtype=np.float32)

    x_train, x_test, y_train, y_test = train_test_split(x_scaled, y_categorical, test_size=0.2, random_state=42)

    smote = SMOTE()
    x_train_resampled, y_train_resampled = smote.fit_resample(x_train, y_train.argmax(axis=1))
    y_train_resampled = tf.keras.utils.to_categorical(y_train_resampled, num_classes=y_categorical.shape[1])

    # Calcular class_weights
    class_weights = class_weight.compute_class_weight('balanced',
                                                      classes=np.unique(y_train_resampled.argmax(axis=1)),
                                                      y=y_train_resampled.argmax(axis=1))

    class_weights = {i: class_weights[i] for i in range(len(class_weights))}

    model = tf.keras.Sequential([
        layers.Input(shape=(x_train.shape[1],)),

        layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(0.001), kernel_initializer='he_normal'),
        BatchNormalization(),
        layers.Dropout(0.3),

        layers.Dense(256, activation="relu", kernel_regularizer=regularizers.l2(0.001), kernel_initializer='he_normal'),
        BatchNormalization(),
        layers.Dropout(0.3),

        layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(0.0001), kernel_initializer='he_normal'),
        BatchNormalization(),
        layers.Dropout(0.3),

        layers.Dense(y_categorical.shape[1], activation="softmax")
    ])

    # Learning rate scheduler
    lr_scheduler = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)

    # Early stopping
    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1)

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

    history = model.fit(x_train, y_train, class_weight=class_weights, epochs=200, batch_size=128, validation_split=0.2, callbacks=[lr_scheduler, early_stopping])

    model.export("weather_prediction_saved_model")
    # model.save("model.h5")

    # Convertir el modelo a TensorFlow.js
    subprocess.run(["tensorflowjs_converter", "--input_format=tf_saved_model", "--output_format=tfjs_graph_model",
                    "weather_prediction_saved_model", "weather_prediction_tfjs"])

    evaluate_model(model, x_test, y_test, label_encoder)

    plot_history(history)

if __name__ == "__main__":
    main()