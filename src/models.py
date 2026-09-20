"""Ensemble members: a lightweight CNN and three classical heads (SVM / RF / ANN).

IMPORTANT design fact (matches the README diagram): the SVM, Random Forest and ANN are NOT
trained on raw pixels. They are trained on the 64-D penultimate embedding of the *same* CNN
(trained on the training partition). The four members are therefore correlated, not independent.
TensorFlow is imported lazily so the classical/ensemble code can be tested without it.
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .dataset import IMG_CHANNELS, IMG_SIZE, RANDOM_SEED

# ----------------------------------------------------------------------------- config
ENSEMBLE_BASE_MODELS = ["cnn", "svm", "random_forest", "ann"]
EMBEDDING_DIM = 64
CNN_EPOCHS = 15
CNN_BATCH_SIZE = 32
CNN_LEARNING_RATE = 1e-3
USE_CLASS_WEIGHTS = True          # CNN: weight of the hotspot class = n_NHS / n_HS of the training split
EARLY_STOP_PATIENCE = 5           # monitor = val_auc on the validation split
RANDOM_FOREST_N_ESTIMATORS = 300
SVM_KERNEL = "rbf"
ANN_HIDDEN_LAYERS = (128, 64)
ANN_MAX_ITER = 500                # NOTE: sklearn's MLPClassifier has no class weighting


# ----------------------------------------------------------------------------- CNN
def build_cnn(img_size=IMG_SIZE, channels=IMG_CHANNELS, embedding_dim=EMBEDDING_DIM):
    """Two 'basic blocks' (3x Conv3x3 [ELU, ELU, linear] + BatchNorm + ELU + MaxPool2) with 16 and 32
    filters, then Flatten -> Dropout(0.3) -> Dense(64, ReLU) [embedding] -> Dropout(0.3) -> Dense(1, sigmoid).
    ~323 k parameters (NOT the 12,873-parameter network of the reference work)."""
    from tensorflow.keras import layers, models

    inputs = layers.Input(shape=(img_size, img_size, channels))

    def basic_block(x, filters):
        x = layers.Conv2D(filters, 3, padding="same", activation="elu")(x)
        x = layers.Conv2D(filters, 3, padding="same", activation="elu")(x)
        x = layers.Conv2D(filters, 3, padding="same", activation=None)(x)
        x = layers.BatchNormalization(momentum=0.99, epsilon=1e-3)(x)
        x = layers.Activation("elu")(x)
        return layers.MaxPooling2D(pool_size=2)(x)

    x = basic_block(inputs, 16)
    x = basic_block(x, 32)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.3)(x)
    embedding = layers.Dense(embedding_dim, activation="relu", name="embedding")(x)
    x = layers.Dropout(0.3)(embedding)
    output = layers.Dense(1, activation="sigmoid", name="output")(x)
    full_model = models.Model(inputs, output, name="lightweight_cnn")
    embedding_model = models.Model(inputs, embedding, name="cnn_embedding_extractor")
    return full_model, embedding_model


def train_cnn(X_train, y_train, X_val, y_val, seed=RANDOM_SEED, verbose=1):
    """Train the CNN on the training split. The validation split is used for early stopping
    (restore_best_weights on val_auc) and later also to fit the combiners - see README limitations."""
    import tensorflow as tf
    from tensorflow.keras import callbacks, optimizers

    tf.random.set_seed(seed)
    full_model, embedding_model = build_cnn()
    full_model.compile(
        optimizer=optimizers.Nadam(learning_rate=CNN_LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(name="auc"),
                 tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall")],
    )
    class_weight = None
    if USE_CLASS_WEIGHTS:
        n_pos = int(y_train.sum())
        n_neg = int(len(y_train) - n_pos)
        class_weight = {0: 1.0, 1: n_neg / max(n_pos, 1)}
    early_stop = callbacks.EarlyStopping(monitor="val_auc", mode="max",
                                         patience=EARLY_STOP_PATIENCE, restore_best_weights=True)
    history = full_model.fit(X_train, y_train, validation_data=(X_val, y_val),
                             epochs=CNN_EPOCHS, batch_size=CNN_BATCH_SIZE,
                             class_weight=class_weight, callbacks=[early_stop], verbose=verbose)
    return full_model, embedding_model, history


def build_joint_model(full_model):
    """One forward pass returning [embedding, probability] (used for latency measurement only)."""
    import tensorflow as tf
    return tf.keras.Model(full_model.input, [full_model.get_layer("embedding").output, full_model.output])


def cnn_predict_proba(full_model, X, batch_size=256):
    return full_model.predict(X, batch_size=batch_size, verbose=0).reshape(-1)


def extract_embeddings(embedding_model, X, batch_size=256):
    return embedding_model.predict(X, batch_size=batch_size, verbose=0)


# ----------------------------------------------------------------------------- classical heads
def train_classical_base_models(X_train_feat, y_train, seed=RANDOM_SEED):
    """SVM (RBF, Platt probabilities, balanced class weights), Random Forest (balanced class
    weights) and ANN (sklearn MLP, NO class weights) on the CNN embeddings of the training split."""
    scaler = StandardScaler().fit(X_train_feat)
    X_scaled = scaler.transform(X_train_feat)

    svm = SVC(kernel=SVM_KERNEL, probability=True, class_weight="balanced", random_state=seed)
    svm.fit(X_scaled, y_train)

    rf = RandomForestClassifier(n_estimators=RANDOM_FOREST_N_ESTIMATORS, class_weight="balanced",
                                random_state=seed, n_jobs=-1)
    rf.fit(X_train_feat, y_train)

    ann = MLPClassifier(hidden_layer_sizes=ANN_HIDDEN_LAYERS, max_iter=ANN_MAX_ITER,
                        random_state=seed, early_stopping=True)
    ann.fit(X_scaled, y_train)
    return {"svm": (svm, scaler), "random_forest": (rf, None), "ann": (ann, scaler)}


def classical_predict_proba(model, scaler, X_feat):
    X_in = scaler.transform(X_feat) if scaler is not None else X_feat
    return model.predict_proba(X_in)[:, 1]


def get_all_base_probabilities(cnn_full_model, classical_models, X_images, X_feat,
                               base_models=ENSEMBLE_BASE_MODELS):
    """Return (matrix [n_samples x n_models] in `base_models` order, dict name -> probability vector)."""
    probs = {"cnn": cnn_predict_proba(cnn_full_model, X_images)}
    for name in ("svm", "random_forest", "ann"):
        model, scaler = classical_models[name]
        probs[name] = classical_predict_proba(model, scaler, X_feat)
    matrix = np.column_stack([probs[name] for name in base_models])
    return matrix, probs
