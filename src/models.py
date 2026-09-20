import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


def build_cnn(img_size=48, channels=1, embedding_dim=64):
    """Builds the 12,873-parameter lightweight CNN baseline model."""
    inputs = layers.Input(shape=(img_size, img_size, channels))

    def basic_block(x, filters):
        x = layers.Conv2D(filters, 3, padding="same", activation="elu")(x)
        x = layers.Conv2D(filters, 3, padding="same", activation="elu")(x)
        x = layers.Conv2D(filters, 3, padding="same", activation=None)(x)
        x = layers.BatchNormalization(momentum=0.99, epsilon=1e-3)(x)
        x = layers.Activation("elu")(x)
        x = layers.MaxPooling2D(pool_size=2)(x)
        return x

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


def train_cnn(
    X_train,
    y_train,
    X_val,
    y_val,
    img_size=48,
    channels=1,
    lr=1e-3,
    epochs=15,
    batch_size=32,
    use_class_weights=True,
    seed=42,
    verbose=1,
):
    """Compiles and trains the lightweight CNN on training split."""
    tf.random.set_seed(seed)
    full_model, embedding_model = build_cnn(img_size, channels)

    full_model.compile(
        optimizer=optimizers.Nadam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )

    class_weight = None
    if use_class_weights:
        n_pos = int(y_train.sum())
        n_neg = int(len(y_train) - n_pos)
        class_weight = {0: 1.0, 1: n_neg / max(n_pos, 1)}

    early_stop = callbacks.EarlyStopping(
        monitor="val_auc", mode="max", patience=5, restore_best_weights=True
    )

    history = full_model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight,
        callbacks=[early_stop],
        verbose=verbose,
    )
    return full_model, embedding_model, history


def cnn_predict_proba(full_model, X, batch_size=256):
    """Generates CNN probability predictions."""
    return full_model.predict(X, batch_size=batch_size, verbose=0).reshape(-1)


def extract_embeddings(embedding_model, X, batch_size=256):
    """Extracts 64-dimensional CNN feature embeddings."""
    return embedding_model.predict(X, batch_size=batch_size, verbose=0)


def train_classical_base_models(
    X_train_feat,
    y_train,
    seed=42,
    rf_estimators=300,
    svm_kernel="rbf",
    ann_layers=(128, 64),
    ann_max_iter=500,
):
    """Trains SVM, Random Forest, and ANN classifiers on CNN feature embeddings."""
    scaler = StandardScaler().fit(X_train_feat)
    X_scaled = scaler.transform(X_train_feat)

    svm = SVC(kernel=svm_kernel, probability=True, class_weight="balanced", random_state=seed)
    svm.fit(X_scaled, y_train)

    rf = RandomForestClassifier(
        n_estimators=rf_estimators, class_weight="balanced", random_state=seed, n_jobs=-1
    )
    rf.fit(X_train_feat, y_train)

    ann = MLPClassifier(
        hidden_layer_sizes=ann_layers, max_iter=ann_max_iter, random_state=seed, early_stopping=True
    )
    ann.fit(X_scaled, y_train)

    return {"svm": (svm, scaler), "random_forest": (rf, None), "ann": (ann, scaler)}


def classical_predict_proba(model, scaler, X_feat):
    """Generates probability predictions from classical models."""
    X_in = scaler.transform(X_feat) if scaler is not None else X_feat
    return model.predict_proba(X_in)[:, 1]


def get_all_base_probabilities(
    cnn_full_model, classical_models, X_images, X_feat, base_models=["cnn", "svm", "random_forest", "ann"]
):
    """Computes probability matrix for all base learners."""
    probs = {"cnn": cnn_predict_proba(cnn_full_model, X_images)}
    for name in ["svm", "random_forest", "ann"]:
        if name in classical_models:
            model, scaler = classical_models[name]
            probs[name] = classical_predict_proba(model, scaler, X_feat)

    matrix = np.column_stack([probs[name] for name in base_models])
    return matrix, probs
