"""AMR-EnsembleNet reproduction (Part C.3).

Attribution: Siddiqui & Tarannum (2025), arXiv:2509.23552.
Source repository: github.com/Saiful185/AMR-EnsembleNet

Ported from `giessen-cross-validation.ipynb`'s own cell explicitly labelled
"CELL 6 (OFFICIAL) — AMR-EnsembleNet REPRODUCTION USING AUTHORS' EXACT CODE",
which states its own provenance as code extracted verbatim from the authors'
`AMR_Project_1D_CNN_Tuning.ipynb` (CNN architecture + training loop) and
`AMR_Project_XGBoost_Baseline_ALL.ipynb` (XGBoost hyperparameters), applied
to this project's own stratified split for a fair comparison.

Suggested thesis citation (from the source notebook, reused verbatim):
    "We reproduce AMR-EnsembleNet (Siddiqui & Tarannum, 2025) using the
    authors' official implementation from
    https://github.com/Saiful185/AMR-EnsembleNet. The CNN architecture,
    hyperparameters, XGBoost configuration, and soft voting ensemble logic
    are used verbatim from their repository. The only modification is the
    train/validation/test partitioning, which uses our stratified random
    split to enable fair comparison against our proposed fusion strategies."

**Not executed or numerically re-verified in this refactor.** This
environment has no `tensorflow` installed (`ModuleNotFoundError` confirmed
directly), so the CNN half of this module could not be run end-to-end here.
The code below is a faithful, structure-preserving transcription of the
notebook cell (architecture, hyperparameters, training callbacks, ensemble
logic all unchanged) with the hardcoded Kaggle paths removed (B11) -- it has
NOT been used to regenerate any number. Treat any PR-AUC/MCC figure
attributed to "AMR-EnsembleNet reproduction" in the existing thesis text as
coming from the original notebook run, not from this file, until someone
re-runs this module in an environment with TensorFlow/GPU available.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .metrics import best_threshold_by_f2, compute_metrics

VOCAB_SIZE = 5          # N:0, A:1, G:2, C:3, T:4 -- authors' encoding
EMBEDDING_DIM = 64
DROPOUT_RATE = 0.45
CONV_DROPOUT_RATE = 0.15
EPOCHS = 120
BATCH_SIZE = 32
LEARNING_RATE = 1e-3

XGB_PARAMS = dict(
    objective="binary:logistic",
    eval_metric="auc",
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.7,
    colsample_bytree=0.7,
    n_jobs=-1,
    random_state=42,
    early_stopping_rounds=50,
)


def build_cnn1d_model_extended(maxlen: int, vocab_size: int = VOCAB_SIZE,
                                embed_dim: int = EMBEDDING_DIM,
                                num_classes_binary: int = 1,
                                conv_dropout_rate: float = CONV_DROPOUT_RATE,
                                dense_dropout_rate: float = DROPOUT_RATE):
    """Authors' exact 6-block 1D CNN (verbatim, see module docstring).
    Requires `tensorflow` -- imported locally, not at module load time, so
    the rest of this package (and its tests) work without it installed.
    """
    from tensorflow import keras
    from tensorflow.keras import layers, regularizers

    inputs = keras.Input(shape=(maxlen,), name="snp_input_sequence")
    x = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim,
                          input_length=maxlen, name="embedding_layer")(inputs)

    block_specs = [
        (128, 7, "maxpool"), (128, 7, "maxpool"),
        (64, 5, "maxpool"), (64, 5, "maxpool"),
        (32, 3, "dropout"), (32, 3, "dropout"),
    ]
    for i, (filters, kernel, tail) in enumerate(block_specs, start=1):
        x = layers.Conv1D(filters=filters, kernel_size=kernel, padding="same",
                           kernel_regularizer=regularizers.l2(0.001), name=f"conv1d_{i}")(x)
        x = layers.BatchNormalization(name=f"batchnorm_{i}")(x)
        x = layers.Activation("relu", name=f"relu_{i}")(x)
        if tail == "maxpool":
            x = layers.MaxPooling1D(pool_size=2, name=f"maxpool_{i}")(x)
        else:
            x = layers.Dropout(conv_dropout_rate, name=f"conv_dropout_{i}")(x)

    x = layers.GlobalMaxPooling1D(name="global_max_pooling")(x)
    x = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(0.001),
                      name="dense_1")(x)
    x = layers.Dropout(dense_dropout_rate, name="dense_dropout_1")(x)
    outputs = layers.Dense(num_classes_binary, activation="sigmoid", name="output_layer")(x)

    return keras.Model(inputs, outputs, name="Extended_CNN1D_AMR_Model")


def train_cnn(X_train, y_train, X_val, y_val, labels_all: np.ndarray, ab_name: str,
              *, epochs: int = EPOCHS, batch_size: int = BATCH_SIZE, lr: float = LEARNING_RATE):
    """Authors' exact training loop (class weights, callbacks) -- verbatim."""
    from tensorflow import keras

    neg_total, pos_total = np.bincount(labels_all)
    total = neg_total + pos_total
    class_weight = {
        0: (1 / neg_total) * (total / 2.0) if neg_total > 0 else 0,
        1: (1 / pos_total) * (total / 2.0) if pos_total > 0 else 0,
    }

    model = build_cnn1d_model_extended(X_train.shape[1])
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc"),
                 keras.metrics.Precision(name="precision"), keras.metrics.Recall(name="recall")],
    )
    callbacks = [
        keras.callbacks.ModelCheckpoint(filepath=f"best_cnn1d_model_{ab_name}.keras",
                                        save_best_only=True, monitor="val_auc", mode="max"),
        keras.callbacks.EarlyStopping(monitor="val_auc", patience=60, mode="max", restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=4, min_lr=5e-6),
    ]
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,
              validation_data=(X_val, y_val), callbacks=callbacks,
              class_weight=class_weight, verbose=0)
    return model


def train_xgb(X_train, y_train, X_val, y_val, *, xgb_params: Optional[dict] = None):
    """Authors' exact XGBoost hyperparameters -- verbatim."""
    import xgboost as xgb

    params = dict(xgb_params or XGB_PARAMS)
    neg_count, pos_count = np.bincount(y_train)
    params["scale_pos_weight"] = neg_count / pos_count if pos_count > 0 else 1
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model


def soft_vote(p_cnn: np.ndarray, p_xgb: np.ndarray) -> np.ndarray:
    """Authors' ensemble rule: simple average of the two probabilities."""
    return (p_cnn + p_xgb) / 2.0


def run_one_antibiotic(
    X_tr, y_tr, X_va, y_va, X_te, y_te, ab_name: str,
) -> dict[str, dict]:
    """Train CNN + XGBoost + soft-vote ensemble for one antibiotic, return
    test metrics for all three (threshold chosen on val by F2, per the
    source notebook's `best_thr_f2`)."""
    labels_all = np.concatenate([y_tr, y_va, y_te])

    cnn_model = train_cnn(X_tr, y_tr, X_va, y_va, labels_all, ab_name)
    p_val_cnn = cnn_model.predict(X_va, verbose=0).ravel()
    p_te_cnn = cnn_model.predict(X_te, verbose=0).ravel()

    xgb_model = train_xgb(X_tr, y_tr, X_va, y_va)
    p_val_xgb = xgb_model.predict_proba(X_va)[:, 1]
    p_te_xgb = xgb_model.predict_proba(X_te)[:, 1]

    p_val_ens = soft_vote(p_val_cnn, p_val_xgb)
    p_te_ens = soft_vote(p_te_cnn, p_te_xgb)

    results = {}
    for name, p_val, p_te in [
        ("CNN-only (official)", p_val_cnn, p_te_cnn),
        ("XGB-only (official)", p_val_xgb, p_te_xgb),
        ("Ensemble (soft vote, official)", p_val_ens, p_te_ens),
    ]:
        thr = best_threshold_by_f2(y_va, p_val)
        results[name] = compute_metrics(y_te, p_te, thr)

    from tensorflow import keras
    keras.backend.clear_session()
    return results
