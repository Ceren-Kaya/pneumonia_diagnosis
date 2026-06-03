import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.metrics import Precision, Recall, AUC

from data import make_gens, compute_class_weights
from model import build_model

DATA_DIR = "chest_xray"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
BEST_PATH = "best_model.keras"

# data generators
train_gen, val_gen, test_gen, class_names = make_gens(
    DATA_DIR, IMG_SIZE, BATCH_SIZE, val_split=0.1
)

# class weights (class imbalance)
class_weights = compute_class_weights(train_gen)
print("Class weights:", class_weights)

# build model
model, base_model = build_model(IMG_SIZE)

# callbacks
callbacks = [
    EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=2, min_lr=1e-6),
    ModelCheckpoint(BEST_PATH, monitor="val_loss", save_best_only=True),
]

# train
model.compile(
    optimizer=Adam(1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy", Precision(name="precision"), Recall(name="recall"), AUC(name="auc")],
)

model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=5,
    callbacks=callbacks,
    class_weight=class_weights,
    verbose=1,
)

# fine-tuning
base_model.trainable = True
for layer in base_model.layers[:-50]:
    layer.trainable = False

model.compile(
    optimizer=Adam(1e-5),
    loss="binary_crossentropy",
    metrics=["accuracy", Precision(name="precision"), Recall(name="recall"), AUC(name="auc")],
)

model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10,
    callbacks=callbacks,
    class_weight=class_weights,
    verbose=1,
)