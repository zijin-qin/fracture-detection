import keras_tuner as kt
from tensorflow.keras import layers, callbacks

INPUT_SHAPE = (224, 224, 1)
NUM_CLASSES = len(np.unique(y_train))

def build_vit_model(hp):
    patch_size = hp.Choice('patch_size', values=[8, 16, 32, 64])
    d_model = hp.Choice('d_model', values=[64, 128, 256, 512])
    num_heads = hp.Choice('num_heads', values=[2, 4, 8, 16])
    num_layers = hp.Int('num_layers', min_value=2, max_value=10, step=2)
    mlp_dim = hp.Choice('mlp_dim', values=[128, 256, 512, 1024])
    dropout_rate = hp.Choice('dropout_rate', values=[0.1, 0.2, 0.3, 0.4, 0.5])
    learning_rate = hp.Choice('learning_rate', values=[1e-3, 1e-4, 1e-5, 1e-6]) 

    num_patches = (INPUT_SHAPE[0] // patch_size) * (INPUT_SHAPE[1] // patch_size)

    inputs = layers.Input(shape=INPUT_SHAPE)
    patches = Patches(patch_size)(inputs)
    encoded_patches = PatchEncoder(num_patches, d_model)(patches)

    for i in range(num_layers):
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model, dropout=dropout_rate
        )(x1, x1)
        x2 = layers.Add()([attention_output, encoded_patches])

        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = layers.Dense(mlp_dim, activation=tf.nn.gelu)(x3)
        x3 = layers.Dropout(dropout_rate)(x3) 
        x3 = layers.Dense(d_model)(x3)
        encoded_patches = layers.Add()([x3, x2])

    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.GlobalAveragePooling1D()(representation)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(representation)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

tuner = kt.RandomSearch(
    build_vit_model,
    objective='val_accuracy',
    max_trials=30,
    executions_per_trial=3,
    directory='vit_hyperparam_tuning',
    project_name='vision_transformer'
)

early_stopping = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

lr_reduction = callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-6
)

tensorboard_cb = callbacks.TensorBoard(
    log_dir='./logs',
    histogram_freq=1
)

tuner.search(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=32,
    callbacks=[early_stopping, lr_reduction, tensorboard_cb]
)

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print(f"Best Patch Size: {best_hps.get('patch_size')}")
print(f"Best Embedding Dimension: {best_hps.get('d_model')}")
print(f"Best Number of Heads: {best_hps.get('num_heads')}")
print(f"Best Number of Layers: {best_hps.get('num_layers')}")
print(f"Best MLP Dim: {best_hps.get('mlp_dim')}")
print(f"Best Dropout Rate: {best_hps.get('dropout_rate')}")
print(f"Best Learning Rate: {best_hps.get('learning_rate')}")