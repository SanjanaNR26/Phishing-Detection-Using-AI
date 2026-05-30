"""End-to-end training pipeline."""
import argparse
import os
import matplotlib.pyplot as plt
import tensorflow as tf

from preprocess import build_dataset, prepare
from model import build_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True, help='CSV with columns url,label')
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch', type=int, default=256)
    p.add_argument('--models_dir', default='models')
    p.add_argument('--max_rows', type=int, default=None)
    args = p.parse_args()

    X, y = build_dataset(args.data, max_rows=args.max_rows)
    (Xtr, ytr), (Xval, yval), (Xte, yte) = prepare(X, y, args.models_dir)

    model = build_model(input_dim=Xtr.shape[1])
    model.summary()

    cbs = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5),
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(args.models_dir, 'phishing_dnn.keras'),
            monitor='val_auc', mode='max', save_best_only=True),
    ]

    hist = model.fit(Xtr, ytr, validation_data=(Xval, yval),
                     epochs=args.epochs, batch_size=args.batch,
                     callbacks=cbs, verbose=2)

    # Test
    print("\n=== Test set ===")
    res = model.evaluate(Xte, yte, verbose=0)
    for n, v in zip(model.metrics_names, res):
        print(f"  {n}: {v:.4f}")

    # Save curves
    os.makedirs(args.models_dir, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(hist.history['loss'], label='train')
    ax[0].plot(hist.history['val_loss'], label='val')
    ax[0].set_title('Loss'); ax[0].legend()
    ax[1].plot(hist.history['accuracy'], label='train')
    ax[1].plot(hist.history['val_accuracy'], label='val')
    ax[1].set_title('Accuracy'); ax[1].legend()
    plt.savefig(os.path.join(args.models_dir, 'training_curves.png'), dpi=120)
    print("Saved training_curves.png")


if __name__ == '__main__':
    main()
