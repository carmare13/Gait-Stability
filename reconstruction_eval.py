import argparse
import numpy as np
import tensorflow as tf


def reconstruct_and_evaluate(model_path: str, data: np.ndarray, attr_idx: list[int]):
    """Reconstruct selected attributes and compute reconstruction error.

    Parameters
    ----------
    model_path : str
        Path to the saved Keras autoencoder model.
    data : np.ndarray
        Array with shape (n_samples, n_timesteps, n_features) containing the
        original sequences.
    attr_idx : list[int]
        Indices of the attributes to evaluate.

    Returns
    -------
    dict
        Dictionary with MSE, MAE and RMSE for each selected attribute.
    np.ndarray
        Reconstructed data for the selected attributes.
    """
    # Load model
    model = tf.keras.models.load_model(model_path, compile=False)

    # Reconstruct
    recon = model.predict(data, verbose=0)

    # Select requested attributes
    orig_subset = data[:, :, attr_idx]
    recon_subset = recon[:, :, attr_idx]

    # Compute metrics along samples and time dimensions
    mse = np.mean((orig_subset - recon_subset) ** 2, axis=(0, 1))
    mae = np.mean(np.abs(orig_subset - recon_subset), axis=(0, 1))
    rmse = np.sqrt(mse)

    metrics = {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
    }
    return metrics, recon_subset


def main():
    parser = argparse.ArgumentParser(description="Evaluate AE reconstruction error")
    parser.add_argument("--model", required=True, help="Path to saved Keras model")
    parser.add_argument("--data", required=True, help="Path to .npy array")
    parser.add_argument("--attrs", nargs="+", type=int, required=True,
                        help="Indices of attributes to reconstruct")
    parser.add_argument("--out", default="reconstructed_attrs.npy",
                        help="File to save reconstructed attributes")
    args = parser.parse_args()

    # Load data
    data = np.load(args.data).astype(np.float32)

    metrics, recon_subset = reconstruct_and_evaluate(args.model, data, args.attrs)

    np.save(args.out, recon_subset)

    print("Reconstruction error per attribute:")
    for i, idx in enumerate(args.attrs):
        print(f"  Attr {idx}: RMSE={metrics['rmse'][i]:.6f} MSE={metrics['mse'][i]:.6f}")


if __name__ == "__main__":
    main()
