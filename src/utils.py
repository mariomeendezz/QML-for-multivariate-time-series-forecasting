import matplotlib.pyplot as plt
import torch
import pennylane as qml
from pennylane import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import numpy as np_base
import random
import csv
from pathlib import Path

def reset_seeds(seed: int) -> None:
    """
    Reset random seeds for reproducibility across different libraries.
    """
    random.seed(seed)
    np_base.random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def generate_lorenz(
    npoints: int,
    h: float,
    params: tuple[float, float, float],
    init_point: tuple[float, float, float]
) -> np.ndarray:
    """
    Generate a Lorenz system trajectory using Euler integration.
    Returns the x, y, z coordinates as a single dataset.
    """
    # Lorenz system parameters
    sigma, rho, beta = params[0], params[1], params[2]

    # Initial conditions
    x, y, z = [init_point[0]], [init_point[1]], [init_point[2]]

    # Solve the Lorenz system using Euler integration
    for _ in range(npoints - 1):
        x_t, y_t, z_t = x[-1], y[-1], z[-1]

        x.append(x_t + h * sigma * (y_t - x_t))
        y.append(y_t + h * (-y_t - z_t * x_t + rho * x_t))
        z.append(z_t + h * (-beta * z_t + x_t * y_t))

    # Combine the three coordinates into a single dataset
    dataset = np.column_stack((x, y, z))

    return dataset

def plot_lorenz(
    train: np.ndarray,
    val: np.ndarray,
    time_tr: np.ndarray,
    time_val: np.ndarray
) -> None:
    """
    Plot the Lorenz system coordinates for training and validation data.
    """
    fig, axes = plt.subplots(3, 1, figsize=(11, 5), sharex=True)

    # Set common y-axis limits and grid
    for ax in axes:
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)

    # Plot x coordinate
    axes[0].plot(time_tr, train[:, 0], label="Train")
    axes[0].plot(time_val, val[:, 0], label="Validation")
    axes[0].legend()
    axes[0].set_ylabel("x")

    # Plot y coordinate
    axes[1].plot(time_tr, train[:, 1])
    axes[1].plot(time_val, val[:, 1])
    axes[1].set_ylabel("y")

    # Plot z coordinate
    axes[2].plot(time_tr, train[:, 2])
    axes[2].plot(time_val, val[:, 2])
    axes[2].set_ylabel("z")
    axes[2].set_xlabel("Time")
    axes[2].set_xlim(0, 10)

    fig.suptitle("Scaled Lorenz system", fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.show()

def create_sequences(
    data: np.ndarray,
    window_size: int,
    horizon: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create input-output sequences for time series forecasting.
    Each input uses the previous window_size points to predict the next horizon points.
    """
    n_data = len(data)
    x = []
    y = []

    # Slide a window over the time series
    for i in range(window_size, n_data - horizon + 1):

        # Input sequence: previous window_size points
        x.append(data[i - window_size:i])

        # Target sequence: next horizon points
        y.append(data[i:i + horizon])

    return np.array(x), np.array(y)

def create_channel_loader(
    x: torch.Tensor,
    y: torch.Tensor,
    channel: int,
    batch_size: int = 128,
    shuffle: bool = False
) -> DataLoader:
    """
    Create a DataLoader for one independent time series channel.
    """
    # Select one channel from the input and target tensors
    x_channel = x[:, :, channel]  # (n_samples, window_size)
    y_channel = y[:, :, channel]  # (n_samples, horizon)

    # Create dataset and DataLoader
    dataset = TensorDataset(x_channel, y_channel)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return loader

def angle_encoding(
    nqubits: int, 
    inputs: list[float]
) -> None:
    """
    Encodes classical input data into a quantum circuit using angle encoding 
    by applying Ry rotations with angle pi * input[i]
    """
    # Encode each classical value into one qubit
    for i in range(nqubits):
        qml.RY(np.pi * inputs[..., i], wires=i)

def dense_encoding(
    nqubits: int, 
    inputs: list[float]
) -> None:
    """
    Encodes classical input data into a quantum circuit using dense encoding by
    preparing the state |+> and applying Rz -> Ry -> Rz
    """
    for i in range(nqubits):
        qml.H(wires=i)
        base = i * 3
        qml.RZ(np.pi * inputs[..., base], wires=i)
        qml.RY(np.pi * inputs[..., base + 1], wires=i)
        qml.RZ(np.pi * inputs[..., base + 2], wires=i)

def VQC_strong(
    nqubits: int,
    theta: list[float],
    reps: int
) -> None:
    """
    Builds a variational quantum circuit.

    Each layer applies three parameterized rotations to every qubit:
    Rz -> Ry -> Rz.

    After the rotation block, a circular CNOT entangling layer is applied.
    """
    for rep in range(reps):
        # Rotation block
        for i in range(nqubits):
            base = rep * 3 * nqubits
            qml.RZ(theta[base + i], wires=i)
            qml.RY(theta[base + nqubits + i], wires=i)
            qml.RZ(theta[base + 2 * nqubits + i], wires=i)
        
        # Entangling block
        for i in range(nqubits-1):
            qml.CNOT(wires=[i, i+1])
        
        # Circular connection between last and first qubit
        qml.CNOT(wires=[nqubits-1, 0])

def QSANN_circuit(
    nqubits: int,
    angles: list[float],
    reps: int
) -> None:
    """
    Builds a variational quantum circuit.

    First applies Rx and Ry rotations to all qubits.

    The applies "reps" repetitions of a block consisting
    of circular entanglement with CNOT gates followed by a Ry.
    """
    expected_parameters = nqubits * (reps + 2)
    received_parameters = qml.math.shape(angles)[-1]

    if received_parameters != expected_parameters:
        raise ValueError(
            f"Expected {expected_parameters} parameters, "
            f"but received {received_parameters}."
        )
    
    for i in range(nqubits):
        qml.RX(angles[..., i], wires=i)
        qml.RY(angles[..., i+nqubits], wires=i)

    for rep in range(reps):
        # Entangling block
        for i in range(nqubits-1):
            qml.CNOT(wires=[i, i+1])
        
        # Circular connection between last and first qubit
        qml.CNOT(wires=[nqubits-1, 0])

        # Rotation
        for i in range(nqubits):
            base = 2 * nqubits + rep * nqubits
            qml.RY(angles[..., base+i], wires=i)


# We use the same variational circuit for the QSANN 
# embedding and Q,K and V projections.

def QSANN_embedding(nqubits, inputs, reps):
    QSANN_circuit(
        nqubits=nqubits,
        angles=inputs,
        reps=reps
    )

def QSANN_VQC(nqubits, theta, reps):
    QSANN_circuit(
        nqubits=nqubits,
        angles=theta,
        reps=reps
    )


def plot_loss(
    history: dict[str, list[float]],
    title: str | None = None
) -> None:
    """
    Plot the training and validation loss curves from the model history.
    """
    # Plot training and validation loss
    plt.plot(history["Loss"], label="Training loss")
    plt.plot(history["Val loss"], label="Validation loss")

    # Set plot labels and style
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(alpha=0.5)
    plt.tight_layout()
    plt.legend()

    # Add title if provided
    if title is not None:
        plt.title(title)

    plt.show()


def plot_preds_and_error(
    pred_x: np.ndarray | torch.Tensor,
    pred_y: np.ndarray | torch.Tensor,
    pred_z: np.ndarray | torch.Tensor,
    mean_error_total: np.ndarray | torch.Tensor,
    time_val: np.ndarray,
    window_size: int,
    val_scaled: np.ndarray,
    error: str,
    horizon: int = 1,
    horizon_step: int = 1,
    title: str | None = None
) -> None:
    """
    Plot real and predicted Lorenz coordinates together with the mean error.
    Supports selecting a specific prediction step when using multi-step horizons.
    """
    h = horizon_step - 1

    # Check that the selected horizon step is valid
    if horizon_step < 1 or horizon_step > horizon:
        raise ValueError(f"horizon_step must be between 1 and {horizon}")

    # Convert tensors to arrays if needed
    def to_numpy(a: np.ndarray | torch.Tensor) -> np.ndarray:
        if isinstance(a, torch.Tensor):
            return a.detach().cpu().numpy()
        return a

    pred_x = to_numpy(pred_x)
    pred_y = to_numpy(pred_y)
    pred_z = to_numpy(pred_z)
    mean_error_total = to_numpy(mean_error_total)

    # Select the requested horizon step
    if horizon > 1:
        pred_x_plot = pred_x[:, h]
        pred_y_plot = pred_y[:, h]
        pred_z_plot = pred_z[:, h]
    else:
        pred_x_plot = pred_x.reshape(-1)
        pred_y_plot = pred_y.reshape(-1)
        pred_z_plot = pred_z.reshape(-1)

    # Time values aligned with the selected horizon step
    time_pred = time_val[window_size + h:window_size + h + len(pred_x_plot)]

    # Real values aligned with the selected horizon step
    real_x = val_scaled[window_size + h:window_size + h + len(pred_x_plot), 0]
    real_y = val_scaled[window_size + h:window_size + h + len(pred_y_plot), 1]
    real_z = val_scaled[window_size + h:window_size + h + len(pred_z_plot), 2]

    fig, axes = plt.subplots(4, 1, figsize=(13, 7), sharex=True)

    # Set common y-axis limits and grid
    for ax in axes[:-1]:
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)

    # Plot x coordinate
    axes[0].plot(time_pred, real_x, label="Real values")
    axes[0].plot(time_pred, pred_x_plot, label=f"Prediction t+{horizon_step}", linestyle="--")
    axes[0].legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.)
    axes[0].set_ylabel("x")

    # Plot y coordinate
    axes[1].plot(time_pred, real_y, label="Real values")
    axes[1].plot(time_pred, pred_y_plot, label=f"Prediction t+{horizon_step}", linestyle="--")
    axes[1].set_ylabel("y")

    # Plot z coordinate
    axes[2].plot(time_pred, real_z, label="Real values")
    axes[2].plot(time_pred, pred_z_plot, label=f"Prediction t+{horizon_step}", linestyle="--")
    axes[2].set_ylabel("z")

    # Plot mean error
    axes[3].plot(time_pred, mean_error_total, color="red", label="Mean error")
    axes[3].set_ylabel(error)
    axes[3].set_xlabel("Time")
    axes[3].grid(alpha=0.3)
    axes[3].set_xlim(7.5, 10)

    # Add title if provided
    if title is not None:
        fig.suptitle(title, fontsize=14, fontweight="bold")

    plt.tight_layout(rect=[0, 0, 0.85, 0.95])
    plt.show()


def to_float(value: float | torch.Tensor) -> float:
    """
    Convert a numeric value or tensor into a Python float.
    """
    # Convert tensor to float
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())

    return float(value)


def save_result_csv(
    path: str | Path,
    row: dict,
    key_fields: tuple[str, ...] = ("forecasting", "model")
) -> None:
    """
    Save one result row to a CSV file, replacing existing rows with the same key fields.
    """
    # Create output path if needed
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert metric values to floats
    row = row.copy()
    for metric in ("MAPE", "MAE", "RMSE"):
        row[metric] = to_float(row[metric])

    # Read previous rows if the file exists
    rows = []
    if path.exists():
        with path.open("r", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

    # Replace row with matching key fields
    replaced = False
    for i, existing in enumerate(rows):
        if all(str(existing.get(k)) == str(row.get(k)) for k in key_fields):
            rows[i] = row
            replaced = True
            break

    # Add row if no previous match was found
    if not replaced:
        rows.append(row)

    fieldnames = ["forecasting", "model", "MAPE", "MAE", "RMSE"]

    # Write updated results to CSV
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def count_parameters(model) -> tuple[int, int]:
    """
    Counts the total and trainable parameters of a model
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return total, trainable

