import matplotlib.pyplot as plt
import torch

def plot_preds_and_error(
    pred_x,
    pred_y,
    pred_z,
    mean_error_total,
    time_val,
    window_size,
    val_scaled,
    error,
    horizon=1,
    horizon_step=1,
    title=None
):
    """
    Plots real and predicted values for the three Lorenz channels.

    For horizon > 1, horizon_step selects which future step is plotted:
    horizon_step=1 plots t+1, horizon_step=5 plots t+5, etc.
    """
    h = horizon_step - 1

    if horizon_step < 1 or horizon_step > horizon:
        raise ValueError(f"horizon_step must be between 1 and {horizon}")

    # Convert tensors to arrays if needed
    def to_numpy(a):
        if isinstance(a, torch.Tensor):
            return a.detach().cpu().numpy()
        return a

    pred_x = to_numpy(pred_x)
    pred_y = to_numpy(pred_y)
    pred_z = to_numpy(pred_z)
    mean_error_total = to_numpy(mean_error_total)

    # If predictions are 2D, select the requested horizon step
    if horizon > 1:
        pred_x_plot = pred_x[:, h]
        pred_y_plot = pred_y[:, h]
        pred_z_plot = pred_z[:, h]
    else:
        pred_x_plot = pred_x.reshape(-1)
        pred_y_plot = pred_y.reshape(-1)
        pred_z_plot = pred_z.reshape(-1)

    # Time values aligned with the selected horizon step
    time_pred = time_val[window_size + h : window_size + h + len(pred_x_plot)]

    # Real values aligned with the selected horizon step
    real_x = val_scaled[window_size + h : window_size + h + len(pred_x_plot), 0]
    real_y = val_scaled[window_size + h : window_size + h + len(pred_y_plot), 1]
    real_z = val_scaled[window_size + h : window_size + h + len(pred_z_plot), 2]

    fig, axes = plt.subplots(4, 1, figsize=(11, 5), sharex=True)

    for ax in axes[:-1]:
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)

    axes[0].plot(time_pred, real_x, label="Real values")
    axes[0].plot(time_pred, pred_x_plot, label=f"Prediction t+{horizon_step}", linestyle="--")
    axes[0].legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.)
    axes[0].set_ylabel("x")

    axes[1].plot(time_pred, real_y, label="Real values")
    axes[1].plot(time_pred, pred_y_plot, label=f"Prediction t+{horizon_step}", linestyle="--")
    axes[1].set_ylabel("y")

    axes[2].plot(time_pred, real_z, label="Real values")
    axes[2].plot(time_pred, pred_z_plot, label=f"Prediction t+{horizon_step}", linestyle="--")
    axes[2].set_ylabel("z")

    axes[3].plot(time_pred, mean_error_total, color="red", label="Mean error")
    axes[3].set_ylabel(error)
    axes[3].set_xlabel("Time")
    axes[3].grid(alpha=0.3)
    axes[3].set_xlim(7.5, 10)

    if title is not None:
        fig.suptitle(title, fontsize=14, fontweight="bold")

    plt.tight_layout(rect=[0, 0, 0.85, 0.95])
    plt.show()