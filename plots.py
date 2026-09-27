import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def plot_loss_fitness(loss_history, fitness_history, out_path, title_suffix=""):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(loss_history)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"Training Loss {title_suffix}")
    axes[1].plot(fitness_history)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Fitness")
    axes[1].set_title(f"Dream Fitness {title_suffix}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def plot_reconstruction(real, dream, out_path, title_suffix=""):
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(np.clip(real, 0, 1))
    axes[0].set_title(f"Real Frame {title_suffix}")
    axes[0].axis("off")
    axes[1].imshow(np.clip(dream, 0, 1))
    axes[1].set_title(f"Dream Reconstruction {title_suffix}")
    axes[1].axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def plot_collapse_curve(fg_history, bg_history, collapse_epoch, out_path, title_suffix=""):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(fg_history, label="foreground error", color="crimson")
    ax.plot(bg_history, label="background error", color="steelblue")
    if collapse_epoch is not None:
        ax.axvline(collapse_epoch, color="black", linestyle="--", label=f"collapse @ epoch {collapse_epoch}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.set_yscale("log")
    ax.set_title(f"Foreground vs Background Error {title_suffix}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def plot_scene_preview(scene_frames_dict, out_path):
    n = len(scene_frames_dict)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3))
    if n == 1:
        axes = [axes]
    for ax, (name, frame) in zip(axes, scene_frames_dict.items()):
        ax.imshow(frame)
        ax.set_title(name)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def plot_latent_sweep(results, out_path):
    variants = sorted(set(r["decoder_variant"] for r in results))
    latent_sizes = sorted(set(r["latent_size"] for r in results))
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    metrics = ["final_loss", "final_fitness", "compression_ratio", "final_psnr"]
    titles = ["Final Loss", "Final Fitness", "Compression Ratio", "Final PSNR (dB)"]
    for ax, metric, title in zip(axes.flat, metrics, titles):
        for variant in variants:
            xs, ys = [], []
            for latent in latent_sizes:
                matches = [r for r in results if r["decoder_variant"] == variant and r["latent_size"] == latent]
                if matches:
                    xs.append(latent)
                    ys.append(matches[0][metric])
            ax.plot(xs, ys, marker="o", label=variant)
        ax.set_xlabel("Latent size")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.set_xscale("log", base=2)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def plot_motion_weight_ablation(results, out_path):
    scenes = sorted(set(r["scene"] for r in results))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    width = 0.35
    x = np.arange(len(scenes))
    for ax_idx, metric in enumerate(["collapse_epoch_display", "final_fg_error"]):
        ax = axes[ax_idx]
        on_vals, off_vals = [], []
        for scene in scenes:
            on_r = [r for r in results if r["scene"] == scene and r["use_motion_weight"]]
            off_r = [r for r in results if r["scene"] == scene and not r["use_motion_weight"]]
            on_vals.append(on_r[0][metric] if on_r else np.nan)
            off_vals.append(off_r[0][metric] if off_r else np.nan)
        ax.bar(x - width / 2, on_vals, width, label="motion-weighted")
        ax.bar(x + width / 2, off_vals, width, label="plain MSE")
        ax.set_xticks(x)
        ax.set_xticklabels(scenes, rotation=20)
        ax.set_title(metric)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def plot_summary_table(results, out_path, columns):
    fig, ax = plt.subplots(figsize=(1.4 * len(columns), 0.5 * (len(results) + 1)))
    ax.axis("off")
    rows = [[str(r.get(c, "")) for c in columns] for r in results]
    table = ax.table(cellText=rows, colLabels=columns, loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
