import os
import csv
import time
import numpy as np
import torch
from colorama import Fore, init as colorama_init

import config as cfg
from data import build_dataloaders, frame_motion_weights
from models import build_model, KalmanFilter2D
from metrics import (
    batch_fg_bg_error,
    batch_quality,
    detect_collapse,
    compression_ratio,
    fitness_from_loss,
)
from plots import (
    ensure_dir,
    plot_loss_fitness,
    plot_reconstruction,
    plot_collapse_curve,
    plot_scene_preview,
    plot_latent_sweep,
    plot_motion_weight_ablation,
    plot_summary_table,
)

colorama_init(autoreset=True)


def evaluate(model, loader, device):
    if loader is None:
        return float("nan"), float("nan"), float("nan"), float("nan")
    model.eval()
    fg_errs, bg_errs, psnrs, ssims = [], [], [], []
    with torch.no_grad():
        for sequence, target, mask in loader:
            sequence = sequence.to(device)
            target = target.to(device)
            prediction = model(sequence)
            pred_np = prediction.permute(0, 2, 3, 1).cpu().numpy()
            target_np = target.permute(0, 2, 3, 1).cpu().numpy()
            mask_np = mask.cpu().numpy()
            fg, bg = batch_fg_bg_error(pred_np, target_np, mask_np)
            p, s = batch_quality(pred_np, target_np)
            if not np.isnan(fg):
                fg_errs.append(fg)
            if not np.isnan(bg):
                bg_errs.append(bg)
            psnrs.append(p)
            ssims.append(s)
    model.train()
    fg_mean = float(np.mean(fg_errs)) if fg_errs else float("nan")
    bg_mean = float(np.mean(bg_errs)) if bg_errs else float("nan")
    psnr_mean = float(np.mean(psnrs)) if psnrs else float("nan")
    ssim_mean = float(np.mean(ssims)) if ssims else float("nan")
    return fg_mean, bg_mean, psnr_mean, ssim_mean


def train_one_config(config, output_root=cfg.OUTPUT_DIR, verbose=True):
    device = cfg.DEVICE
    run_dir = ensure_dir(os.path.join(output_root, config.label()))
    train_loader, val_loader, dataset, frames, fg_mask = build_dataloaders(config)
    model = build_model(config, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    loss_history, fitness_history = [], []
    fg_history, bg_history = [], []
    psnr_history, ssim_history = [], []

    if verbose:
        print(Fore.CYAN + "=" * 60)
        print(Fore.CYAN + f"Run: {config.label()}  device={device}")
        print(Fore.CYAN + "=" * 60)

    eval_loader = val_loader if val_loader is not None else train_loader

    for epoch in range(config.epochs):
        total_loss = 0.0
        for sequence, target, mask in train_loader:
            sequence = sequence.to(device)
            target = target.to(device)
            prediction = model(sequence)

            if config.use_motion_weight:
                weights = frame_motion_weights(target, sequence[:, -1], config.motion_weight_k)
                loss = (((prediction - target) ** 2) * weights).mean()
            else:
                loss = ((prediction - target) ** 2).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(train_loader))
        fitness = fitness_from_loss(avg_loss)
        loss_history.append(avg_loss)
        fitness_history.append(fitness)

        fg_err, bg_err, psnr_val, ssim_val = evaluate(model, eval_loader, device)
        fg_history.append(fg_err)
        bg_history.append(bg_err)
        psnr_history.append(psnr_val)
        ssim_history.append(ssim_val)

        if verbose:
            color = Fore.GREEN if epoch == config.epochs - 1 else Fore.YELLOW
            print(
                color
                + f"Epoch {epoch + 1}/{config.epochs}"
                + f" | Loss:{avg_loss:.5f}"
                + f" | Fitness:{fitness:.2f}"
                + f" | fg_err:{fg_err:.5f}"
                + f" | bg_err:{bg_err:.5f}"
                + f" | PSNR:{psnr_val:.2f}"
                + f" | SSIM:{ssim_val:.3f}"
            )

    collapse_epoch, ratio_curve = detect_collapse(
        fg_history, bg_history, cfg.COLLAPSE_RATIO_THRESHOLD, cfg.COLLAPSE_PATIENCE
    )
    collapse_epoch_display = collapse_epoch if collapse_epoch is not None else config.epochs

    plot_loss_fitness(loss_history, fitness_history, os.path.join(run_dir, "loss_fitness.png"), config.label())
    plot_collapse_curve(fg_history, bg_history, collapse_epoch, os.path.join(run_dir, "collapse.png"), config.label())

    sample_x, sample_y, sample_mask = dataset[0]
    model.eval()
    with torch.no_grad():
        dream = model(sample_x.unsqueeze(0).to(device))
    dream_np = dream[0].permute(1, 2, 0).detach().cpu().numpy()
    target_np = sample_y.permute(1, 2, 0).numpy()
    plot_reconstruction(target_np, dream_np, os.path.join(run_dir, "reconstruction.png"), config.label())

    kf = KalmanFilter2D()
    gray = dream_np.mean(axis=2)
    y_pos, x_pos = np.unravel_index(np.argmax(gray), gray.shape)
    kf.predict()
    kf.update([x_pos, y_pos])
    filtered_pos = (float(kf.x[0]), float(kf.x[1]))

    ratio = compression_ratio(config.frame_size, config.latent_size)

    result = dict(
        run_id=config.run_id,
        scene=config.scene,
        latent_size=config.latent_size,
        decoder_variant=config.decoder_variant,
        use_motion_weight=config.use_motion_weight,
        epochs=config.epochs,
        final_loss=loss_history[-1],
        final_fitness=fitness_history[-1],
        final_fg_error=fg_history[-1],
        final_bg_error=bg_history[-1],
        final_psnr=psnr_history[-1],
        final_ssim=ssim_history[-1],
        collapse_epoch=collapse_epoch,
        collapse_epoch_display=collapse_epoch_display,
        compression_ratio=ratio,
        filtered_position=filtered_pos,
        run_dir=run_dir,
    )

    if verbose:
        print(Fore.MAGENTA + f"Filtered position: {int(kf.x[0])}, {int(kf.x[1])}")
        print(Fore.MAGENTA + f"Compression: {ratio:.2f}:1")
        collapse_msg = f"epoch {collapse_epoch}" if collapse_epoch is not None else "no collapse detected"
        print(Fore.MAGENTA + f"Collapse: {collapse_msg}")

    return result, dict(
        loss_history=loss_history,
        fitness_history=fitness_history,
        fg_history=fg_history,
        bg_history=bg_history,
        psnr_history=psnr_history,
        ssim_history=ssim_history,
        ratio_curve=ratio_curve.tolist(),
    )


def write_results_csv(results, path):
    if not results:
        return
    keys = [k for k in results[0].keys() if k != "filtered_position"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in keys})


def run_grid(configs, output_root=cfg.OUTPUT_DIR, verbose=True):
    ensure_dir(output_root)
    results = []
    histories = {}
    start = time.time()
    for i, config in enumerate(configs):
        if verbose:
            print(Fore.BLUE + f"\n[{i + 1}/{len(configs)}] starting {config.label()}")
        result, history = train_one_config(config, output_root, verbose)
        results.append(result)
        histories[config.label()] = history
    elapsed = time.time() - start
    if verbose:
        print(Fore.GREEN + f"\nCompleted {len(configs)} runs in {elapsed:.1f}s")

    write_results_csv(results, os.path.join(output_root, "results.csv"))

    quality_results = [r for r in results if r["run_id"] == "quality"]
    collapse_results = [r for r in results if r["run_id"] == "collapse"]

    if quality_results:
        plot_latent_sweep(quality_results, os.path.join(output_root, "latent_sweep_summary.png"))
        plot_summary_table(
            quality_results,
            os.path.join(output_root, "quality_table.png"),
            ["scene", "latent_size", "decoder_variant", "final_loss", "final_psnr", "final_ssim", "compression_ratio"],
        )

    if collapse_results:
        plot_motion_weight_ablation(collapse_results, os.path.join(output_root, "motion_weight_ablation.png"))
        plot_summary_table(
            collapse_results,
            os.path.join(output_root, "collapse_table.png"),
            ["scene", "use_motion_weight", "collapse_epoch_display", "final_fg_error", "final_bg_error"],
        )

    return results, histories


def run_legacy_video(video_path=cfg.VIDEO_PATH, epochs=cfg.EPOCHS, latent_size=2):
    legacy_cfg = cfg.legacy_video_config(video_path=video_path, epochs=epochs, latent_size=latent_size)
    return train_one_config(legacy_cfg, output_root=cfg.OUTPUT_DIR, verbose=True)


def run_scene_preview(output_root=cfg.OUTPUT_DIR):
    from scenes import get_scene

    previews = {}
    for name in cfg.SCENES:
        frames, _ = get_scene(name, n_frames=30, size=cfg.FRAME_SIZE)
        previews[name] = frames[15] / 255.0
    ensure_dir(output_root)
    plot_scene_preview(previews, os.path.join(output_root, "scene_preview.png"))


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["auto", "video", "grid", "smoke", "collapse", "quality"], default="auto")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    run_scene_preview()

    if args.mode == "smoke":
        configs = cfg.quick_smoke_grid()
        run_grid(configs)
        return

    if args.mode == "video" or (args.mode == "auto" and os.path.exists(cfg.VIDEO_PATH)):
        epochs = args.epochs or cfg.EPOCHS
        run_legacy_video(epochs=epochs)
        return

    if args.mode == "collapse":
        epochs = args.epochs or 40
        configs = cfg.build_collapse_grid(epochs=epochs)
        run_grid(configs)
        return

    if args.mode == "quality":
        epochs = args.epochs or 30
        configs = cfg.build_quality_grid(epochs=epochs)
        run_grid(configs)
        return

    collapse_epochs = args.epochs or 40
    quality_epochs = args.epochs or 30
    configs = cfg.build_full_grid(
        collapse_kwargs=dict(epochs=collapse_epochs),
        quality_kwargs=dict(epochs=quality_epochs),
    )
    run_grid(configs)


if __name__ == "__main__":
    main()
