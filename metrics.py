import numpy as np


def fg_bg_error(pred, target, fg_mask):
    err = (pred - target) ** 2
    err = err.mean(axis=-1)
    fg = err[fg_mask]
    bg = err[~fg_mask]
    fg_mse = float(fg.mean()) if fg.size else float("nan")
    bg_mse = float(bg.mean()) if bg.size else float("nan")
    return fg_mse, bg_mse


def batch_fg_bg_error(pred_batch, target_batch, fg_mask_batch):
    fgs, bgs = [], []
    for i in range(pred_batch.shape[0]):
        fg, bg = fg_bg_error(pred_batch[i], target_batch[i], fg_mask_batch[i])
        if not np.isnan(fg):
            fgs.append(fg)
        if not np.isnan(bg):
            bgs.append(bg)
    fg_mean = float(np.mean(fgs)) if fgs else float("nan")
    bg_mean = float(np.mean(bgs)) if bgs else float("nan")
    return fg_mean, bg_mean


def detect_collapse(fg_history, bg_history, ratio_threshold=8.0, patience=3):
    fg = np.asarray(fg_history, dtype=float)
    bg = np.asarray(bg_history, dtype=float) + 1e-8
    ratio = fg / bg
    for e in range(len(ratio) - patience + 1):
        if np.all(ratio[e:e + patience] >= ratio_threshold):
            return e, ratio
    return None, ratio


def psnr(pred, target, max_val=1.0):
    mse = float(np.mean((pred - target) ** 2))
    if mse <= 1e-12:
        return 99.0
    return 20 * np.log10(max_val) - 10 * np.log10(mse)


def ssim(pred, target, max_val=1.0, win=7):
    p = pred.mean(axis=-1) if pred.ndim == 3 else pred
    t = target.mean(axis=-1) if target.ndim == 3 else target
    c1 = (0.01 * max_val) ** 2
    c2 = (0.03 * max_val) ** 2

    def box(img):
        k = np.ones((win, win)) / (win * win)
        pad = win // 2
        padded = np.pad(img, pad, mode="reflect")
        out = np.zeros_like(img)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                out[i, j] = (padded[i:i + win, j:j + win] * k).sum()
        return out

    mu_p, mu_t = box(p), box(t)
    mu_p2, mu_t2, mu_pt = mu_p ** 2, mu_t ** 2, mu_p * mu_t
    sig_p2 = box(p * p) - mu_p2
    sig_t2 = box(t * t) - mu_t2
    sig_pt = box(p * t) - mu_pt
    num = (2 * mu_pt + c1) * (2 * sig_pt + c2)
    den = (mu_p2 + mu_t2 + c1) * (sig_p2 + sig_t2 + c2)
    return float(np.mean(num / den))


def batch_quality(pred_batch, target_batch):
    psnrs, ssims = [], []
    for i in range(pred_batch.shape[0]):
        psnrs.append(psnr(pred_batch[i], target_batch[i]))
        ssims.append(ssim(pred_batch[i], target_batch[i]))
    return float(np.mean(psnrs)), float(np.mean(ssims))


def compression_ratio(frame_size, latent_size, bits_per_pixel_channel=8, bits_per_latent=32):
    original_bits = frame_size * frame_size * 3 * bits_per_pixel_channel
    compressed_bits = latent_size * bits_per_latent
    return original_bits / compressed_bits


def fitness_from_loss(loss, eps=1e-8):
    return 1.0 / (loss + eps)
