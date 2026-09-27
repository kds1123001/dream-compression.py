import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset

from scenes import get_scene


class SequenceDataset(Dataset):
    def __init__(self, frames, fg_mask, seq_len):
        self.frames = torch.from_numpy(frames.astype(np.float32) / 255.0).permute(0, 3, 1, 2)
        self.fg_mask = torch.from_numpy(fg_mask.astype(bool))
        self.seq_len = seq_len
        if len(self.frames) <= seq_len:
            raise ValueError(f"Need > {seq_len} frames, got {len(self.frames)}")

    def __len__(self):
        return len(self.frames) - self.seq_len

    def __getitem__(self, idx):
        x = self.frames[idx:idx + self.seq_len]
        y = self.frames[idx + self.seq_len]
        y_mask = self.fg_mask[idx + self.seq_len]
        return x, y, y_mask


def split_dataset(dataset, val_fraction=0.2, seed=0):
    n = len(dataset)
    if val_fraction <= 0.0:
        return dataset, None
    n_val = max(1, int(n * val_fraction))
    rng = np.random.RandomState(seed)
    indices = np.arange(n)
    rng.shuffle(indices)
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]
    return Subset(dataset, train_idx.tolist()), Subset(dataset, val_idx.tolist())


def build_dataloaders(config):
    frames, fg_mask = get_scene(
        config.scene,
        n_frames=config.n_frames if config.n_frames is not None else 240,
        size=config.frame_size,
        video_path=config.video_path,
    )
    dataset = SequenceDataset(frames, fg_mask, config.seq_len)
    train_set, val_set = split_dataset(dataset, config.val_fraction)
    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, drop_last=False)
    val_loader = None
    if val_set is not None and len(val_set) > 0:
        val_loader = DataLoader(val_set, batch_size=config.batch_size, shuffle=False, drop_last=False)
    return train_loader, val_loader, dataset, frames, fg_mask


def frame_motion_weights(target, prev_frame, k):
    motion = torch.abs(target - prev_frame)
    return 1 + motion * k
