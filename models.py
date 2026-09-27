import numpy as np
import torch
import torch.nn as nn


class KalmanFilter2D:
    def __init__(self):
        self.x = np.array([[0], [0], [0], [0]], dtype=float)
        self.F = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ])
        self.P = np.eye(4) * 1000
        self.R = np.eye(2) * 5
        self.Q = np.eye(4) * 0.1

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        z = np.array(z).reshape(2, 1)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P

    def track_sequence(self, positions):
        tracked = []
        for pos in positions:
            self.predict()
            self.update(pos)
            tracked.append((float(self.x[0]), float(self.x[1])))
        return tracked


class Encoder(nn.Module):
    def __init__(self, latent_size, frame_size=64, in_channels=3):
        super().__init__()
        self.encoded_hw = frame_size // 4
        self.conv1 = nn.Conv2d(in_channels, 32, 4, 2, 1)
        self.conv2 = nn.Conv2d(32, 64, 4, 2, 1)
        self.act = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(64 * self.encoded_hw * self.encoded_hw, latent_size)

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.flatten(x)
        return self.fc(x)


class TransposeDecoder(nn.Module):
    def __init__(self, latent_size, frame_size=64):
        super().__init__()
        self.encoded_hw = frame_size // 4
        self.fc = nn.Linear(latent_size, 64 * self.encoded_hw * self.encoded_hw)
        self.deconv1 = nn.ConvTranspose2d(64, 32, 4, 2, 1)
        self.deconv2 = nn.ConvTranspose2d(32, 3, 4, 2, 1)
        self.act = nn.ReLU()
        self.out_act = nn.Sigmoid()

    def forward(self, z):
        x = self.fc(z).view(-1, 64, self.encoded_hw, self.encoded_hw)
        x = self.act(self.deconv1(x))
        return self.out_act(self.deconv2(x))


class UpsampleDecoder(nn.Module):
    def __init__(self, latent_size, frame_size=64):
        super().__init__()
        self.encoded_hw = frame_size // 4
        self.fc = nn.Linear(latent_size, 64 * self.encoded_hw * self.encoded_hw)
        self.up1 = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv1 = nn.Conv2d(64, 32, 3, 1, 1)
        self.up2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv2 = nn.Conv2d(32, 3, 3, 1, 1)
        self.act = nn.ReLU()
        self.out_act = nn.Sigmoid()

    def forward(self, z):
        x = self.fc(z).view(-1, 64, self.encoded_hw, self.encoded_hw)
        x = self.act(self.conv1(self.up1(x)))
        return self.out_act(self.conv2(self.up2(x)))


class DeepUpsampleDecoder(nn.Module):
    def __init__(self, latent_size, frame_size=64):
        super().__init__()
        self.encoded_hw = frame_size // 4
        self.fc = nn.Linear(latent_size, 64 * self.encoded_hw * self.encoded_hw)
        self.up1 = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv1a = nn.Conv2d(64, 48, 3, 1, 1)
        self.conv1b = nn.Conv2d(48, 32, 3, 1, 1)
        self.up2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv2a = nn.Conv2d(32, 16, 3, 1, 1)
        self.conv2b = nn.Conv2d(16, 3, 3, 1, 1)
        self.norm1 = nn.GroupNorm(8, 48)
        self.norm2 = nn.GroupNorm(4, 16)
        self.act = nn.ReLU()
        self.out_act = nn.Sigmoid()

    def forward(self, z):
        x = self.fc(z).view(-1, 64, self.encoded_hw, self.encoded_hw)
        x = self.up1(x)
        x = self.act(self.norm1(self.conv1a(x)))
        x = self.act(self.conv1b(x))
        x = self.up2(x)
        x = self.act(self.norm2(self.conv2a(x)))
        return self.out_act(self.conv2b(x))


DECODER_REGISTRY = {
    "transpose": TransposeDecoder,
    "upsample": UpsampleDecoder,
    "deep_upsample": DeepUpsampleDecoder,
}


class DreamNet(nn.Module):
    def __init__(self, latent_size=2, decoder_variant="transpose", frame_size=64):
        super().__init__()
        if decoder_variant not in DECODER_REGISTRY:
            raise ValueError(f"Unknown decoder_variant '{decoder_variant}'. Options: {list(DECODER_REGISTRY)}")
        if frame_size % 4 != 0:
            raise ValueError("frame_size must be divisible by 4")
        self.latent_size = latent_size
        self.decoder_variant = decoder_variant
        self.encoder = Encoder(latent_size, frame_size)
        self.lstm = nn.LSTM(latent_size, latent_size, batch_first=True)
        self.decoder = DECODER_REGISTRY[decoder_variant](latent_size, frame_size)

    def forward(self, x):
        B, T, C, H, W = x.shape
        latent = []
        for t in range(T):
            z = self.encoder(x[:, t])
            latent.append(z)
        latent = torch.stack(latent, dim=1)
        output, _ = self.lstm(latent)
        predicted = output[:, -1]
        return self.decoder(predicted)

    def encode_sequence(self, x):
        B, T, C, H, W = x.shape
        latent = []
        for t in range(T):
            latent.append(self.encoder(x[:, t]))
        latent = torch.stack(latent, dim=1)
        output, _ = self.lstm(latent)
        return output[:, -1]


def build_model(config, device):
    model = DreamNet(
        latent_size=config.latent_size,
        decoder_variant=config.decoder_variant,
        frame_size=config.frame_size,
    )
    return model.to(device)
