import torch
from dataclasses import dataclass, field
from typing import Optional, Tuple, List

VIDEO_PATH = "video.mp4.webm"
FRAME_SIZE = 64
SEQUENCE_LEN = 5
BATCH_SIZE = 8
EPOCHS = 25
LEARNING_RATE = 0.001
MOTION_WEIGHT_K = 10.0
VAL_FRACTION = 0.2
COLLAPSE_RATIO_THRESHOLD = 8.0
COLLAPSE_PATIENCE = 3
OUTPUT_DIR = "dream_outputs"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SCENES = ["pendulum", "cars", "blockworld", "multi_object"]
LATENT_SIZES = [64, 32, 16, 8, 4, 2]
DECODER_VARIANTS = ["transpose", "upsample", "deep_upsample"]


@dataclass
class ExperimentConfig:
    run_id: str
    scene: str
    latent_size: int
    decoder_variant: str = "transpose"
    use_motion_weight: bool = True
    epochs: int = EPOCHS
    frame_size: int = FRAME_SIZE
    seq_len: int = SEQUENCE_LEN
    batch_size: int = BATCH_SIZE
    n_frames: Optional[int] = 240
    lr: float = LEARNING_RATE
    motion_weight_k: float = MOTION_WEIGHT_K
    val_fraction: float = VAL_FRACTION
    video_path: Optional[str] = None

    def label(self):
        mw = "mw_on" if self.use_motion_weight else "mw_off"
        return f"{self.run_id}_{self.scene}_L{self.latent_size}_{self.decoder_variant}_{mw}"


def legacy_video_config(video_path=VIDEO_PATH, epochs=EPOCHS, latent_size=2):
    return ExperimentConfig(
        run_id="legacy_video",
        scene="video",
        latent_size=latent_size,
        decoder_variant="transpose",
        use_motion_weight=True,
        epochs=epochs,
        frame_size=FRAME_SIZE,
        seq_len=SEQUENCE_LEN,
        batch_size=BATCH_SIZE,
        n_frames=None,
        lr=LEARNING_RATE,
        motion_weight_k=MOTION_WEIGHT_K,
        val_fraction=0.0,
        video_path=video_path,
    )


def build_collapse_grid(
    scenes: List[str] = None,
    latent_size: int = 4,
    decoder_variant: str = "transpose",
    epochs: int = 40,
    n_frames: int = 240,
    motion_weight_options: Tuple[bool, ...] = (True, False),
) -> List[ExperimentConfig]:
    scenes = scenes or SCENES
    configs = []
    for scene in scenes:
        for use_mw in motion_weight_options:
            configs.append(
                ExperimentConfig(
                    run_id="collapse",
                    scene=scene,
                    latent_size=latent_size,
                    decoder_variant=decoder_variant,
                    use_motion_weight=use_mw,
                    epochs=epochs,
                    n_frames=n_frames,
                )
            )
    return configs


def build_quality_grid(
    scene: str = "multi_object",
    latent_sizes: List[int] = None,
    decoder_variants: List[str] = None,
    epochs: int = 30,
    n_frames: int = 240,
    use_motion_weight: bool = True,
) -> List[ExperimentConfig]:
    latent_sizes = latent_sizes or LATENT_SIZES
    decoder_variants = decoder_variants or DECODER_VARIANTS
    configs = []
    for latent_size in latent_sizes:
        for decoder_variant in decoder_variants:
            configs.append(
                ExperimentConfig(
                    run_id="quality",
                    scene=scene,
                    latent_size=latent_size,
                    decoder_variant=decoder_variant,
                    use_motion_weight=use_motion_weight,
                    epochs=epochs,
                    n_frames=n_frames,
                )
            )
    return configs


def build_full_grid(
    collapse_kwargs: dict = None,
    quality_kwargs: dict = None,
) -> List[ExperimentConfig]:
    collapse_kwargs = collapse_kwargs or {}
    quality_kwargs = quality_kwargs or {}
    return build_collapse_grid(**collapse_kwargs) + build_quality_grid(**quality_kwargs)


def quick_smoke_grid() -> List[ExperimentConfig]:
    return [
        ExperimentConfig(
            run_id="smoke",
            scene="pendulum",
            latent_size=4,
            decoder_variant="transpose",
            use_motion_weight=True,
            epochs=2,
            n_frames=40,
        ),
        ExperimentConfig(
            run_id="smoke",
            scene="pendulum",
            latent_size=4,
            decoder_variant="upsample",
            use_motion_weight=False,
            epochs=2,
            n_frames=40,
        ),
    ]
