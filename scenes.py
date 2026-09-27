import os
import numpy as np
import cv2


def _blank(h, w, bg_color):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = bg_color
    return img


def gen_pendulum(n_frames=240, size=64, bg=(15, 15, 15), bob_color=(255, 210, 60)):
    frames = np.zeros((n_frames, size, size, 3), dtype=np.uint8)
    fg_mask = np.zeros((n_frames, size, size), dtype=bool)
    pivot = (size // 2, 6)
    length = size * 0.72
    period = 90
    for t in range(n_frames):
        img = _blank(size, size, bg)
        theta = 0.9 * np.sin(2 * np.pi * t / period)
        bx = int(pivot[0] + length * np.sin(theta))
        by = int(pivot[1] + length * np.cos(theta))
        cv2.line(img, pivot, (bx, by), (80, 80, 80), 1)
        cv2.circle(img, (bx, by), 4, bob_color, -1)
        mask = np.zeros((size, size), dtype=np.uint8)
        cv2.circle(mask, (bx, by), 5, 1, -1)
        frames[t] = img
        fg_mask[t] = mask.astype(bool)
    return frames, fg_mask


def gen_cars(n_frames=240, size=64, bg=(30, 30, 30), n_lanes=3):
    frames = np.zeros((n_frames, size, size, 3), dtype=np.uint8)
    fg_mask = np.zeros((n_frames, size, size), dtype=bool)
    lane_ys = np.linspace(12, size - 12, n_lanes).astype(int)
    speeds = np.random.RandomState(11).uniform(0.6, 1.6, size=n_lanes)
    colors = [(220, 60, 60), (60, 160, 220), (230, 220, 90)]
    car_w, car_h = 8, 4
    for t in range(n_frames):
        img = _blank(size, size, bg)
        for ly in lane_ys:
            cv2.line(img, (0, ly + car_h), (size, ly + car_h), (60, 60, 60), 1)
        mask = np.zeros((size, size), dtype=np.uint8)
        for i, ly in enumerate(lane_ys):
            x = int((t * speeds[i] * 2) % (size + car_w)) - car_w
            cv2.rectangle(img, (x, ly), (x + car_w, ly + car_h), colors[i % len(colors)], -1)
            cv2.rectangle(mask, (x, ly), (x + car_w, ly + car_h), 1, -1)
        frames[t] = img
        fg_mask[t] = mask.astype(bool)
    return frames, fg_mask


def gen_blockworld(n_frames=240, size=64, block=8):
    rng = np.random.RandomState(7)
    n_cells = size // block
    palette = np.array(
        [[46, 125, 50], [76, 175, 80], [121, 85, 72], [141, 110, 99], [96, 96, 96]],
        dtype=np.uint8,
    )
    terrain = palette[rng.randint(0, len(palette), size=(n_cells, n_cells))]
    base = np.zeros((size, size, 3), dtype=np.uint8)
    for i in range(n_cells):
        for j in range(n_cells):
            base[i * block:(i + 1) * block, j * block:(j + 1) * block] = terrain[i, j]

    frames = np.zeros((n_frames, size, size, 3), dtype=np.uint8)
    fg_mask = np.zeros((n_frames, size, size), dtype=bool)
    mob_color = (30, 200, 30)
    for t in range(n_frames):
        img = base.copy()
        mx = int((t * 0.9) % (size + 6)) - 6
        my = size // 2 + int(6 * np.sin(t / 15))
        cv2.rectangle(img, (mx, my), (mx + 5, my + 5), mob_color, -1)
        mask = np.zeros((size, size), dtype=np.uint8)
        cv2.rectangle(mask, (mx, my), (mx + 5, my + 5), 1, -1)
        frames[t] = img
        fg_mask[t] = mask.astype(bool)
    return frames, fg_mask


def gen_multi_object(n_frames=240, size=64, bg=(10, 10, 20), n_objects=4):
    rng = np.random.RandomState(3)
    frames = np.zeros((n_frames, size, size, 3), dtype=np.uint8)
    fg_mask = np.zeros((n_frames, size, size), dtype=bool)
    pos = rng.uniform(10, size - 10, size=(n_objects, 2))
    vel = rng.uniform(-1.2, 1.2, size=(n_objects, 2))
    colors = [tuple(int(c) for c in rng.randint(80, 255, 3)) for _ in range(n_objects)]
    radii = rng.randint(3, 6, size=n_objects)
    for t in range(n_frames):
        img = _blank(size, size, bg)
        mask = np.zeros((size, size), dtype=np.uint8)
        for i in range(n_objects):
            pos[i] += vel[i]
            for d in (0, 1):
                if pos[i, d] < radii[i] or pos[i, d] > size - radii[i]:
                    vel[i, d] *= -1
                    pos[i, d] = np.clip(pos[i, d], radii[i], size - radii[i])
            c = (int(pos[i, 0]), int(pos[i, 1]))
            cv2.circle(img, c, int(radii[i]), colors[i], -1)
            cv2.circle(mask, c, int(radii[i]), 1, -1)
        frames[t] = img
        fg_mask[t] = mask.astype(bool)
    return frames, fg_mask


def load_video_frames(path, frame_size=64, n_frames=None):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Video not found: {path}")
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError("Could not open video")
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (frame_size, frame_size))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame.astype(np.uint8))
        if n_frames is not None and len(frames) >= n_frames:
            break
    cap.release()
    if len(frames) == 0:
        raise ValueError("No frames decoded from video")
    return np.stack(frames, axis=0)


def estimate_motion_fg_mask(frames, diff_threshold=18, dilate=1):
    n = frames.shape[0]
    gray = frames.mean(axis=-1)
    mask = np.zeros((n, frames.shape[1], frames.shape[2]), dtype=bool)
    for t in range(n):
        prev_t = max(0, t - 1)
        next_t = min(n - 1, t + 1)
        diff = np.abs(gray[t].astype(np.float32) - gray[prev_t].astype(np.float32))
        diff += np.abs(gray[next_t].astype(np.float32) - gray[t].astype(np.float32))
        m = diff > diff_threshold
        if dilate > 0:
            m = cv2.dilate(m.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=dilate).astype(bool)
        mask[t] = m
    return mask


SCENE_REGISTRY = {
    "pendulum": gen_pendulum,
    "cars": gen_cars,
    "blockworld": gen_blockworld,
    "multi_object": gen_multi_object,
}


def get_scene(name, n_frames=240, size=64, video_path=None):
    if name == "video":
        if video_path is None:
            raise ValueError("video_path required for scene 'video'")
        frames = load_video_frames(video_path, frame_size=size, n_frames=n_frames)
        fg_mask = estimate_motion_fg_mask(frames)
        return frames, fg_mask
    if name not in SCENE_REGISTRY:
        raise ValueError(f"Unknown scene '{name}'. Options: {list(SCENE_REGISTRY) + ['video']}")
    return SCENE_REGISTRY[name](n_frames=n_frames, size=size)
