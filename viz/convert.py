"""Episode logs (one track, several methods) -> the data file the Kiln game scene reads.

World: CarRacing units, y up, heading CCW from +x (body), slip = path - body.
Scene data: canvas px, y down, 100 px = 1 scene unit, centred on (800, 470);
`head` is the direction of travel and `slip` the body's angle off it (Kiln section 22).
"""
import numpy as np

from cars.eval.episode_log import read_episode

PX_PER_UNIT = 4.8          # road 2 x 6.67 units -> 64 px, the Kiln road width
CENTRE = np.array([800.0, 470.0])
RESAMPLE_PX = 4.0
KERB_CURV = 0.004          # rad per px; outside of bends sharper than this gets kerbs


def _resample_closed(points: np.ndarray, step: float) -> np.ndarray:
    closed = np.vstack([points, points[:1]])
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(closed, axis=0), axis=1))]
    s = np.arange(0, d[-1], step)
    return np.c_[np.interp(s, d, closed[:, 0]), np.interp(s, d, closed[:, 1])]


def _smooth_closed(a: np.ndarray, k: int) -> np.ndarray:
    ker = np.ones(2 * k + 1) / (2 * k + 1)
    return np.convolve(np.r_[a[-k:], a, a[:k]], ker, "valid")


def _geometry(centre_px: np.ndarray):
    t = np.roll(centre_px, -1, axis=0) - centre_px
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    t = np.c_[_smooth_closed(t[:, 0], 3), _smooth_closed(t[:, 1], 3)]
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    normal = np.c_[-t[:, 1], t[:, 0]]
    ang = np.unwrap(np.arctan2(t[:, 1], t[:, 0]))
    curv = _smooth_closed(np.gradient(ang) / RESAMPLE_PX, 6)
    return normal, curv


def _place_stand(centre: np.ndarray, normal: np.ndarray, curv: np.ndarray, hw: float) -> dict:
    """The grandstand goes beside the straightest 40-point stretch with open ground on that side."""
    n = len(centre)
    window = np.array([np.abs(curv[(i + np.arange(40)) % n]).sum() for i in range(n)])
    for i in np.argsort(window):
        mid = (i + 20) % n
        for side in (1, -1):
            spot = centre[mid] + normal[mid] * side * (hw + 62)
            if np.min(np.linalg.norm(centre - spot, axis=1)) > hw + 45:
                return {"i": int(i), "side": side}
    return {"i": 0, "side": 1}


def convert(runs: list[dict], hot: str, meta: dict) -> dict:
    """runs: [{"key", "name", "tag", "log"}]; all logs must share one track (same seed and generator)."""
    episodes = {r["key"]: read_episode(r["log"]) for r in runs}
    first = next(iter(episodes.values()))
    seeds = {e["meta"]["seed"] for e in episodes.values()}
    if len(seeds) != 1:
        raise ValueError(f"runs are on different tracks: {seeds}")

    world = np.asarray(first["track"]["centre"])
    lo, hi = world.min(axis=0), world.max(axis=0)
    mid = (lo + hi) / 2
    to_px = lambda x, y: np.c_[(np.asarray(x) - mid[0]) * PX_PER_UNIT + CENTRE[0],
                               -(np.asarray(y) - mid[1]) * PX_PER_UNIT + CENTRE[1]]
    hw_px = float(first["track"]["half_width"][0]) * PX_PER_UNIT

    # Start the resampled centre line at tile 0 (the start line) so cen[0] is the start.
    centre = _resample_closed(to_px(world[:, 0], world[:, 1]), RESAMPLE_PX)
    normal, curv = _geometry(centre)
    left, right = centre + normal * hw_px, centre - normal * hw_px
    span = (hi - lo) * PX_PER_UNIT / 100

    tile_len = len(world)
    out_runs = {}
    for r in runs:
        e = episodes[r["key"]]
        s = e["steps"]
        xy = to_px(s["x"], s["y"])
        heading, slip = np.asarray(s["heading"]), np.asarray(s["slip_angle"])
        # progress in tiles, unwrapped over the lap, monotone (for the order and the gaps)
        tiles = np.unwrap(np.asarray(s["tile_idx"], float) / tile_len * 2 * np.pi) / (2 * np.pi) * tile_len
        tiles = np.maximum.accumulate(tiles - tiles[0])
        finish = e["events"]["finish_step"]
        out_runs[r["key"]] = {
            "name": r["name"], "tag": r["tag"],
            "t": np.round(s["t"], 4).tolist(),
            "x": np.round(xy[:, 0], 2).tolist(), "y": np.round(xy[:, 1], 2).tolist(),
            "v": np.round(np.asarray(s["speed"]) * PX_PER_UNIT, 1).tolist(),
            "head": np.round(-(heading + slip), 4).tolist(),
            "slip": np.round(slip, 4).tolist(),
            "steer": np.round(s["steer"], 3).tolist(),
            "prog": np.round(tiles, 2).tolist(),
            "hits": [i - 1 for i in e["events"]["off_track_events"]],
            "lap": float(s["t"][-1]), "finished": finish is not None,
            "summary": e["summary"],
        }

    crash = None
    for k in [hot] + [r["key"] for r in runs if r["key"] != hot]:
        if out_runs[k]["hits"]:
            crash = {"run": k, "i": out_runs[k]["hits"][0]}
            break

    return {
        "meta": meta | {"seed": first["meta"]["seed"], "split": first["meta"]["split"]},
        "order": [r["key"] for r in runs], "hot": hot,
        "center": np.round(centre, 2).tolist(), "left": np.round(left, 2).tolist(),
        "right": np.round(right, 2).tolist(), "N": np.round(normal, 4).tolist(),
        "curv": np.round(curv, 5).tolist(),
        "kerb": (np.abs(curv) > KERB_CURV).astype(int).tolist(),
        "width": 2 * hw_px,
        "slab": [round(float(span[0]) + 1.6, 2), round(float(span[1]) + 1.6, 2)],
        "stand": _place_stand(centre, normal, curv, hw_px),
        "bend": int(np.argmax(np.abs(curv))),
        "crash": crash, "runs": out_runs,
    }
