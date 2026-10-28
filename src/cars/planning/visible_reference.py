"""Privileged current-geometry reference, clipped before path construction."""
import numpy as np


def visible_reference(centre: np.ndarray, radius: float, resolution: float) -> np.ndarray | None:
    """Nearest contiguous directed centreline inside the sensor circle.

    Clips each segment, then joins only visible endpoints. Hidden arcs cannot
    connect disconnected visible pieces. Direction is privileged supervision,
    not inferred from the binary sensor. No global progress/index is returned.
    """
    centre = np.asarray(centre, dtype=float)
    if (centre.ndim != 2 or centre.shape[1] != 2 or len(centre) < 3
            or not np.isfinite(centre).all() or radius <= 0 or resolution <= 0):
        raise ValueError("Finite closed centreline and positive geometry scales required")
    pieces: list[list[np.ndarray]] = []
    for start, end in zip(centre, np.roll(centre, -1, axis=0), strict=True):
        delta = end - start
        length2 = float(delta @ delta)
        if length2 < 1e-12:
            continue
        b = float(start @ delta)
        discriminant = b * b - length2 * (float(start @ start) - radius * radius)
        if discriminant <= 0:
            continue
        root = np.sqrt(discriminant)
        low, high = max(0., (-b - root) / length2), min(1., (-b + root) / length2)
        if high - low <= 1e-10:
            continue
        a, z = start + low * delta, start + high * delta
        if pieces and np.linalg.norm(pieces[-1][-1] - a) < 1e-7:
            pieces[-1].append(z)
        else:
            pieces.append([a, z])
    if not pieces:
        return None
    if len(pieces) > 1 and np.linalg.norm(pieces[-1][-1] - pieces[0][0]) < 1e-7:
        pieces[0] = pieces.pop()[:-1] + pieces[0]

    def distance(points: list[np.ndarray]) -> float:
        p = np.asarray(points)
        delta = np.diff(p, axis=0)
        t = np.clip(-np.sum(p[:-1] * delta, axis=1) / np.sum(delta**2, axis=1), 0, 1)
        return float(np.min(np.sum((p[:-1] + t[:, None] * delta)**2, axis=1)))

    path = np.asarray(min(pieces, key=distance))
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    if arc[-1] < resolution:
        return None
    samples = np.r_[np.arange(0., arc[-1], resolution), arc[-1]]
    return np.column_stack([np.interp(samples, arc, path[:, axis]) for axis in (0, 1)])
