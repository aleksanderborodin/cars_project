"""Explicit privileged continuation of an unchanged visible reference."""
import numpy as np


def extend_true_reference(visible: np.ndarray, centre: np.ndarray,
                          length: int = 100) -> np.ndarray:
    """Append one-metre arc samples on the directed true track after the exit.

    Preserve every visible sample. Anchor its last point to the supplied loop,
    then follow directed segments; no global progress or evaluation state is used.
    The loop is privileged input. This function is never called by local methods.
    """
    visible, centre = np.asarray(visible, float), np.asarray(centre, float)
    if (visible.ndim != 2 or visible.shape[1] != 2 or len(visible) < 2
            or centre.ndim != 2 or centre.shape[1] != 2 or len(centre) < 3
            or not np.isfinite(visible).all() or not np.isfinite(centre).all()
            or isinstance(length, bool) or not isinstance(length, int) or length < 1):
        raise ValueError("Finite visible path, directed loop and positive integer length required")
    delta = np.roll(centre, -1, axis=0) - centre
    lengths = np.linalg.norm(delta, axis=1)
    usable = lengths > 1e-10
    if not usable.any():
        raise ValueError("Degenerate true track")
    fraction = np.clip(np.divide(np.sum((visible[-1] - centre) * delta, axis=1), lengths**2,
                                out=np.zeros(len(centre)), where=usable), 0., 1.)
    projected = centre + fraction[:, None] * delta
    matches = np.flatnonzero(usable & (np.linalg.norm(projected - visible[-1], axis=1) <= 1e-6))
    if not len(matches):
        raise ValueError("Visible endpoint does not lie on the true track")
    tangent = visible[-1] - visible[-2]
    if np.linalg.norm(tangent) < 1e-10:
        raise ValueError("Visible endpoint needs a direction")
    tangent /= np.linalg.norm(tangent)
    # At a vertex choose its outgoing segment. Geometric tie breaking avoids
    # depending on which vertex the map array happens to list first.
    index = min(matches, key=lambda i: (
        fraction[i] >= 1. - 1e-9, -float(tangent @ delta[i] / lengths[i]),
        tuple(centre[i]), tuple(centre[(i + 1) % len(centre)])))
    points, arc = [visible[-1].copy()], [0.]
    maximum_segments = len(centre) * (int(np.ceil(length / lengths.sum())) + 2)
    for step in range(maximum_segments):
        point = centre[(index + step + 1) % len(centre)]
        distance = float(np.linalg.norm(point - points[-1]))
        if distance <= 1e-10:
            continue
        points.append(point)
        arc.append(arc[-1] + distance)
        if arc[-1] >= length:
            break
    if arc[-1] < length:
        raise ValueError("Unable to construct the declared reference extent")
    points = np.asarray(points)
    samples = np.arange(1., length + 1.)
    tail = np.column_stack([np.interp(samples, arc, points[:, axis]) for axis in (0, 1)])
    return np.concatenate((visible, tail))
