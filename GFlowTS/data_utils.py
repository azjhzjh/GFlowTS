import numpy as np

def to_traj(file):
    traj = []
    f = open(file)
    for line in f:
        temp = line.strip().split(' ')
        if len(temp) < 3:
            continue
        traj.append([float(temp[0]), float(temp[1]), int(float(temp[2]))])
    f.close()
    return traj

def rdp_simplify(trajectory: np.ndarray, epsilon: float) -> np.ndarray:

    if len(trajectory) < 3:
        return trajectory

    dmax = 0.0
    index = 0
    end = len(trajectory) - 1

    start_pt = trajectory[0, :2]
    end_pt = trajectory[-1, :2]

    line_vec = end_pt - start_pt
    line_len_sq = np.dot(line_vec, line_vec)

    if line_len_sq < 1e-10:

        dists = np.linalg.norm(trajectory[1:-1, :2] - start_pt, axis=1)
    else:

        pts_vec = trajectory[1:-1, :2] - start_pt

        t = np.dot(pts_vec, line_vec) / line_len_sq
        t = np.clip(t, 0, 1)

        projection = start_pt + np.outer(t, line_vec)

        dists = np.linalg.norm(trajectory[1:-1, :2] - projection, axis=1)

    if len(dists) > 0:
        index = np.argmax(dists) + 1
        dmax = dists[index - 1]

    if dmax > epsilon:

        rec_results1 = rdp_simplify(trajectory[:index+1], epsilon)
        rec_results2 = rdp_simplify(trajectory[index:], epsilon)

        return np.vstack((rec_results1[:-1], rec_results2))
    else:
        return np.vstack((trajectory[0], trajectory[-1]))
