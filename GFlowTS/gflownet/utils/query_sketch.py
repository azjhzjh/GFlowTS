import numpy as np
from typing import Tuple, Optional

class QuerySketch:

    __slots__ = ['cells', 'grid_size', 'x_range', 'y_range', 't_range',
                 '_num_updates', '_last_point', '_bbox']

    def __init__(
        self,
        grid_size: int = 64,
        x_range: Tuple[float, float] = (0.0, 1.0),
        y_range: Tuple[float, float] = (0.0, 1.0),
        t_range: Tuple[float, float] = (0.0, 1.0)
    ):

        self.grid_size = grid_size
        self.x_range = x_range
        self.y_range = y_range
        self.t_range = t_range

        self.cells = set()

        self._num_updates = 0
        self._last_point = None
        self._bbox = {
            'x_min': float('inf'), 'x_max': float('-inf'),
            'y_min': float('inf'), 'y_max': float('-inf'),
            't_min': float('inf'), 't_max': float('-inf')
        }

    def reset(self):

        self.cells.clear()
        self._num_updates = 0
        self._last_point = None
        self._bbox = {
            'x_min': float('inf'), 'x_max': float('-inf'),
            'y_min': float('inf'), 'y_max': float('-inf'),
            't_min': float('inf'), 't_max': float('-inf')
        }

    def update_with_point(self, point: np.ndarray):

        cell_id = self._spatial_hash(point)
        self.cells.add(cell_id)

        self._num_updates += 1
        self._last_point = point.copy() if isinstance(point, np.ndarray) else np.array(point)

        x, y, t = float(point[0]), float(point[1]), float(point[2])
        self._bbox['x_min'] = min(self._bbox['x_min'], x)
        self._bbox['x_max'] = max(self._bbox['x_max'], x)
        self._bbox['y_min'] = min(self._bbox['y_min'], y)
        self._bbox['y_max'] = max(self._bbox['y_max'], y)
        self._bbox['t_min'] = min(self._bbox['t_min'], t)
        self._bbox['t_max'] = max(self._bbox['t_max'], t)

    def _spatial_hash(self, point: np.ndarray) -> int:

        x, y, t = float(point[0]), float(point[1]), float(point[2])

        x_norm = (x - self.x_range[0]) / (self.x_range[1] - self.x_range[0] + 1e-9)
        y_norm = (y - self.y_range[0]) / (self.y_range[1] - self.y_range[0] + 1e-9)

        xi = int(np.clip(x_norm * self.grid_size, 0, self.grid_size - 1))
        yi = int(np.clip(y_norm * self.grid_size, 0, self.grid_size - 1))

        return xi * self.grid_size + yi

    def get_coverage_ratio(self) -> float:

        total_cells = self.grid_size * self.grid_size
        return len(self.cells) / total_cells

    def get_bbox_features(self) -> np.ndarray:

        if self._num_updates == 0:
            return np.zeros(6, dtype=np.float32)

        x_span = self._bbox['x_max'] - self._bbox['x_min']
        y_span = self._bbox['y_max'] - self._bbox['y_min']
        t_span = self._bbox['t_max'] - self._bbox['t_min']

        x_center = (self._bbox['x_max'] + self._bbox['x_min']) / 2
        y_center = (self._bbox['y_max'] + self._bbox['y_min']) / 2
        t_center = (self._bbox['t_max'] + self._bbox['t_min']) / 2

        return np.array([x_span, y_span, t_span, x_center, y_center, t_center],
                       dtype=np.float32)

    def to_tensor(self) -> np.ndarray:

        total_cells = self.grid_size * self.grid_size

        features = [
            self.get_coverage_ratio(),
            len(self.cells) / total_cells,
            min(1.0, self._num_updates / 100.0),
        ]

        if self._last_point is not None:
            features.extend([
                float(self._last_point[0]),
                float(self._last_point[1]),
                float(self._last_point[2])
            ])
        else:
            features.extend([0.0, 0.0, 0.0])

        features.extend(self.get_bbox_features().tolist())

        return np.array(features, dtype=np.float32)

    @staticmethod
    def dim() -> int:

        return 12

    def __repr__(self) -> str:
        return (f"QuerySketch(grid_size={self.grid_size}, "
                f"cells={len(self.cells)}, points={self._num_updates})")

class ConstantSizeState:

    __slots__ = ['last_kept_point', 'sketch', 'num_points', 'budget_left',
                 'current_idx', 'total_points']

    def __init__(
        self,
        total_points: int,
        grid_size: int = 64,
        target_compression: float = 0.1
    ):
        self.last_kept_point = None
        self.sketch = QuerySketch(grid_size=grid_size)
        self.num_points = 0
        self.budget_left = 1.0
        self.current_idx = 0
        self.total_points = total_points

    def reset(self, keep_first: bool = True, first_point: Optional[np.ndarray] = None):

        self.sketch.reset()
        self.num_points = 0
        self.budget_left = 1.0
        self.current_idx = 0
        self.last_kept_point = None

        if keep_first and first_point is not None:
            self.add_point(first_point)

    def add_point(self, point: np.ndarray):

        self.sketch.update_with_point(point)
        self.last_kept_point = point.copy() if isinstance(point, np.ndarray) else np.array(point)
        self.num_points += 1
        self.budget_left = max(0, 1.0 - self.num_points / max(1, self.total_points))

    def to_tensor(self) -> np.ndarray:

        features = []

        if self.last_kept_point is not None:
            features.extend([
                float(self.last_kept_point[0]),
                float(self.last_kept_point[1]),
                float(self.last_kept_point[2])
            ])
        else:
            features.extend([0.0, 0.0, 0.0])

        features.extend([
            min(1.0, self.num_points / max(1, self.total_points)),
            self.budget_left,
            self.current_idx / max(1, self.total_points)
        ])

        features.extend(self.sketch.to_tensor().tolist())

        return np.array(features, dtype=np.float32)

    @staticmethod
    def dim() -> int:

        return 3 + 3 + QuerySketch.dim()

    def __repr__(self) -> str:
        return (f"ConstantSizeState(points={self.num_points}, "
                f"idx={self.current_idx}/{self.total_points}, "
                f"budget={self.budget_left:.2f})")

class QueryCoverageSketch:

    __slots__ = ['cells', 'grid_size', 'x_range', 'y_range', 't_range']

    def __init__(
        self,
        grid_size: int = 64,
        x_range: Tuple[float, float] = (0.0, 1.0),
        y_range: Tuple[float, float] = (0.0, 1.0),
        t_range: Tuple[float, float] = (0.0, 1.0)
    ):

        self.grid_size = grid_size
        self.x_range = x_range
        self.y_range = y_range
        self.t_range = t_range
        self.cells = set()

    def reset(self):

        self.cells.clear()

    def add_point(self, point: np.ndarray):

        cell = self._query_aware_cell(point)
        self.cells.add(cell)

    def _query_aware_cell(self, point: np.ndarray) -> int:

        x, y, t = float(point[0]), float(point[1]), float(point[2])

        x_norm = (x - self.x_range[0]) / (self.x_range[1] - self.x_range[0] + 1e-9)
        y_norm = (y - self.y_range[0]) / (self.y_range[1] - self.y_range[0] + 1e-9)
        t_norm = (t - self.t_range[0]) / (self.t_range[1] - self.t_range[0] + 1e-9)

        xi = np.clip(x_norm * self.grid_size, 0, self.grid_size - 1).astype(np.int32)
        yi = np.clip(y_norm * self.grid_size, 0, self.grid_size - 1).astype(np.int32)
        ti = np.clip(t_norm * (self.grid_size // 4), 0, (self.grid_size // 4) - 1).astype(np.int32)

        return xi * self.grid_size * (self.grid_size // 4) + yi * (self.grid_size // 4) + ti

    def add_points(self, points: np.ndarray):

        if len(points) == 0:
            return

        x = points[:, 0]
        y = points[:, 1]
        t = points[:, 2]

        x_norm = (x - self.x_range[0]) / (self.x_range[1] - self.x_range[0] + 1e-9)
        y_norm = (y - self.y_range[0]) / (self.y_range[1] - self.y_range[0] + 1e-9)
        t_norm = (t - self.t_range[0]) / (self.t_range[1] - self.t_range[0] + 1e-9)

        xi = np.clip(x_norm * self.grid_size, 0, self.grid_size - 1).astype(np.int32)
        yi = np.clip(y_norm * self.grid_size, 0, self.grid_size - 1).astype(np.int32)
        ti = np.clip(t_norm * (self.grid_size // 4), 0, (self.grid_size // 4) - 1).astype(np.int32)

        cell_ids = xi * self.grid_size * (self.grid_size // 4) + yi * (self.grid_size // 4) + ti

        self.cells.update(cell_ids.tolist())

    def _estimate_segment_steps(self, p1: np.ndarray, p2: np.ndarray) -> int:
        x1_norm = (p1[0] - self.x_range[0]) / (self.x_range[1] - self.x_range[0] + 1e-9)
        y1_norm = (p1[1] - self.y_range[0]) / (self.y_range[1] - self.y_range[0] + 1e-9)
        t1_norm = (p1[2] - self.t_range[0]) / (self.t_range[1] - self.t_range[0] + 1e-9)

        x2_norm = (p2[0] - self.x_range[0]) / (self.x_range[1] - self.x_range[0] + 1e-9)
        y2_norm = (p2[1] - self.y_range[0]) / (self.y_range[1] - self.y_range[0] + 1e-9)
        t2_norm = (p2[2] - self.t_range[0]) / (self.t_range[1] - self.t_range[0] + 1e-9)

        dist = max(abs(x2_norm - x1_norm), abs(y2_norm - y1_norm), abs(t2_norm - t1_norm))
        return max(2, int(dist * self.grid_size * 2))

    def add_segment_with_step_cap(self, p1: np.ndarray, p2: np.ndarray, max_steps: int = 200):
        num_steps = min(self._estimate_segment_steps(p1, p2), max(2, int(max_steps)))
        alphas = np.linspace(0, 1, num_steps).reshape(-1, 1)
        points = p1 * (1 - alphas) + p2 * alphas
        self.add_points(points)

    def add_segment(self, p1: np.ndarray, p2: np.ndarray, max_steps: int = 200):

        x1_norm = (p1[0] - self.x_range[0]) / (self.x_range[1] - self.x_range[0] + 1e-9)
        y1_norm = (p1[1] - self.y_range[0]) / (self.y_range[1] - self.y_range[0] + 1e-9)
        t1_norm = (p1[2] - self.t_range[0]) / (self.t_range[1] - self.t_range[0] + 1e-9)

        x2_norm = (p2[0] - self.x_range[0]) / (self.x_range[1] - self.x_range[0] + 1e-9)
        y2_norm = (p2[1] - self.y_range[0]) / (self.y_range[1] - self.y_range[0] + 1e-9)
        t2_norm = (p2[2] - self.t_range[0]) / (self.t_range[1] - self.t_range[0] + 1e-9)

        dist = max(abs(x2_norm - x1_norm), abs(y2_norm - y1_norm), abs(t2_norm - t1_norm))

        num_steps = max(2, int(dist * self.grid_size * 2))

        if num_steps > 200:
            num_steps = 200

        alphas = np.linspace(0, 1, num_steps).reshape(-1, 1)
        points = p1 * (1 - alphas) + p2 * alphas

        self.add_points(points)

    def size(self) -> int:

        return len(self.cells)

    def intersection(self, other: 'QueryCoverageSketch') -> set:

        return self.cells.intersection(other.cells)

    def __repr__(self) -> str:
        return f"QCS(cells={len(self.cells)}, grid={self.grid_size})"

def f1_lower_bound(qcs_compressed: QueryCoverageSketch, qcs_original: QueryCoverageSketch) -> float:

    if qcs_compressed.size() == 0 or qcs_original.size() == 0:
        return 0.0

    inter = qcs_compressed.intersection(qcs_original)
    denominator = qcs_compressed.size() + qcs_original.size()

    if denominator == 0:
        return 0.0

    return 2.0 * len(inter) / denominator
