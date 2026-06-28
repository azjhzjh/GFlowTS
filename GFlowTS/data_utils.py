import numpy as np
from collections import defaultdict
import faiss
from rtree import index
import math
import heapq
import os
from config import BALLTREE_CONFIG, FEATURE_CONFIG, CONTINUITY_CONFIG
try:
    from feature_extraction import TrajectoryFeatureExtractor
except ImportError:
    TrajectoryFeatureExtractor = None


#Main.include('F:/query/GFlowNet-one/call.jl')

class CustomBallTree:
    """自定义BallTree实现，基于主成分分析的递归分割"""
    
    def __init__(self, points, traj_ids=None, use_semantic_split=True,
                 min_leaf_size=5, semantic_weights=(0.5, 0.5), config=None):
        self.points = np.array(points)
        self.n_points = len(self.points)
        self.dimension = self.points.shape[1] if self.n_points > 0 else 0
        self.use_semantic_split = use_semantic_split
        self.min_leaf_size = min_leaf_size
        self.semantic_weights = semantic_weights
        
        # 加载配置
        self.config = config if config else BALLTREE_CONFIG
        self.N_leaf = self.config.get('N_leaf', 5)
        self.maxDepth = self.config.get('maxDepth', 50)
        self.k_radius = self.config.get('k_radius', 2.0)
        self.lambda_depth = self.config.get('lambda_depth', 0.1)
        self.large_node_threshold = self.config.get('large_node_threshold', 500)
        self.sample_ratio = self.config.get('sample_ratio', 0.2)
        self.min_sample_size = self.config.get('min_sample_size', 200)
        
        # 特征提取器（如果可用）
        self.feature_extractor = None
        if TrajectoryFeatureExtractor is not None:
            self.feature_extractor = TrajectoryFeatureExtractor()
        
        # 语义权重缓存
        self.point_semantic_weights = None
        
        # 兼容性检查：如果没有traj_ids或者不使用语义分割，使用原始PCA方法
        if not use_semantic_split or traj_ids is None:
            self.use_legacy_mode = True
            self.point_to_traj = None
            self.point_directions = None
            self.point_speeds = None
        else:
            self.use_legacy_mode = False
            self._precompute_trajectory_features(traj_ids)
            # 计算语义权重
            if self.feature_extractor is not None:
                self._compute_semantic_weights()
        
        # 构建树结构
        self.root = self._build_tree(self.points, list(range(self.n_points)))
    
    def _precompute_trajectory_features(self, traj_ids):
        """预计算轨迹特征（方向、速度等）- 全向量化极致性能版"""
        self.point_to_traj = traj_ids
        n_points = len(self.points)
        self.point_directions = np.zeros((n_points, 3))
        self.point_speeds = np.zeros(n_points)
        
        if n_points < 2: return
        
        # 1. 向量化排序：先按轨迹ID排，轨迹内按时间排
        traj_ids_arr = np.array(traj_ids)
        times = self.points[:, 2]
        # lexsort(keys) 其中 keys[0] 是次要键，keys[1] 是主要键
        sort_idx = np.lexsort((times, traj_ids_arr))
        
        sorted_points = self.points[sort_idx]
        sorted_traj_ids = traj_ids_arr[sort_idx]
        
        # 2. 识别轨迹边界
        # traj_change[i] 为 True 表示 i 和 i-1 属于不同轨迹
        traj_change = np.concatenate(([True], sorted_traj_ids[1:] != sorted_traj_ids[:-1], [True]))
        
        # 3. 计算位移向量 (Next - Current)
        diffs = np.zeros_like(sorted_points)
        diffs[:-1] = sorted_points[1:] - sorted_points[:-1]
        
        # 屏蔽跨轨迹的差分
        valid_diff_mask = ~traj_change[1:-1]
        
        # 4. 计算方向 (向量化)
        # 这里的逻辑：每个点的方向是其前后两段位移的平均
        # 为了速度，我们直接用 forward 差分并处理边界
        directions = np.zeros_like(sorted_points)
        directions[:-1] = diffs[:-1]
        # 最后一个点使用前一个点的方向
        directions[1:] += diffs[:-1] 
        # 处理跨轨迹边界（不要累加不同轨迹的方向）
        reset_mask = traj_change[1:-1]
        # 如果 i 是新轨迹起点，i-1 的 next 差分不应加到 i 的方向上
        # ...这里稍作简化：直接使用中心差分
        
        # 归一化
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        valid_norm = norms[:, 0] > 1e-10
        directions[valid_norm] /= norms[valid_norm]
        
        # 5. 计算速度 (向量化)
        dist_2d = np.linalg.norm(diffs[:-1, :2], axis=1)
        time_diff = np.diff(sorted_points[:, 2])
        
        speeds = np.zeros(n_points)
        # 避免除零
        v_mask = (time_diff > 1e-8) & (~traj_change[1:-1])
        speeds[:-1][v_mask] = dist_2d[v_mask] / time_diff[v_mask]
        # 补偿每条轨迹的最后一个点
        speeds[1:] += np.where(traj_change[1:-1], 0, speeds[:-1]) # 简单传递
        
        # 6. 写回原顺序
        inv_sort_idx = np.argsort(sort_idx)
        self.point_directions = directions[inv_sort_idx]
        self.point_speeds = speeds[inv_sort_idx]
        
        # 7. 预计算关键点掩码 (Critical Points Mask)
        # 关键点包括：每条轨迹的起点、终点、以及低速停留点
        
        # 7.1 轨迹边界点 (在 sorted_idx顺序下)
        is_boundary_sorted = traj_change[1:] | traj_change[:-1] # 起点或终点
        
        # 7.2 停留点 (低速点)
        # 计算全局低速阈值 (25分位数)
        speed_threshold = np.percentile(speeds, 25)
        is_stay_sorted = speeds < speed_threshold
        
        # 合并掩码
        is_critical_sorted = is_boundary_sorted | is_stay_sorted
        
        # 映射回原始顺序
        self.is_critical_mask = is_critical_sorted[inv_sort_idx]
    
    def _determine_split_strategy_enhanced(self, points, indices, depth):
        """增强的分割策略判断（包含时间分割）"""
        # 计算特征指标
        time_span = points[:, 2].max() - points[:, 2].min()
        time_span_days = time_span / (24 * 3600)  # 转换为天数
        
        # 计算方向一致性
        if not self.use_legacy_mode and self.point_directions is not None:
            directions = self.point_directions[indices]
            if len(directions) > 0:
                # 计算Rayleigh statistic: R = ||Σ u_i|| / Σ||u_i||
                sum_vectors = np.sum(directions, axis=0)
                norm_sum = np.linalg.norm(sum_vectors)
                sum_norms = np.sum(np.linalg.norm(directions, axis=1))
                if sum_norms > 1e-10:
                    direction_consistency = norm_sum / sum_norms
                else:
                    direction_consistency = 0.0
            else:
                direction_consistency = 0.0
        else:
            direction_consistency = 0.0
        
        n_points = len(points)
        M_time = FEATURE_CONFIG.get('M_time', 100)
        time_window_days = FEATURE_CONFIG.get('time_window_days', 10)
        R_thres = FEATURE_CONFIG.get('R_thres', 0.85)
        
        # 动态选择分割方向（优先级规则）
        # 1. 时间跨度 T > 10 days 且点数 > M_time → 时间窗口分割
        if time_span_days > time_window_days and n_points > M_time:
            return "TIME_SPLIT"
        # 2. 方向一致性 R > R_thres → 语义方向分割
        elif direction_consistency > R_thres and n_points > 50:
            return "SEMANTIC_SPLIT"
        # 3. 其他 → PCA分割
        else:
            return "SPATIAL_SPLIT"
    
    def _determine_split_strategy(self, points, indices, depth):
        """根据数据特征确定分割策略（保持向后兼容）"""
        return self._determine_split_strategy_enhanced(points, indices, depth)
    
    def _select_split_direction(self, points, indices, strategy_type, depth):
        """选择最佳分割方向 - 优化版本"""
        centered_points = points - points.mean(axis=0)
        
        if len(points) < 50:
            # 小点数直接使用PCA
            cov_matrix = np.cov(centered_points.T)
            if np.allclose(cov_matrix, 0):
                return np.array([1, 0, 0], dtype=np.float64)
            eigenvals, eigenvecs = np.linalg.eig(cov_matrix)
            return eigenvecs[:, np.argmax(eigenvals)].real.astype(np.float64)
        
        # 优化3：根据策略类型限制候选方向，减少评估次数
        candidates = []
        
        # 总是计算PCA方向（最重要）
        cov_matrix = np.cov(centered_points.T)
        if not np.allclose(cov_matrix, 0):
            eigenvals, eigenvecs = np.linalg.eig(cov_matrix)
            pca_dir = eigenvecs[:, np.argmax(eigenvals)].real.astype(np.float64)
            candidates.append(pca_dir)
        
        # 根据策略类型选择其他候选方向
        if strategy_type == "SEMANTIC_SPLIT" and not self.use_legacy_mode:
            # 语义分割：添加轨迹方向
            avg_direction = self.point_directions[indices].mean(axis=0)
            norm = np.linalg.norm(avg_direction)
            if norm > 0.1:  # 只添加有意义的方向
                candidates.append((avg_direction / norm).astype(np.float64))
        
        if strategy_type == "SPATIAL_SPLIT" or len(candidates) < 2:
            # 空间分割：添加坐标轴方向（只添加主要的）
            # 计算每个轴的方差，只添加方差大的轴
            variances = np.var(centered_points, axis=0)
            max_var_axis = np.argmax(variances)
            axis_dirs = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])]
            candidates.append(axis_dirs[max_var_axis].astype(np.float64))
        
        # 如果没有候选方向，使用PCA
        if len(candidates) == 0:
            return np.array([1, 0, 0], dtype=np.float64)
        
        # 优化4：快速评估 - 只对前2-3个候选方向进行完整评估
        # 先快速评估所有方向的空间紧凑度（计算快）
        quick_scores = []
        for direction in candidates:
            projections = np.dot(centered_points, direction)
            median_proj = np.median(projections)
            left_mask = projections <= median_proj
            right_mask = projections > median_proj
            
            if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                quick_scores.append(-1)
                continue
            
            # 快速紧凑度评估（不使用体积，直接用半径）
            left_center = points[left_mask].mean(axis=0)
            right_center = points[right_mask].mean(axis=0)
            left_radius = np.linalg.norm(points[left_mask] - left_center, axis=1).max()
            right_radius = np.linalg.norm(points[right_mask] - right_center, axis=1).max()
            quick_score = 1.0 / (left_radius + right_radius + 1e-6)
            quick_scores.append(quick_score)
        
        # 选择紧凑度最好的2-3个方向进行完整评估
        if len(candidates) > 3:
            top_indices = np.argsort(quick_scores)[-3:][::-1]  # 选择前3个
            candidates = [candidates[i] for i in top_indices]
        
        # 完整评估剩余候选方向
        best_score = -float('inf')
        best_direction = candidates[0]
        depth_factor = min(depth / 15.0, 1.0)
        
        for direction in candidates:
            projections = np.dot(centered_points, direction)
            median_proj = np.median(projections)
            left_mask = projections <= median_proj
            right_mask = projections > median_proj
            
            if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                continue
            
            # 语义一致性得分（简化评估：仅在 SEMANTIC_SPLIT 下计算，且不计算协方差）
            semantic_score = 1.0
            if strategy_type == "SEMANTIC_SPLIT" and not self.use_legacy_mode:
                # 简化：使用方向的标准差（mean of variances）代替特征分解
                left_vars = self.point_directions[indices][left_mask].var(axis=0).sum()
                right_vars = self.point_directions[indices][right_mask].var(axis=0).sum()
                semantic_score = 1.0 / (left_vars + right_vars + 0.1)
            
            # 空间紧凑度得分（使用半径而不是体积，计算更快）
            left_points_subset = points[left_mask]
            right_points_subset = points[right_mask]
            
            # 进一步加速：只使用采样点计算中心
            left_center = left_points_subset.mean(axis=0)
            right_center = right_points_subset.mean(axis=0)
            
            # 快速计算最大距离
            left_radius = np.max(np.linalg.norm(left_points_subset - left_center, axis=1)) 
            right_radius = np.max(np.linalg.norm(right_points_subset - right_center, axis=1))
            compactness_score = 1.0 / (left_radius + right_radius + 1e-6)
            
            # 综合评分
            if strategy_type == "TIME_SPLIT":
                α, β = 0.2, 0.8
            elif strategy_type == "SEMANTIC_SPLIT":
                α = 0.7 * (1 - depth_factor) + 0.3 * depth_factor
                β = 0.3 * depth_factor + 0.7 * (1 - depth_factor)
            else:
                α, β = 0.3, 0.7
            
            total_score = α * semantic_score + β * compactness_score
            
            if total_score > best_score:
                best_score = total_score
                best_direction = direction
        
        return best_direction
    
    def _hierarchical_sampling(self, points, indices, traj_ids=None):
        """
        分层随机采样（保留轨迹边界点、停留点及随机点）
        
        Args:
            points: 点数组
            indices: 点索引列表
            traj_ids: 轨迹ID列表（可选）
        
        Returns:
            sampled_points: 采样后的点
            sampled_indices: 采样后的索引
        """
        n_points = len(points)
        if n_points <= self.large_node_threshold:
            return points, indices
        
        # 计算采样数量
    def _hierarchical_sampling(self, points, indices, traj_ids=None):
        """
        分层随机采样（向量化极速版）
        不再使用字典分组，而是使用预计算的掩码直接过滤
        """
        n_points = len(points)
        if n_points <= self.large_node_threshold:
            return points, indices
        
        # 计算采样数量
        sample_size = max(self.min_sample_size, int(n_points * self.sample_ratio))
        sample_size = min(sample_size, n_points)
        
        if traj_ids is not None and not self.use_legacy_mode and hasattr(self, 'is_critical_mask'):
            # 1. 快速筛选关键点
            # current_indices 是 indices 在 self.points 中的全局索引
            # 我们需要检查这些全局索引对应的 mask
            
            # 转换 indices 为 numpy array 以支持批量索引
            indices_arr = np.array(indices)
            
            # 获取当前节点内点的关键性掩码
            node_critical_mask = self.is_critical_mask[indices_arr]
            
            # 获取关键点的本地索引 (在 indices 列表中的位置)
            critical_local_indices = np.where(node_critical_mask)[0]
            
            # 如果关键点太多，随机选择一部分；如果太少，全保留并随机补齐
            if len(critical_local_indices) > sample_size:
                # 关键点过多，从中随机选
                selected_local_indices = np.random.choice(critical_local_indices, size=sample_size, replace=False)
            else:
                # 保留所有关键点
                selected_local_indices = critical_local_indices
                
                # 还需要补充的数量
                n_needed = sample_size - len(selected_local_indices)
                if n_needed > 0:
                    # 从非关键点中随机选
                    non_critical_mask = ~node_critical_mask
                    non_critical_local_indices = np.where(non_critical_mask)[0]
                    
                    if len(non_critical_local_indices) > 0:
                        # 确保补充数量不超过剩余数量
                        n_random = min(n_needed, len(non_critical_local_indices))
                        random_local_indices = np.random.choice(non_critical_local_indices, size=n_random, replace=False)
                        
                        # 合并
                        selected_local_indices = np.concatenate((selected_local_indices, random_local_indices))
            
            # 提取采样后的点和全局索引
            # selected_local_indices 包含的是 indices 列表的下标
            sampled_points = points[selected_local_indices]
            # indices_arr[selected_local_indices] 得到对应的全局索引
            sampled_indices = indices_arr[selected_local_indices].tolist()
            
            return sampled_points, sampled_indices
            
        else:
            # 回退到简单随机采样
            sampled_indices = np.random.choice(indices, size=sample_size, replace=False).tolist()
            sampled_points = points[[indices.index(idx) for idx in sampled_indices]]
            return sampled_points, sampled_indices
    
    def _time_window_split(self, points, indices):
        """
        时间窗口分割实现（按时间quantile划分）
        
        Args:
            points: 点数组
            indices: 点索引列表
        
        Returns:
            split_direction: 分割方向（时间轴）
            median_time: 中位数时间
        """
        # 按时间quantile分割
        times = points[:, 2]
        median_time = np.median(times)
        
        # 时间轴方向
        split_direction = np.array([0, 0, 1], dtype=np.float64)
        
        return split_direction, median_time
    
    def _check_node_continuity(self, left_node, right_node):
        """
        子节点连续性检测
        
        Args:
            left_node: 左子节点
            right_node: 右子节点
        
        Returns:
            continuity_score: 连续性得分 (0-1)
            continuity_metrics: 连续性指标字典
        """
        if left_node is None or right_node is None:
            return 0.0, {}
        
        if left_node.center is None or right_node.center is None:
            return 0.0, {}
        
        metrics = {}
        
        # 1. 时间相连性（时间区间 gap < τ_time_gap）
        if hasattr(left_node, 'time_span') and hasattr(right_node, 'time_span'):
            left_time_span = left_node.time_span
            right_time_span = right_node.time_span
            
            if isinstance(left_time_span, tuple) and isinstance(right_time_span, tuple):
                left_max = left_time_span[1]
                right_min = right_time_span[0]
                time_gap = right_min - left_max
                
                tau_time_gap = CONTINUITY_CONFIG.get('tau_time_gap', 600)
                metrics['time_gap'] = time_gap
                metrics['time_continuous'] = time_gap < tau_time_gap
            else:
                metrics['time_continuous'] = False
        else:
            # 从点数据计算时间范围
            if len(left_node.points) > 0 and len(right_node.points) > 0:
                left_times = left_node.points[:, 2]
                right_times = right_node.points[:, 2]
                time_gap = right_times.min() - left_times.max()
                
                tau_time_gap = CONTINUITY_CONFIG.get('tau_time_gap', 600)
                metrics['time_gap'] = time_gap
                metrics['time_continuous'] = time_gap < tau_time_gap
            else:
                metrics['time_continuous'] = False
        
        # 2. 空间接近度（最近点距 < τ_space）
        left_points = left_node.points
        right_points = right_node.points
        
        if len(left_points) > 0 and len(right_points) > 0:
            # 计算最近点距离
            distances = []
            for lp in left_points:
                for rp in right_points:
                    dist = np.linalg.norm(lp[:2] - rp[:2])  # 只考虑空间距离
                    distances.append(dist)
            
            min_distance = min(distances) if distances else float('inf')
            tau_space = CONTINUITY_CONFIG.get('tau_space', 100.0)
            metrics['min_distance'] = min_distance
            metrics['space_continuous'] = min_distance < tau_space
        else:
            metrics['space_continuous'] = False
        
        # 3. 方向/模式一致性（方向一致性差异 < τ_dir）
        if (hasattr(left_node, 'avg_direction') and hasattr(right_node, 'avg_direction') and
            left_node.avg_direction is not None and right_node.avg_direction is not None):
            left_dir = left_node.avg_direction
            right_dir = right_node.avg_direction
            
            # 计算方向差异（使用点积）
            if len(left_dir) >= 2 and len(right_dir) >= 2:
                left_dir_2d = left_dir[:2] / (np.linalg.norm(left_dir[:2]) + 1e-10)
                right_dir_2d = right_dir[:2] / (np.linalg.norm(right_dir[:2]) + 1e-10)
                dir_similarity = np.dot(left_dir_2d, right_dir_2d)
                dir_diff = 1.0 - dir_similarity
                
                tau_dir = CONTINUITY_CONFIG.get('tau_dir', 0.3)
                metrics['direction_diff'] = dir_diff
                metrics['direction_continuous'] = dir_diff < tau_dir
            else:
                metrics['direction_continuous'] = False
        else:
            metrics['direction_continuous'] = False
        
        # 计算综合连续性得分
        continuity_score = 0.0
        if metrics.get('time_continuous', False):
            continuity_score += 0.4
        if metrics.get('space_continuous', False):
            continuity_score += 0.4
        if metrics.get('direction_continuous', False):
            continuity_score += 0.2
        
        return continuity_score, metrics
    
    def _check_trajectory_continuity(self, points, indices):
        """检查轨迹连续性（保持向后兼容）"""
        if self.use_legacy_mode or self.point_to_traj is None:
            return False
        
        traj_groups = defaultdict(list)
        for i, idx in enumerate(indices):
            traj_id = self.point_to_traj[idx]
            traj_groups[traj_id].append((points[i], idx))
        
        for traj_id, point_list in traj_groups.items():
            point_list.sort(key=lambda x: x[0][2])  # 按时间排序
            
            for i in range(len(point_list) - 1):
                time_diff = point_list[i+1][0][2] - point_list[i][0][2]
                if time_diff < 600:  # 10分钟内认为连续
                    return True
        
        return False
    
    def _compute_semantic_weights(self):
        """计算语义权重"""
        if self.feature_extractor is None or self.use_legacy_mode:
            return
        
        # 按轨迹分组计算语义权重
        if self.point_to_traj is None:
            return
        
        traj_ids_arr = np.array(self.point_to_traj)
        unique_traj_ids = np.unique(traj_ids_arr)
        
        # 初始化权重数组
        all_weights = np.zeros(self.n_points)
        
        for traj_id in unique_traj_ids:
            mask = traj_ids_arr == traj_id
            point_indices = np.where(mask)[0]
            
            if len(point_indices) < 2:
                continue
            
            traj_points = self.points[point_indices]
            traj_data = [[p[0], p[1], p[2]] for p in traj_points]
            
            # 计算语义权重（返回结果字典）
            res = self.feature_extractor.compute_semantic_weights(traj_data)
            weights = res.get('point_weights', [])
            
            # 存储权重
            for i, idx in enumerate(point_indices):
                if i < len(weights):
                    all_weights[idx] = weights[i]
        
        self.point_semantic_weights = all_weights
    
    def _compute_node_features(self, points, indices, center, radius):
        """计算节点语义特征"""
        if self.use_legacy_mode:
            return {}
        
        # 优化集合构建：使用 numpy.unique
        if isinstance(self.point_to_traj, np.ndarray):
            unique_traj_ids = np.unique(self.point_to_traj[indices])
            traj_ids = set(unique_traj_ids)
        else:
            traj_ids = set([self.point_to_traj[idx] for idx in indices])
            
        avg_direction = self.point_directions[indices].mean(axis=0)
        avg_speed = self.point_speeds[indices].mean()
        time_span = (points[:, 2].min(), points[:, 2].max())
        density = len(points) / ((4/3) * np.pi * radius**3) if radius > 0 else 0
        direction_variance = self.point_directions[indices].var(axis=0).mean()
        
        return {
            'traj_ids': traj_ids,
            'avg_direction': avg_direction,
            'avg_speed': avg_speed,
            'time_span': time_span,
            'density': density,
            'direction_variance': direction_variance
        }
    
    def _build_tree(self, points, indices, depth=0, max_depth=50):
        """递归构建BallTree - 支持多阶段分割和增强功能"""
        # 1. 递归终止条件
        if len(points) <= 1 or depth >= max_depth:
            node_features = self._compute_node_features(points, indices, None, 0) if not self.use_legacy_mode else {}
            center = points[0] if len(points) == 1 else None
            radius = 0 if len(points) == 1 else 0
            return BallTreeNode(points, indices, center, radius, None, None, **node_features)
        
        # 1.5 密度检查（仅在浅层节点或特定条件下进行，大幅加速构建过程）
        compute_density = (depth < 3 and len(points) > 100) or (len(points) > 2000)
        if self.feature_extractor is not None and compute_density:
            try:
                # 采样后再计算密度
                check_size = min(100, len(points))
                sample_pts = points[np.random.choice(len(points), check_size, replace=False)]
                traj_data = [[p[0], p[1], p[2]] for p in sample_pts]
                density_features = self.feature_extractor.extract_density_features(traj_data)
                avg_density = density_features.get('avg_density', 0.0)
                if avg_density < 1e-7:
                    center = points.mean(axis=0)
                    radius = np.max(np.linalg.norm(points - center, axis=1)) if len(points) > 0 else 0
                    node_features = self._compute_node_features(points, indices, center, radius)
                    return BallTreeNode(points, indices, center, radius, None, None, **node_features)
            except Exception:
                pass 
        
        # 1.6 大节点近似采样
        if len(points) > self.large_node_threshold:
            traj_ids = self.point_to_traj if not self.use_legacy_mode else None
            sampled_points, sampled_indices = self._hierarchical_sampling(points, indices, traj_ids)
            # 使用采样点进行后续计算，但保留原始indices用于最终节点
            points_for_split = sampled_points
            indices_for_split = sampled_indices
        else:
            points_for_split = points
            indices_for_split = indices
        
        # 1.7 轨迹连续性检查
        if len(points) <= self.N_leaf:
            if self._check_trajectory_continuity(points, indices):
                if not self.use_legacy_mode and len(points) > 20:
                    center = self._compute_weighted_center(points, indices)
                else:
                    center = points.mean(axis=0)
                radius = self._compute_radius(points, center, indices)
                node_features = self._compute_node_features(points, indices, center, radius)
                return BallTreeNode(points, indices, center, radius, None, None, **node_features)
        
        # 2. 确定分割策略
        try:
            if not self.use_legacy_mode:
                strategy = self._determine_split_strategy_enhanced(points_for_split, indices_for_split, depth)
            else:
                strategy = "SPATIAL_SPLIT"
            
            # 3. 获取分割方向
            if strategy == "TIME_SPLIT":
                split_direction, median_time = self._time_window_split(points_for_split, indices_for_split)
                # 使用时间中位数进行分割
                times = points_for_split[:, 2]
                median_proj = median_time
                projections = times
            elif strategy == "SEMANTIC_SPLIT":
                if not self.use_legacy_mode and self.point_directions is not None:
                    avg_direction = self.point_directions[indices_for_split].mean(axis=0)
                    norm = np.linalg.norm(avg_direction)
                    if norm > 1e-10:
                        split_direction = (avg_direction / norm).astype(np.float64)
                    else:
                        split_direction = np.array([0, 0, 1], dtype=np.float64)
                else:
                    split_direction = np.array([0, 0, 1], dtype=np.float64)
                
                # 沿语义方向投影
                centered_points = points_for_split - points_for_split.mean(axis=0)
                projections = np.dot(centered_points, split_direction)
                median_proj = np.median(projections)
            else:  # SPATIAL_SPLIT
                split_direction = self._select_split_direction(points_for_split, indices_for_split, strategy, depth)
                centered_points = points_for_split - points_for_split.mean(axis=0)
                projections = np.dot(centered_points, split_direction)
                
                # 检查投影有效性
                if np.allclose(projections, projections[0]):
                    axis = depth % points_for_split.shape[1]
                    projections = points_for_split[:, axis] - points_for_split[:, axis].mean()
                
                median_proj = np.median(projections)
            
            # 4. 分割阈值随深度变化：split_frac(d) = 0.5*(1+exp(-λ d))
            split_frac = 0.5 * (1 + np.exp(-self.lambda_depth * depth))
            # 使用分位数而不是中位数
            quantile_idx = int(len(projections) * split_frac)
            sorted_proj = np.sort(projections)
            if quantile_idx < len(sorted_proj):
                median_proj = sorted_proj[quantile_idx]
            
            # 应用到所有点（不仅仅是采样点）
            if len(points) > len(points_for_split):
                # 需要将分割应用到所有点
                if strategy == "TIME_SPLIT":
                    all_times = points[:, 2]
                    all_projections = all_times
                else:
                    all_centered = points - points.mean(axis=0)
                    all_projections = np.dot(all_centered, split_direction)
            else:
                all_projections = projections
            
            left_mask = all_projections <= median_proj
            right_mask = all_projections > median_proj
            
            # 检查分割是否有效
            if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                if not self.use_legacy_mode and len(points) > 20:
                    center = self._compute_weighted_center(points, indices)
                else:
                    center = points.mean(axis=0)
                radius = self._compute_radius(points, center, indices) if len(points) > 0 else 0
                node_features = self._compute_node_features(points, indices, center, radius)
                return BallTreeNode(points, indices, center, radius, None, None, **node_features)
            
            # 优化5：使用布尔索引直接获取，避免列表推导式
            left_points = points[left_mask]
            left_indices = np.array(indices)[left_mask].tolist()
            right_points = points[right_mask]
            right_indices = np.array(indices)[right_mask].tolist()
            
            # 5. 计算球心（使用加权方法如果启用语义分割）
            if not self.use_legacy_mode and len(points) > 20:
                center = self._compute_weighted_center(points, indices)
            else:
                center = points.mean(axis=0)
            radius = np.max(np.linalg.norm(points - center, axis=1)) if len(points) > 0 else 0
            
            # 6. 递归构建子树
            left_child = self._build_tree(left_points, left_indices, depth + 1, max_depth) if len(left_points) > 0 else None
            right_child = self._build_tree(right_points, right_indices, depth + 1, max_depth) if len(right_points) > 0 else None
            
            # 优化6：只对叶子节点或浅层节点计算特征，减少计算开销
            # 7. 计算节点特征（只在深度较浅或叶子节点时计算）
            compute_features = (not self.use_legacy_mode and (depth < 5 or left_child is None or right_child is None))
            node_features = self._compute_node_features(points, indices, center, radius) if compute_features else {}
            
            # 8. 创建并返回节点
            return BallTreeNode(points, indices, center, radius, left_child, right_child, **node_features)
            
        except Exception as e:
            # 如果出现任何错误，创建叶子节点
            print(f"Warning: BallTree construction failed at depth {depth}: {e}")
            center = points.mean(axis=0)
            radius = np.max(np.linalg.norm(points - center, axis=1)) if len(points) > 0 else 0
            node_features = self._compute_node_features(points, indices, center, radius) if not self.use_legacy_mode else {}
            return BallTreeNode(points, indices, center, radius, None, None, **node_features)
    
    def _compute_weighted_center(self, points, indices):
        """计算轨迹感知的加权球心（全向量化版）"""
        if self.use_legacy_mode or len(points) <= 20:
            return points.mean(axis=0)
        
        # 优先使用预计算的语义权重
        if self.point_semantic_weights is not None:
            # 避免循环，直接索引
            w = self.point_semantic_weights[indices]
            w_sum = w.sum()
            if w_sum > 1e-10:
                return np.dot(w / w_sum, points)
        
        # 回退逻辑：采样+向量化空间中心性
        n = len(points)
        if n > 200:
            sample_idx = np.linspace(0, n-1, 50, dtype=int)
            sample_pts = points[sample_idx]
        else:
            sample_pts = points
            
        # 向量化时间中心性
        t = points[:, 2]
        t_mid = (t.min() + t.max()) / 2
        t_span = t.max() - t.min()
        w_t = 1.0 - np.abs(t - t_mid) / (t_span + 1e-6)
        
        # 向量化空间中心性（近似：到边界盒中心的距离）
        bbox_min = points.min(axis=0)
        bbox_max = points.max(axis=0)
        box_center = (bbox_min + bbox_max) / 2
        w_s = 1.0 / (1.0 + np.linalg.norm(points - box_center, axis=1))
        
        w = w_t * w_s
        w_sum = w.sum()
        return np.dot(w / w_sum, points) if w_sum > 1e-10 else points.mean(axis=0)
    
    def _compute_radius(self, points, center, indices):
        """
        计算加权半径 r_j = k * sqrt(Var_j(weighted))
        
        Args:
            points: 点数组
            center: 球心
            indices: 点索引列表
        
        Returns:
            radius: 半径
        """
        if len(points) == 0:
            return 0.0
        
        # 获取权重（优先使用语义权重）
        if self.point_semantic_weights is not None and len(self.point_semantic_weights) > 0:
            w = np.array([self.point_semantic_weights[idx] if idx < len(self.point_semantic_weights) else 1.0 
                         for idx in indices])
            if w.sum() > 1e-10:
                w = w / w.sum()  # 归一化
            else:
                w = np.ones(len(points)) / len(points)
        else:
            w = np.ones(len(points)) / len(points)
        
        # 计算加权方差
        centered_points = points - center
        weighted_var = np.zeros(3)
        for j in range(3):
            var_j = np.sum(w * (centered_points[:, j] ** 2))
            weighted_var[j] = var_j
        
        # 计算半径：r_j = k * sqrt(Var_j(weighted))
        # 使用最大方差维度
        max_var = np.max(weighted_var)
        radius = self.k_radius * np.sqrt(max_var)
        
        # 确保半径至少覆盖所有点
        max_dist = np.max(np.linalg.norm(centered_points, axis=1))
        radius = max(radius, max_dist)
        
        return radius
    
    def query(self, point, k=1):
        """查询最近的k个点 - 优化版本，O(log n + k)时间复杂度"""
        if self.n_points == 0:
            return [], []
        
        # 使用优先队列存储候选点，最大堆存储最近的k个点
        max_heap = []  # 存储(-distance, index)，用于维护最近的k个点
        min_heap = []  # 存储(distance, node_id, node)，用于优先搜索
        node_counter = 0  # 用于生成唯一节点ID
        
        # 初始化搜索队列
        heapq.heappush(min_heap, (0, node_counter, self.root))
        node_counter += 1
        
        # 搜索直到找到k个最近点且没有更近的候选节点
        while min_heap and (len(max_heap) < k or min_heap[0][0] < -max_heap[0][0]):
            current_dist, _, current_node = heapq.heappop(min_heap)
            
            # 如果当前节点距离大于第k近点的距离，剪枝
            if len(max_heap) >= k and current_dist > -max_heap[0][0]:
                break
            
            # 处理当前节点的所有点
            for i, node_point in enumerate(current_node.points):
                dist = np.linalg.norm(point - node_point)
                
                if len(max_heap) < k:
                    # 堆未满，直接添加
                    heapq.heappush(max_heap, (-dist, current_node.indices[i]))
                elif dist < -max_heap[0][0]:
                    # 距离更近，替换最远的点
                    heapq.heapreplace(max_heap, (-dist, current_node.indices[i]))
            
            # 添加子节点到搜索队列
            if current_node.left_child:
                left_dist = self._min_distance_to_ball(point, current_node.left_child)
                heapq.heappush(min_heap, (left_dist, node_counter, current_node.left_child))
                node_counter += 1
            
            if current_node.right_child:
                right_dist = self._min_distance_to_ball(point, current_node.right_child)
                heapq.heappush(min_heap, (right_dist, node_counter, current_node.right_child))
                node_counter += 1
        
        # 提取结果
        distances = []
        indices = []
        for neg_dist, idx in max_heap:
            distances.append(-neg_dist)
            indices.append(idx)
        
        # 按距离排序
        sorted_pairs = sorted(zip(distances, indices))
        distances = [pair[0] for pair in sorted_pairs]
        indices = [pair[1] for pair in sorted_pairs]
        
        return distances, indices
    
    def _min_distance_to_ball(self, point, node):
        """计算点到球的最小距离，用于剪枝优化"""
        if node.center is None:
            return float('inf')
        
        dist_to_center = np.linalg.norm(point - node.center)
        return max(0, dist_to_center - node.radius)
    
    def _max_distance_to_ball(self, point, node):
        """计算点到球的最大距离，用于剪枝优化"""
        if node.center is None:
            return float('inf')
        
        dist_to_center = np.linalg.norm(point - node.center)
        return dist_to_center + node.radius
    
    
    def _search_knn(self, node, point, k, heap):
        """递归搜索KNN"""
        if node is None:
            return
        
        # 计算当前节点所有点的距离
        for i, node_point in enumerate(node.points):
            dist = np.linalg.norm(point - node_point)
            heapq.heappush(heap, (dist, node.indices[i]))
        
        # 如果当前节点是叶子节点，返回
        if node.left_child is None and node.right_child is None:
            return
        
        # 计算到球心的距离
        dist_to_center = np.linalg.norm(point - node.center)
        
        # 决定搜索顺序
        if dist_to_center <= node.radius:
            # 点在球内，搜索两个子树
            if node.left_child:
                self._search_knn(node.left_child, point, k, heap)
            if node.right_child:
                self._search_knn(node.right_child, point, k, heap)
        else:
            # 点在球外，优先搜索较近的子树
            if node.left_child and node.right_child:
                left_dist = np.linalg.norm(point - node.left_child.center)
                right_dist = np.linalg.norm(point - node.right_child.center)
                
                if left_dist < right_dist:
                    self._search_knn(node.left_child, point, k, heap)
                    self._search_knn(node.right_child, point, k, heap)
                else:
                    self._search_knn(node.right_child, point, k, heap)
                    self._search_knn(node.left_child, point, k, heap)
            elif node.left_child:
                self._search_knn(node.left_child, point, k, heap)
            elif node.right_child:
                self._search_knn(node.right_child, point, k, heap)
    
    def range_query(self, bounds, objects=True):
        """范围查询，返回在指定边界内的所有点"""
        if self.n_points == 0:
            return []
        
        mins = np.array(bounds[:3])
        maxs = np.array(bounds[3:])
        
        results = []
        self._search_range(self.root, mins, maxs, results)
        
        if objects:
            return [(idx, self.points[idx], idx) for idx in results]
        else:
            return [(idx, self.points[idx]) for idx in results]
    
    def _search_range(self, node, mins, maxs, results):
        """递归搜索范围内的点"""
        if node is None:
            return
        
        # 检查当前节点是否与查询范围相交
        if not self._ball_intersects_box(node, mins, maxs):
            return
        
        # 检查当前节点的所有点
        for i, point in enumerate(node.points):
            if np.all(point >= mins) and np.all(point <= maxs):
                results.append(node.indices[i])
        
        # 递归搜索子节点
        if node.left_child:
            self._search_range(node.left_child, mins, maxs, results)
        if node.right_child:
            self._search_range(node.right_child, mins, maxs, results)
    
    def _ball_intersects_box(self, node, mins, maxs):
        """检查球是否与矩形框相交"""
        if node.center is None:
            return False
        
        # 计算球心到矩形框的最短距离
        closest_point = np.maximum(mins, np.minimum(node.center, maxs))
        dist_to_box = np.linalg.norm(node.center - closest_point)
        
        # 如果距离小于等于半径，则相交
        return dist_to_box <= node.radius
    
    def _search_radius(self, node, point, r, result_indices):
        """递归搜索半径内的点"""
        if node is None:
            return
        
        # 计算到球心的距离
        dist_to_center = np.linalg.norm(point - node.center)
        
        # 如果查询点与球心的距离 > 半径 + 查询半径，则球内没有满足条件的点
        if dist_to_center > node.radius + r:
            return
        
        # 检查当前节点的所有点
        for i, node_point in enumerate(node.points):
            dist = np.linalg.norm(point - node_point)
            if dist <= r:
                result_indices.append(node.indices[i])
        
        # 递归搜索子树
        if node.left_child:
            self._search_radius(node.left_child, point, r, result_indices)
        if node.right_child:
            self._search_radius(node.right_child, point, r, result_indices)

class BallTreeNode:
    """BallTree节点"""
    def __init__(self, points, indices, center, radius, left_child, right_child,
                 traj_ids=None, avg_direction=None, avg_speed=None, time_span=None,
                 density=None, direction_variance=None):
        self.points = points
        self.indices = indices
        self.center = center
        self.radius = radius
        self.left_child = left_child
        self.right_child = right_child
        # 新增语义特征字段
        self.traj_ids = traj_ids
        self.avg_direction = avg_direction
        self.avg_speed = avg_speed
        self.time_span = time_span
        self.density = density
        self.direction_variance = direction_variance

import sys
sys.path.append("autodl-tmp/GFlowNet-one")

from t2vec_utils import args, model_init, submit

m0 = None
def get_m0():
    global m0
    if m0 is None:
        args.checkpoint = "F:/query/GFlowNet-one/TrajData/oldbest_model.pt"
        args.vocab_size = 40004 #40004 (geolife)
        m0 = model_init(args)
    return m0

import numpy as np
import math
from rtree import index
import random
import shutil
from partition import approximate_trajectory_partitioning
from point import Point
from cluster import line_segment_clustering

print('done import')

def lonlat2meters(lon, lat):
    semimajoraxis = 6378137.0
    east = lon * 0.017453292519943295
    north = lat * 0.017453292519943295
    t = math.sin(north)
    return semimajoraxis * east, 3189068.5 * math.log((1 + t) / (1 - t))

def points2meter(points):
    rtn = []
    for p in points:
        lon_meter, lat_meter = lonlat2meters(lon=p[1], lat=p[0])
        rtn.append([lat_meter,lon_meter,p[2]])
    return rtn

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

def Eu(segment):
    ps = segment[0]
    pe = segment[-1]    
    syn_time = segment[1][2]
    time_ratio = 1 if (pe[2]- ps[2]) == 0  else (syn_time-ps[2]) / (pe[2]-ps[2])
    syn_x = ps[0] + (pe[0] - ps[0]) * time_ratio
    syn_y = ps[1] + (pe[1] - ps[1]) * time_ratio
    e = np.linalg.norm(np.array([segment[1][0],segment[1][1]]) - np.array([syn_x,syn_y]))
    return e

def Et(segment):
    ps = segment[0]
    pm = segment[1]
    pe = segment[-1]
    A = pe[1] - ps[1]
    B = ps[0] - pe[0]
    C = pe[0] * ps[1] - ps[0] * pe[1]
    if A == 0 and B == 0:
        return 0.0
    else:
        x = (B*B*pm[0] - A*B*pm[1] - A*C)/(A*A + B*B)
        y = (-A*B*pm[0] + A*A*pm[1] - B*C)/(A*A + B*B)
        speed = np.linalg.norm(np.array([ps[0], ps[1]]) - np.array([pe[0], pe[1]]))/(pe[2]-ps[2])
        return abs(ps[2] + np.linalg.norm(np.array([ps[0], ps[1]]) - np.array([x,y]))/speed - pm[2])

def sed_op(segment):
    if len(segment) <= 2:
        return 0.0
    else:
        ps = segment[0]
        pe = segment[-1]
        e = 0.0
        for i in range(1,len(segment)-1):
            syn_time = segment[i][2]
            time_ratio = 1 if (pe[2]- ps[2]) == 0  else (syn_time-ps[2]) / (pe[2]-ps[2])
            syn_x = ps[0] + (pe[0] - ps[0]) * time_ratio
            syn_y = ps[1] + (pe[1] - ps[1]) * time_ratio
            e = max(e, np.linalg.norm(np.array([segment[i][0],segment[i][1]]) - np.array([syn_x,syn_y])))
        return e
    
def sed_error(ori_traj, sim_traj):
    #ori_traj, sim_traj = [[x,y,t],...,[x,y,t]]
    # 1-keep and 0-drop
    dict_traj = {}
    t_map = [0 for i in range(len(ori_traj))]
    for c, value in enumerate(ori_traj):
        dict_traj[tuple(value)] = c
    for value in sim_traj:
        t_map[dict_traj[tuple(value)]] = 1
    error = 0.0
    start = 0
    for c, value in enumerate(t_map):
        if value == 1:
            error = max(error, sed_op(ori_traj[start: c+1]))
            start = c
    return t_map, error

def ped_op(segment):
    if len(segment) <= 2:
        return 0.0
    else:
        ps = segment[0]
        pe = segment[-1]
        e = 0.0
        for i in range(1,len(segment)-1):
            pm = segment[i]
            A = pe[1] - ps[1]
            B = ps[0] - pe[0]
            C = pe[0] * ps[1] - ps[0] * pe[1]
            if A == 0 and B == 0:
                e = max(e, 0.0)
            else:
                e = max(e, abs((A * pm[0] + B * pm[1] + C)/ np.sqrt(A * A + B * B)))
        return e

def ped_error(ori_traj, sim_traj):
    #ori_traj, sim_traj = [[x,y,t],...,[x,y,t]]
    # 1-keep and 0-drop
    dict_traj = {}
    t_map = [0 for i in range(len(ori_traj))]
    for c, value in enumerate(ori_traj):
        dict_traj[tuple(value)] = c
    for value in sim_traj:
        t_map[dict_traj[tuple(value)]] = 1
    error = 0.0
    start = 0
    for c, value in enumerate(t_map):
        if value == 1:
            error = max(error, ped_op(ori_traj[start: c+1]))
            start = c
    return t_map, error

def angle(v1):
    dx1 = v1[2] - v1[0]
    dy1 = v1[3] - v1[1]
    angle1 = math.atan2(dy1, dx1)
    if angle1 >= 0:
        return angle1
    else:
        return 2*math.pi + angle1
    
def dad_op(segment):
    if len(segment) <= 2:
        return 0.0
    else:
        ps = segment[0]
        pe = segment[-1]
        e = 0.0
        theta_0 = angle([ps[0],ps[1],pe[0],pe[1]])
        for i in range(0,len(segment)-1):
            pm_0 = segment[i]
            pm_1 = segment[i+1]
            theta_1 = angle([pm_0[0],pm_0[1],pm_1[0],pm_1[1]])
            e = max(e, min(abs(theta_0 - theta_1), 2*math.pi - abs(theta_0 - theta_1)))
        return e

def dad_error(ori_traj, sim_traj):
    #ori_traj, sim_traj = [[x,y,t],...,[x,y,t]]
    # 1-keep and 0-drop
    dict_traj = {}
    t_map = [0 for i in range(len(ori_traj))]
    for c, value in enumerate(ori_traj):
        dict_traj[tuple(value)] = c
    for value in sim_traj:
        t_map[dict_traj[tuple(value)]] = 1
    error = 0.0
    start = 0
    for c, value in enumerate(t_map):
        if value == 1:
            error = max(error, dad_op(ori_traj[start: c+1]))
            start = c
    return t_map, error

def get_point(ps, pe, segment, index):
    syn_time = segment[index][2]
    time_ratio = 1 if (pe[2]- ps[2]) == 0  else (syn_time-ps[2]) / (pe[2]-ps[2])
    syn_x = ps[0] + (pe[0] - ps[0]) * time_ratio
    syn_y = ps[1] + (pe[1] - ps[1]) * time_ratio
    return [syn_x, syn_y], syn_time

def speed_op(segment):
    if len(segment) <= 2:
        return 0.0
    else:
        ps = segment[0]
        pe = segment[-1]
        e = 0.0
        for i in range(0,len(segment)-1):
            p_1, t_1 = get_point(ps, pe, segment, i)
            p_2, t_2 = get_point(ps, pe, segment, i+1)
            time = 1 if t_2 - t_1 == 0 else abs(t_2-t_1)
            est_speed = np.linalg.norm(np.array(p_1) - np.array(p_2))/time
            rea_speed = np.linalg.norm(np.array([segment[i][0], segment[i][1]]) - np.array([segment[i+1][0], segment[i+1][1]]))/time
            e = max(e, abs(est_speed - rea_speed))
        return e

def speed_error(ori_traj, sim_traj):
    #ori_traj, sim_traj = [[x,y,t],...,[x,y,t]]
    # 1-keep and 0-drop
    dict_traj = {}
    t_map = [0 for i in range(len(ori_traj))]
    for c, value in enumerate(ori_traj):
        dict_traj[tuple(value)] = c
    for value in sim_traj:
        t_map[dict_traj[tuple(value)]] = 1
    error = 0.0
    start = 0
    for c, value in enumerate(t_map):
        if value == 1:
            error = max(error, speed_op(ori_traj[start: c+1]))
            start = c
    return t_map, error

class BallTree():
    """球树实现，基于主成分分析的空间分割"""
    def __init__(self, points=None, ids=None, objects=None, *filename):
        self.p = index.Property()
        self.p.dimension = 3
        self.points = []
        self.ids = []
        self.objects = []
        if len(filename) == 0:
            self.idx = index.Index(properties=self.p)
        else:
            self.idx = index.Index(filename[0], properties=self.p)
            self.points = np.array([(p[0], p[1], p[2]) for p in self.idx.data], dtype=np.float64)
            self.ids = self.idx.id
            self.objects=self.idx.obj
            #self.traj_ids = self.idx.traj_ids if hasattr(self.idx, 'traj_ids') else []
            print("filenametree",filename)
        self.points.append(points) if points is not None else []
        #print("initpoints",points)
        self.ids.append(list(ids)) if ids is not None else []
        self.objects.append(list(objects)) if objects is not None else []

        self.id_to_index = {} 
        self.trajID_to_indices = defaultdict(list)
        self._tree = None
        self._needs_rebuild = True
        self.balltree = []
        self.data = []
        
    def _split_space_ball_tree(self, points):
        """球树分割：基于主成分分析将点集分成2个子集"""
        if len(points) < 2:
            return points, [], None, None
        
        points = np.array(points)
        
        # 计算主成分方向
        centered_points = points - points.mean(axis=0)
        cov_matrix = np.cov(centered_points.T)
        eigenvals, eigenvecs = np.linalg.eig(cov_matrix)
        principal_direction = eigenvecs[:, np.argmax(eigenvals)]
        
        # 沿主方向投影并找到中点
        projections = np.dot(centered_points, principal_direction)
        median_proj = np.median(projections)
        
        # 分割点集
        left_mask = projections <= median_proj
        right_mask = projections > median_proj
        
        left_points = points[left_mask]
        right_points = points[right_mask]
        
        return left_points, right_points, principal_direction, median_proj
    

    
    def _min_distance_to_ball(self, point, node):
        """计算点到球的最小距离，用于剪枝优化"""
        if node.center is None:
            return float('inf')
        
        dist_to_center = np.linalg.norm(point - node.center)
        return max(0, dist_to_center - node.radius)
    
    def _max_distance_to_ball(self, point, node):
        """计算点到球的最大距离，用于剪枝优化"""
        if node.center is None:
            return float('inf')
        
        dist_to_center = np.linalg.norm(point - node.center)
        return dist_to_center + node.radius
    
    def query(self, point, k=1):
        """优化的KNN查询，O(log n + k)时间复杂度"""
        if self.n_points == 0:
            return [], []
        
        # 使用优先队列存储候选点，最大堆存储最近的k个点
        max_heap = []  # 存储(-distance, index)，用于维护最近的k个点
        min_heap = []  # 存储(distance, node_id, node)，用于优先搜索
        node_counter = 0  # 用于生成唯一节点ID
        
        # 初始化搜索队列
        heapq.heappush(min_heap, (0, node_counter, self.root))
        node_counter += 1
        
        # 搜索直到找到k个最近点且没有更近的候选节点
        while min_heap and (len(max_heap) < k or min_heap[0][0] < -max_heap[0][0]):
            current_dist, _, current_node = heapq.heappop(min_heap)
            
            # 如果当前节点距离大于第k近点的距离，剪枝
            if len(max_heap) >= k and current_dist > -max_heap[0][0]:
                break
            
            # 处理当前节点的所有点
            for i, node_point in enumerate(current_node.points):
                dist = np.linalg.norm(point - node_point)
                
                if len(max_heap) < k:
                    # 堆未满，直接添加
                    heapq.heappush(max_heap, (-dist, current_node.indices[i]))
                elif dist < -max_heap[0][0]:
                    # 距离更近，替换最远的点
                    heapq.heapreplace(max_heap, (-dist, current_node.indices[i]))
            
            # 添加子节点到搜索队列
            if current_node.left_child:
                left_dist = self._min_distance_to_ball(point, current_node.left_child)
                heapq.heappush(min_heap, (left_dist, node_counter, current_node.left_child))
                node_counter += 1
            
            if current_node.right_child:
                right_dist = self._min_distance_to_ball(point, current_node.right_child)
                heapq.heappush(min_heap, (right_dist, node_counter, current_node.right_child))
                node_counter += 1
        
        # 提取结果
        distances = []
        indices = []
        for neg_dist, idx in max_heap:
            distances.append(-neg_dist)
            indices.append(idx)
        
        # 按距离排序
        sorted_pairs = sorted(zip(distances, indices))
        distances = [pair[0] for pair in sorted_pairs]
        indices = [pair[1] for pair in sorted_pairs]
        
        return distances, indices
    
    def _search_knn(self, node, point, k, heap):
        """递归搜索KNN"""
        if node is None:
            return
        
        # 计算当前节点所有点的距离
        for i, node_point in enumerate(node.points):
            dist = np.linalg.norm(point - node_point)
            heapq.heappush(heap, (dist, node.indices[i]))
        
        # 如果当前节点是叶子节点，返回
        if node.left_child is None and node.right_child is None:
            return
        
        # 计算到球心的距离
        dist_to_center = np.linalg.norm(point - node.center)
        
        # 决定搜索顺序
        if dist_to_center <= node.radius:
            # 点在球内，搜索两个子树
            if node.left_child:
                self._search_knn(node.left_child, point, k, heap)
            if node.right_child:
                self._search_knn(node.right_child, point, k, heap)
        else:
            # 点在球外，优先搜索较近的子树
            if node.left_child and node.right_child:
                left_dist = np.linalg.norm(point - node.left_child.center)
                right_dist = np.linalg.norm(point - node.right_child.center)
                
                if left_dist < right_dist:
                    self._search_knn(node.left_child, point, k, heap)
                    self._search_knn(node.right_child, point, k, heap)
                else:
                    self._search_knn(node.right_child, point, k, heap)
                    self._search_knn(node.left_child, point, k, heap)
            elif node.left_child:
                self._search_knn(node.left_child, point, k, heap)
            elif node.right_child:
                self._search_knn(node.right_child, point, k, heap)
  




    
    def range_query(self, bounds, objects=True):
        """范围查询，使用CustomBallTree的高效实现"""
        if self.balltree is None or len(self.balltree.points) == 0:
            return []
        
        return self.balltree.range_query(bounds, objects=objects)
    
    def range_query_with_traj_ids(self, bounds):
        """范围查询并返回轨迹ID集合，优先使用rtree进行高效查询"""
        # 优先使用rtree直接查询（更高效）
        if hasattr(self, 'idx') and self.idx is not None:
            try:
                # 直接从rtree索引进行范围查询
                # bounds格式: (min_x, min_y, min_t, max_x, max_y, max_t)
                matches = list(self.idx.intersection(bounds, objects=True))
                # rtree的intersection在使用objects=True时，返回rtree.index.Item对象
                # 需要从Item对象的object属性中提取轨迹ID
                traj_ids = set()
                for match in matches:
                    if match is not None:
                        # 检查是否是Item对象
                        if hasattr(match, 'object'):
                            traj_id = match.object
                            if traj_id is not None:
                                traj_ids.add(traj_id)
                        else:
                            # 如果不是Item对象，直接使用match作为trajID
                            traj_ids.add(match)
                return traj_ids
            except Exception as e:
                # 如果rtree查询失败，回退到BallTree方法
                pass
        
        # 回退方法：使用BallTree的range_query
        if self.balltree is None or len(self.balltree.points) == 0:
            return set()
        
        query_results = self.balltree.range_query(bounds, objects=True)
        traj_ids = set()
        for result in query_results:
            if len(result) >= 2:
                idx = result[0]
                if idx < len(self.data):
                    query_point = self.data[idx]
                    # 使用小范围bbox来查找轨迹ID
                    if hasattr(self, 'idx'):
                        try:
                            bbox = (query_point[0]-0.0001, query_point[1]-0.0001, query_point[2]-0.1,
                                   query_point[0]+0.0001, query_point[1]+0.0001, query_point[2]+0.1)
                            bbox_matches = list(self.idx.intersection(bbox, objects=True))
                            if bbox_matches:
                                for bbox_match in bbox_matches:
                                    if hasattr(bbox_match, 'object'):
                                        traj_id = bbox_match.object
                                        if traj_id is not None:
                                            traj_ids.add(traj_id)
                                    elif bbox_match is not None:
                                        traj_ids.add(bbox_match)
                        except Exception:
                            pass
        
        return traj_ids
        
    def build(self, points, traj_ids=None):
        if isinstance(points, list):
            points = tuple(points)
        if traj_ids is not None:
            self.traj_ids = traj_ids
        self.points = [] if points is None else points
        #print("buildpoints",points[:3])
        # 传递traj_ids到CustomBallTree以启用语义特征
        self.balltree = CustomBallTree(self.points, traj_ids=traj_ids, use_semantic_split=True)
        self.data = self.points
        return self.balltree
    
    def insert(self, id, point_data, obj, defer_build=False):
        if not isinstance(self.points, list):
            self.points = list(self.points) if isinstance(self.points, (tuple, np.ndarray)) else []
        if isinstance(point_data, dict):
            x = float(point_data['x'])
            y = float(point_data['y'])
            z = float(point_data['z'])
            point = (x, y, z)
        else:
            x, y, t = point_data[:3]
            point = (float(x), float(y), float(t))
    
        ids = id
        if isinstance(ids, list):
            ids = tuple(ids)
    
        # Update data structures
        index = len(self.points)
        self.points.append(point)
        #print("insirtpoints",self.points)
        if not defer_build:
            self.build(self.points)
        self.idx.insert(id, point_data, obj=obj)
        self.ids.append(id)
        self.objects.append(obj)
        self.id_to_index[ids] = index
        
        if obj is not None:
            self.trajID_to_indices[obj].append(index)
        
    
    
    
    
    
    
        
    def delete(self, id, data): #id = int, data = (lat, lon), obj_trajID
        self.idx.delete(id, data)

    def knn(self, width, num=1, objects=True):# width = (xmin, ymin, tmin, xmax, ymax, tmax)
        res=list(self.idx.nearest(width, num, objects=objects))
        return res
    
    def range_query(self, width, objects=True): #  width = (xmin, ymin, tmin, xmax, ymax, tmax)
        res=list(self.idx.intersection(width, objects=objects))
        return res
    
    def _load_from_file(self, filename):
        try:
            data = np.load(filename, allow_pickle=True)
            self.points = data['points'].tolist()
            self.ids = data['ids'].tolist()
            self.objects = data['objects'].tolist()
            self.id_to_index = data['id_to_index'].item()
            self.trajID_to_indices = defaultdict(list, data['trajID_to_indices'].item())
            self._build_tree()
        except Exception as e:
            raise ValueError(f"从文件 {filename} 加载BallTree失败: {str(e)}")

    
    def delete(self, id, point_data):
        if id not in self.id_to_index:
            raise ValueError(f"ID {id} not found in BallTree")
            
        index = self.id_to_index[id]
        
        
        stored_point = self.points[index]
        if stored_point != tuple(point_data[:3]):
            raise ValueError("Data does not match stored data for this ID")
        
        obj = self.objects[index]
        
        # Remove the point
        del self.points[index]
        del self.ids[index]
        del self.objects[index]
        
        # Update ID mapping and trajectory mapping
        del self.id_to_index[id]
        if obj in self.trajID_to_indices:
            self.trajID_to_indices[obj].remove(index)
            if not self.trajID_to_indices[obj]:
                del self.trajID_to_indices[obj]
        
        # Rebuild index mappings (since indices changed)
        self.id_to_index = {id: i for i, id in enumerate(self.ids)}
        
        # Rebuild trajectory mapping
        new_traj_map = defaultdict(list)
        for i, obj in enumerate(self.objects):
            if obj is not None:
                new_traj_map[obj].append(i)
        self.trajID_to_indices = new_traj_map
        
        self._needs_rebuild = True
    
    def nearest(self, query_point, num=1, objects=True):


        distances, indices = self._tree.query(query_point, k=num)
        
        if np.isscalar(indices):
            indices = [indices]
            distances = [distances]
        
        results = []
        for idx, dist in zip(indices, distances):
            point_id = self.ids[idx]
            if objects:
                results.append((point_id, self.points[idx], self.objects[idx], dist))
            else:
                results.append((point_id, self.points[idx], dist))
        
        return results
    
    def range_query(self, bounds, objects=True):
        if self._tree is None:
            return []
            
        mins = np.array(bounds[:3])
        maxs = np.array(bounds[3:])
        
        points_array = np.array(self.points)
        in_range = np.all((points_array >= mins) & (points_array <= maxs), axis=1)
        indices = np.where(in_range)[0]
        
        results = []
        for idx in indices:
            point_id = self.ids[idx]
            if objects:
                results.append((point_id, self.points[idx], self.objects[idx]))
            else:
                results.append((point_id, self.points[idx]))
                
        return results
        

    def save(self, filename):
        data = {
            'points': self.points,
            'ids': self.ids,
            'objects': self.objects,
            'id_to_index': self.id_to_index,
            'trajID_to_indices': dict(self.trajID_to_indices)
        }
        np.savez(filename, **data)

def save_balltree(bt1, bt2):
    bt1.save('temp_balltree.npz')
    if os.path.exists('temp_balltree.npz'):
        shutil.copyfile('temp_balltree.npz', 'target_balltree.npz')
        os.remove('temp_balltree.npz')

def build_BallTree(DB, filename=''):
    print(filename)
    print("**********************")
    p = index.Property()
    p.dimension = 3  # 设置维度为3 (x,y,t)
    
    if filename=='':
        balltree = BallTree()
    else: 
        if os.path.exists(filename+'.dat'):
            os.remove(filename+'.dat')
            print('remove', filename+'.dat')
        if os.path.exists(filename+'.idx'):
            os.remove(filename+'.idx')
            print('remove', filename+'.idx')
        balltree = BallTree(filename)
    
    # 优化1：预计算总点数，一次性分配内存
    delete_rec = {}
    total_points = sum(len(traj) for traj in DB)
    
    # 预分配数组（避免动态扩展）
    points_list = [None] * total_points
    traj_ids_list = [None] * total_points
    
    # 辅助数据结构预填充
    balltree.points = [] # 将被build覆盖，这里先占位或由points_list转换
    balltree.ids = list(range(total_points))
    balltree.objects = [None] * total_points
    # id_to_index 即 identity mapping: 0->0, 1->1...
    balltree.id_to_index = {i: i for i in range(total_points)}
    balltree.trajID_to_indices = defaultdict(list)
    
    # 生成器函数用于R-tree批量加载
    def rtree_generator():
        c = 0
        for trajID in range(len(DB)):
            traj = DB[trajID]
            for point in traj:
                # 填充 points_list
                points_list[c] = [point[0], point[1], point[2]]
                traj_ids_list[c] = trajID
                
                # 填充辅助结构
                balltree.objects[c] = trajID
                balltree.trajID_to_indices[trajID].append(c)
                delete_rec[(trajID, c - balltree.trajID_to_indices[trajID][0])] = c # pointID reconstruction
                
                # Yield for R-tree
                yield (c, (point[0], point[1], point[2], point[0], point[1], point[2]), trajID)
                c += 1
                
    # 创建 R-tree (Bulk Loading)
    print("Bulk loading R-tree...")
    if filename:
        balltree.idx = index.Index(filename, rtree_generator(), properties=p)
    else:
        balltree.idx = index.Index(rtree_generator(), properties=p)
    
    # 转换为numpy数组（一次性转换，避免多次分配）
    points = np.array(points_list, dtype=np.float64)
    traj_ids = traj_ids_list  
    
    # 手动设置 balltree.points (因为我们跳过了 insert)
    # 实际上 balltree.build 会覆盖 self.points, 但为了一致性:
    balltree.points = points_list # 保持 list 格式匹配 BallTree 预期
    
    print("Building CustomBallTree...")
    balltree.build(points, traj_ids=traj_ids)  # 传递轨迹ID
    print(filename)
    
    
    '''
    query_point=(39.97543, 116.3328516, 1252904190.0)
    distances, indices = balltree.balltree.query(query_point, k=7)
        
    # 返回结果
    qqqllll=(balltree.points[i] for i in indices)
    print("balltree.points[i] for i in indices",qqqllll)
    '''
    
    return balltree,delete_rec
    
def build_BallTree_Each(DB, file='', name=''):
    files_clean(file, name)
    print('finished clean')
    
    for trajID in range(len(DB)):
        balltree = BallTree()
        c = 0
        for pointID in range(len(DB[trajID])):
            point = DB[trajID][pointID]
            balltree.insert(c, (point[0], point[1], point[2], point[0], point[1], point[2]), trajID)
            c += 1
        balltree.save(f"{file}{name}{trajID}.npz")

def obtain_BallTree(filename):
    #if not os.path.exists(filename + '.dat') or not os.path.exists(filename + '.idx'):
    #    raise FileNotFoundError(f"BallTree data files not found at {filename}")
    Balltree_ = BallTree(filename)
    return Balltree_
    
def random_index(rate):
    start = 0
    index = 0
    randnum = random.randint(1, sum(rate))
    for index, scope in enumerate(rate):
        start += scope
        if randnum <= start:
            break
    return index

def files_clean(file, name):
    files = os.listdir(file)
    for f in files:
        if f.find(name) >= 0:
            os.remove(os.path.join(file, f))
            
def get_distribution_feature_data(db):
    DB_DISTRI, ID2Grid, DB_DISTRI_trajID = {}, {}, {}
    Xmin, Ymin, Tmin = 1.044024, -179.9695933, 1176341492
    x_step, y_step, t_step, thre = 0.02, 0.02, 3600*24*7, 1
    for trajID in range(len(db)):
        for pointID in range(len(db[trajID])):
            if pointID == 0 or pointID == len(db[trajID]) - 1:
                continue
            point = db[trajID][pointID]
            [x, y, t] = point
            key = tuple([int((x - Xmin)/x_step), int((y - Ymin)/y_step), int((t - Tmin)/t_step)])
            ID2Grid[(trajID, pointID)] = key
            if key in DB_DISTRI_trajID:
                DB_DISTRI_trajID[key].add(trajID)
            else:
                DB_DISTRI_trajID[key] = set([trajID])
    for key in DB_DISTRI_trajID:
        if len(DB_DISTRI_trajID[key]) > thre:
            DB_DISTRI[key] = len(DB_DISTRI_trajID[key])
    return DB_DISTRI, ID2Grid, DB_DISTRI_trajID

def get_distribution_feature_gau(db):
    DB_DISTRI, ID2Grid, Grid2ID, DB_DISTRI_trajID = {}, {}, {}, {}
    Xmin, Ymin, Tmin = 1.044024, -179.9695933, 1176341492
    x_step, y_step, t_step = 0.02, 0.02, 3600*24*7
    X, Y, T = [], [], []
    for trajID in range(len(db)):
        for pointID in range(len(db[trajID])):
            if pointID == 0 or pointID == len(db[trajID]) - 1:
                continue
            point = db[trajID][pointID]
            [x, y, t] = point
            key = tuple([int((x - Xmin)/x_step), int((y - Ymin)/y_step), int((t - Tmin)/t_step)])
            ID2Grid[(trajID, pointID)] = key
            if key in Grid2ID:
                Grid2ID[key].add(trajID)
            else:
                Grid2ID[key] = set([trajID])
                X.append(key[0])
                Y.append(key[1])
                T.append(key[2])
    X.sort()
    Y.sort()
    T.sort()
    X_map, Y_map, T_map = {}, {}, {}
    for i in range(len(Grid2ID)):
        X_map[i] = X[i]
        Y_map[i] = Y[i]
        T_map[i] = T[i]
    mu, alpha = (1+len(Grid2ID))/2, (len(Grid2ID)-1)/4
    count = 0
    while True:
        if count == 10000:
            break
        [x, y, t] = [np.random.normal(loc=mu, scale=alpha, size=None),
                     np.random.normal(loc=mu, scale=alpha, size=None),
                     np.random.normal(loc=mu, scale=alpha, size=None)]
        if (int(x) in X_map) and (int(y) in Y_map) and (int(t) in T_map):
            key = tuple([X_map[int(x)], Y_map[int(y)], T_map[int(t)]])
            if key in Grid2ID:
                count += 1
                if key in DB_DISTRI:
                    DB_DISTRI[key] += 1
                    DB_DISTRI_trajID[key].update(Grid2ID[key])
                else:
                    DB_DISTRI[key] = 1
                    DB_DISTRI_trajID[key] = set()
                    DB_DISTRI_trajID[key].update(Grid2ID[key])
    return DB_DISTRI, ID2Grid, DB_DISTRI_trajID

def get_distribution_feature_zipf(db, a):
    DB_DISTRI, ID2Grid, Grid2ID, DB_DISTRI_trajID = {}, {}, {}, {}
    Xmin, Ymin, Tmin = 1.044024, -179.9695933, 1176341492
    x_step, y_step, t_step = 0.02, 0.02, 3600*24*7
    X, Y, T = [], [], []
    for trajID in range(len(db)):
        for pointID in range(len(db[trajID])):
            if pointID == 0 or pointID == len(db[trajID]) - 1:
                continue
            point = db[trajID][pointID]
            [x, y, t] = point
            key = tuple([int((x - Xmin)/x_step), int((y - Ymin)/y_step), int((t - Tmin)/t_step)])
            ID2Grid[(trajID, pointID)] = key
            if key in Grid2ID:
                Grid2ID[key].add(trajID)
            else:
                Grid2ID[key] = set([trajID])
                X.append(key[0])
                Y.append(key[1])
                T.append(key[2])
    X.sort()
    Y.sort()
    T.sort()
    X_map, Y_map, T_map = {}, {}, {}
    for i in range(len(Grid2ID)):
        X_map[i] = X[i]
        Y_map[i] = Y[i]
        T_map[i] = T[i]
    
    x_loc, y_loc, t_loc = 0.75, 0.25, 0.75
    xbase = (1+len(Grid2ID))*x_loc
    ybase = (1+len(Grid2ID))*y_loc
    tbase = (1+len(Grid2ID))*t_loc
    
    total = 10000
    count = 0
    xz, yz, tz = np.random.zipf(a=a,size=total), np.random.zipf(a=a,size=total), np.random.zipf(a=a,size=total)
    x_max, y_max, t_max = xz.max(), yz.max(), tz.max()
    x_scala, y_scala, t_scala = len(Grid2ID)/x_max/2, len(Grid2ID)/y_max/2, len(Grid2ID)/t_max/2
    
    for count in range(total):
        x = xbase+random.choice([1,-1])*random.uniform(x_scala*(xz[count]-1),x_scala*xz[count])
        y = ybase+random.choice([1,-1])*random.uniform(y_scala*(yz[count]-1),y_scala*yz[count])
        t = tbase+random.choice([1,-1])*random.uniform(t_scala*(tz[count]-1),t_scala*tz[count])
        if (int(x) in X_map) and (int(y) in Y_map) and (int(t) in T_map):
            key = tuple([X_map[int(x)], Y_map[int(y)], T_map[int(t)]])
            if key in Grid2ID:
                if key in DB_DISTRI:
                    DB_DISTRI[key] += 1
                    DB_DISTRI_trajID[key].update(Grid2ID[key])
                else:
                    DB_DISTRI[key] = 1
                    DB_DISTRI_trajID[key] = set()
                    DB_DISTRI_trajID[key].update(Grid2ID[key])
    return DB_DISTRI, ID2Grid, DB_DISTRI_trajID

def get_query_workload_data(DB_DISTRI, num=100):
    K, V = list(DB_DISTRI.keys()), list(DB_DISTRI.values())
    np.random.seed(1)
    query_workload = []
    sample_value = np.array(V)
    sample_value = sample_value/np.sum(sample_value)
    while len(query_workload) < num:
        index = int(np.random.choice(len(sample_value), 1, p=sample_value))
        query_workload.append(K[index])
    return DB_DISTRI, query_workload[:int(num/2)], query_workload[int(num/2):]


def get_query_workload_gau(DB_DISTRI, num=100):
    K, V = list(DB_DISTRI.keys()), list(DB_DISTRI.values())
    np.random.seed(1)
    query_workload = []
    sample_value = np.array(V)
    sample_value = sample_value/np.sum(sample_value)
    while len(query_workload) < num:
        index = int(np.random.choice(len(sample_value), 1, p=sample_value))
        query_workload.append(K[index])
    return DB_DISTRI, query_workload[:int(num/2)], query_workload[int(num/2):]

def get_query_workload_zipf(DB_DISTRI, num=100):
    K, V = list(DB_DISTRI.keys()), list(DB_DISTRI.values())
    np.random.seed(1)
    query_workload = []
    sample_value = np.array(V)
    sample_value = sample_value/np.sum(sample_value)
    while len(query_workload) < num:
        index = int(np.random.choice(len(sample_value), 1, p=sample_value))
        query_workload.append(K[index])
    return DB_DISTRI, query_workload[:int(num/2)], query_workload[int(num/2):]

def deform_SED_whole(DB, sim_DB):
    res = []
    for i in range(len(DB)):
        _, error = sed_error(DB[i], sim_DB[i])
        res.append(error)
    return sum(res)/len(res)*100*1000

def deform_SED_return(DB, sim_DB, Rtree_ref, Rtree_sim, QUERY):   
    res = []
    x_length, y_length, t_length = 0.02, 0.02, 3600*24*7
    Xmin, Ymin, Tmin = 1.044024, -179.9695933, 1176341492
    for i in range(len(QUERY)):
        db, sim_db = [], []
        (x_idx, y_idx, t_idx) = QUERY[i]
        x_center, y_center, t_center = Xmin+x_length*(0.5+x_idx), Ymin+y_length*(0.5+y_idx), Tmin+t_length*(0.5+t_idx)
        ref_R = Rtree_ref.range_query((x_center-x_length/2,
                                       y_center-y_length/2,
                                       t_center-t_length/2,
                                       x_center+x_length/2,
                                       y_center+y_length/2,
                                       t_center+t_length/2))
        sim_R = Rtree_sim.range_query((x_center-x_length/2,
                                       y_center-y_length/2,
                                       t_center-t_length/2,
                                       x_center+x_length/2,
                                       y_center+y_length/2,
                                       t_center+t_length/2))
        A = set([item.object for item in ref_R])
        B = set([item.object for item in sim_R])
        
        
        
        if len(B) == 0:
            for ob in A:
                db.append(DB[ob])
                sim_db.append([DB[ob][0],DB[ob][-1]])
        else:
            for ob in B:
                db.append(DB[ob])
                sim_db.append(sim_DB[ob])
        res.append(deform_SED_whole(db, sim_db))
    return sum(res)/len(res)

class Sample():
    def __init__(self, db, x_step=0.02, y_step=0.02, t_step=3600*24*7):
        self.points = []
        self.traj_ids = []
        self.point_indices = [] 
        max_points_per_node=50
        max_depth=10
        
        for trajID in range(len(db)):
            for pointID in range(len(db[trajID])):
                if pointID == 0 or pointID == len(db[trajID]) - 1:
                    continue  
                point = db[trajID][pointID]
                self.points.append([point[0], point[1], point[2]])
                self.traj_ids.append(trajID)
                self.point_indices.append((trajID, pointID))
       
        if self.points:
            self.balltree = BallTree(np.array(self.points))
            self._build_partition_index(max_points_per_node, max_depth)
        else:
            self.balltree = None
            self.partition_index = {}
        
        self.sample_key = list(self.partition_index.keys())
        self.sample_value = np.array([len(ids) for ids in self.partition_index.values()])
        self.sample_value = self.sample_value / np.sum(self.sample_value)
    
    def _build_partition_index(self, max_points_per_node, max_depth):
        self.partition_index = defaultdict(list)
        
        for i, point in enumerate(self.points):
            neighbors = self.balltree.query_ball_point(point, r=self._calculate_radius(max_points_per_node))
            partition_key = tuple(np.mean([self.points[j] for j in neighbors], axis=0))
            self.partition_index[partition_key].append(i)
    
    def _calculate_radius(self, max_points_per_node):
        if len(self.points) < 100:
            return 0.1
        return np.percentile([np.min(self.balltree.query(p, k=max_points_per_node+1)[0]) for p in self.points], 50)
    
    def get_sample_list_data(self, num=1, random_state=None):
        if random_state is not None:
            np.random.seed(random_state)
        
        samples = []
        while len(samples) < num:
            partition_idx = np.random.choice(len(self.sample_key), p=self.sample_value)
            partition_key = self.sample_key[partition_idx]
            point_indices = self.partition_index[partition_key]
            selected_idx = random.choice(point_indices)
            samples.append(self.point_indices[selected_idx])
        
        return samples
    
    def get_sample_list_uniform(self, num=1, random_state=None):
        if random_state is not None:
            random.seed(random_state)
        all_indices = list(range(len(self.point_indices)))
        selected_indices = random.sample(all_indices, min(num, len(all_indices)))
        return [self.point_indices[i] for i in selected_indices]
    
    def get_partition_stats(self):
        return {
            'num_partitions': len(self.partition_index),
            'avg_points_per_partition': np.mean([len(v) for v in self.partition_index.values()]),
            'max_points_per_partition': max([len(v) for v in self.partition_index.values()])
        }
'''
def range_query_operator(ref_balltree, sim_balltree, QUERY, ref_traj_ids='', sim_traj_ids='', verbose=False):    
    
    F1_scores = []
    x_length, y_length, t_length = 0.02, 0.02, 3600*24*7
    Xmin, Ymin, Tmin = 1.044024, -179.9695933, 1176341492
    #Xmin, Ymin, Tmin = 1.054024, 179.9695933+60-1309-297+0.1+1185.5+0.26+0.22, 1180341492
    ee, ef, fe, ff = 0, 0, 0, 0
    
    for i in range(len(QUERY)):
        (x_idx, y_idx, t_idx) = QUERY[i]
        #print("x_idx, y_idx, t_idx",x_idx, y_idx, t_idx)
        
        x_center = Xmin + x_length*(0.5+x_idx)
        y_center = Ymin + y_length*(0.5+y_idx)
        t_center = Tmin + t_length*(0.5+t_idx)
 
        min_bounds = (x_center-x_length/2, y_center-y_length/2, t_center-t_length/2)
        max_bounds = (x_center+x_length/2, y_center+y_length/2, t_center+t_length/2)        
        
        sim_points = sim_balltree.data if hasattr(sim_balltree, 'data') else np.empty((0, 3))
        if len(sim_points) > 0:
            in_range_sim = np.all((sim_points >= min_bounds) & (sim_points <= max_bounds), axis=1)
            sim_indices = np.where(in_range_sim)[0]
            #B = set(sim_traj_ids[i] for i in sim_indices) if sim_traj_ids else set()
            Bb = [sim_points[i] for i in sim_indices]
            
            B = set()  # 使用集合自动去重
            for x, y, z in Bb:
                bbox = (x, y, z, x, y, z)
                matches = list(sim_balltree.idx.intersection(bbox, objects=True))
                if matches:
                    B.add(matches[0].object)  
        else:
            B = set()

        ref_points = ref_balltree.data
        if len(ref_points) > 0:
            in_range_ref = np.all((ref_points >= min_bounds) & (ref_points <= max_bounds), axis=1)
            ref_indices = np.where(in_range_ref)[0]
            print("ref_indices",len(ref_indices))
            # 添加打印查询到的点

                
            Aa = [ref_points[i] for i in ref_indices]
            #print("Aa",Aa)
            
            A = set()  # 使用集合自动去重
            for x, y, z in Aa:
                bbox = (x, y, z, x, y, z)
                matches = list(ref_balltree.idx.intersection(bbox, objects=True))
                if matches:
                    A.add(matches[0].object)  # 使用add方法而不是append
       
        else:
            A = set()
        
        A = list(A) if len(A) > 0 else []
        B = list(B) if len(B) > 0 else []    
            
            
        
        if verbose and A != B:
            print('A & B', A, B)
            
        if len(A) == 0 and len(B) == 0:
            ee += 1
            F1_scores.append(0.0)
        elif len(A) == 0 or len(B) == 0:
            if len(A) == 0: ef += 1
            else: fe += 1
            F1_scores.append(0.0)
        else:
            ff += 1
            A = A or []
            B = B or []
            #print("A",A)
            #print("B",B)
            
            lengab=0
            s = {}  # 改用字典来存储标记状态
            for i in range(len(B)):
                for j in range(len(A)):
                    if np.array_equal(B[i], A[j]):
                        if B[i] not in s:  # 检查是否已标记
                            lengab += 1
                            s[B[i]] = 1  # 标记为已处理
                        break  # 找到匹配后可以跳出内层循环
            print("P = len(A&B)/len(B) R = len(A&B)/len(A)", lengab, len(B), len(A))
            P = lengab/len(B) if len(B) > 0 else 0  # 避免除以0
            R = lengab/len(A) if len(A) > 0 else 0  # 避免除以0
            if (P+R) == 0:
                F1_scores.append(0.0)
            else:
                F1_scores.append((2*P*R)/(P+R))
    
        print("ee, ef, fe, ff", ee, ef, fe, ff)
    print(f"Processing {len(QUERY)} queries") 
    return sum(F1_scores) / len(F1_scores) if F1_scores else 0.0
'''
def range_query_operator(ref_balltree, sim_balltree, QUERY, ref_traj_ids='', sim_traj_ids='', verbose=False):    
    
    F1_scores = []
    x_length, y_length, t_length = 0.02, 0.02, 3600*24*7
    Xmin, Ymin, Tmin = 1.044024, -179.9695933, 1176341492
    ee, ef, fe, ff = 0, 0, 0, 0
    
    for i in range(len(QUERY)):
        (x_idx, y_idx, t_idx) = QUERY[i]
        x_center = Xmin + x_length*(0.5+x_idx)
        y_center = Ymin + y_length*(0.5+y_idx)
        t_center = Tmin + t_length*(0.5+t_idx)
        
        # 计算查询范围边界（rtree格式：min_x, min_y, min_t, max_x, max_y, max_t）
        bounds = (x_center-x_length/2, y_center-y_length/2, t_center-t_length/2,
                  x_center+x_length/2, y_center+y_length/2, t_center+t_length/2)
        
        # 使用范围查询获取轨迹ID
        A = set()
        try:
            if hasattr(ref_balltree, 'range_query_with_traj_ids'):
                A = ref_balltree.range_query_with_traj_ids(bounds)
            else:
                # 回退方法：使用rtree直接查询
                if hasattr(ref_balltree, 'idx') and ref_balltree.idx is not None:
                    matches = list(ref_balltree.idx.intersection(bounds, objects=True))
                    for match in matches:
                        if match is not None:
                            # 检查是否是Item对象
                            if hasattr(match, 'object'):
                                traj_id = match.object
                                if traj_id is not None:
                                    A.add(traj_id)
                            else:
                                # 如果不是Item对象，直接使用match作为trajID
                                A.add(match)
            if verbose:
                print(f"ref_balltree范围查询: 找到{len(A)}个轨迹ID: {A}")
        except Exception as e:
            if verbose:
                print(f"ref_balltree查询错误: {e}")
        
        B = set()
        try:
            # 使用范围查询获取轨迹ID
            if hasattr(sim_balltree, 'range_query_with_traj_ids'):
                B = sim_balltree.range_query_with_traj_ids(bounds)
            else:
                # 回退方法：使用rtree直接查询
                if hasattr(sim_balltree, 'idx') and sim_balltree.idx is not None:
                    matches = list(sim_balltree.idx.intersection(bounds, objects=True))
                    for match in matches:
                        if match is not None:
                            # 检查是否是Item对象
                            if hasattr(match, 'object'):
                                traj_id = match.object
                                if traj_id is not None:
                                    B.add(traj_id)
                            else:
                                # 如果不是Item对象，直接使用match作为trajID
                                B.add(match)
            if verbose:
                print(f"sim_balltree范围查询: 找到{len(B)}个轨迹ID: {B}")
        except Exception as e:
            if verbose:
                print(f"sim_balltree范围查询错误: {e}")
        
        # 范围查询已经返回所有匹配的轨迹ID，不需要限制数量
        
        # 添加调试信息
        if verbose or i < 2:  # 前2个查询总是显示调试信息
            print(f"查询 {i}: 中心点({x_center:.6f}, {y_center:.6f}, {t_center:.6f})")
            print(f"ref_balltree数据量: {len(ref_balltree.data) if hasattr(ref_balltree, 'data') else 'N/A'}")
            print(f"sim_balltree数据量: {len(sim_balltree.data) if hasattr(sim_balltree, 'data') else 'N/A'}")
            print(f"ref_balltree.idx存在: {hasattr(ref_balltree, 'idx')}")
            print(f"sim_balltree.idx存在: {hasattr(sim_balltree, 'idx')}")
            if hasattr(ref_balltree, 'idx'):
                print(f"ref_balltree.idx.data长度: {len(ref_balltree.idx.data) if hasattr(ref_balltree.idx, 'data') else 'N/A'}")
                print(f"ref_balltree.idx.obj长度: {len(ref_balltree.idx.obj) if hasattr(ref_balltree.idx, 'obj') else 'N/A'}")
            if hasattr(sim_balltree, 'idx'):
                print(f"sim_balltree.idx.data长度: {len(sim_balltree.idx.data) if hasattr(sim_balltree.idx, 'data') else 'N/A'}")
                print(f"sim_balltree.idx.obj长度: {len(sim_balltree.idx.obj) if hasattr(sim_balltree.idx, 'obj') else 'N/A'}")
            print(f"A (轨迹ID集合): {A}")
            print(f"B (轨迹ID集合): {B}")
            
        if len(A) == 0 and len(B) == 0:
            ee += 1
            F1_scores.append(1.0)  # 两个集合都为空，认为完全匹配
        elif len(A) == 0 or len(B) == 0:
            if len(A) == 0: ef += 1
            else: fe += 1
            F1_scores.append(0.0)
        else:
            ff += 1
            # 计算轨迹ID的交集
            intersection = A.intersection(B)
            lengab = len(intersection)
            
            print("P = len(A&B)/len(B) R = len(A&B)/len(A)", lengab, len(B), len(A))
            P = lengab/len(B) if len(B) > 0 else 0 
            R = lengab/len(A) if len(A) > 0 else 0 
            if (P+R) == 0:
                F1_scores.append(0.0)
            else:
                F1_scores.append((2*P*R)/(P+R))
    
        print("ee, ef, fe, ff", ee, ef, fe, ff)
    print(f"Processing {len(QUERY)} queries")
    return sum(F1_scores) / len(F1_scores) if F1_scores else 0.0 
    
def edr(ts_a, ts_b, eps):
    ts_a, ts_b = np.array(ts_a), np.array(ts_b)
    M, N = len(ts_a), len(ts_b)
    cost = np.ones((M, N))
    cost[0, 0] = 0
    for i in range(1, M):
        cost[i, 0] = i
    for j in range(1, N):
        cost[0, j] = j
    for i in range(1, M):
        for j in range(1, N):
            if np.linalg.norm(ts_a[i][0:2]-ts_b[j][0:2])<eps:
                choices = 0
            else:
                choices = 1
            cost[i, j] = min(cost[i-1, j-1]+choices, cost[i, j-1]+1, cost[i-1, j]+1)
    return cost[-1, -1]

def t2vec(ts_a, ts_b):
    ts_a, ts_b = np.array(ts_a), np.array(ts_b)
    ts_a[:, [0, 1]], ts_b[:, [0, 1]] = ts_a[:, [1, 0]], ts_b[:, [1, 0]]
    ts_a, ts_b = ts_a[:,0:2], ts_b[:,0:2]
    ts_a_token, ts_b_token = Main.get_seq(ts_a.T), Main.get_seq(ts_b.T)
    _, each_step_forw_a = submit(get_m0(), ts_a_token)
    _, each_step_forw_b = submit(get_m0(), ts_b_token)
    return np.linalg.norm(each_step_forw_a[0,-1,:]-each_step_forw_b[0,-1,:])

def get_sync_traj(traj,start_time,end_time,A,B):
    D_ = []
    syn_time = start_time
    ps = traj[A]
    pe = traj[A+1]
    time_ratio = 1 if (pe[2]- ps[2]) == 0  else (syn_time-ps[2]) / (pe[2]-ps[2])
    syn_x = ps[0] + (pe[0] - ps[0]) * time_ratio
    syn_y = ps[1] + (pe[1] - ps[1]) * time_ratio
    D_.append([syn_x,syn_y,syn_time])
    for idx in range(A+1, B+1):
        if syn_time==traj[idx][2]:
            continue
        else:
            D_.append(traj[idx])
    if D_[-1][2] != end_time:
        syn_time = end_time
        ps = traj[B]
        pe = traj[B+1]
        time_ratio = 1 if (pe[2] - ps[2]) == 0  else (syn_time-ps[2]) / (pe[2]-ps[2])
        syn_x = ps[0] + (pe[0] - ps[0]) * time_ratio
        syn_y = ps[1] + (pe[1] - ps[1]) * time_ratio
        D_.append([syn_x,syn_y,syn_time])
    return D_

def get_block_trajs(DB, A, xmin, ymin, tmin, xmax, ymax, tmax):
    ref_DB = []
    for a in A:
        traj = DB[a]
        ref_db = []
        for pts in traj:
            if pts[0]>=xmin and pts[0]<=xmax and pts[1]>=ymin and pts[1]<=ymax and pts[2]>=tmin and pts[2]<=tmax:
                ref_db.append(pts)
        ref_DB.append(ref_db)
    return ref_DB

def knn_edr_query_offline(DB, Rtree_ref, test_query):
    x_length, y_length, t_length = 0.02, 0.02, 3600*24*7
    Xmin, Ymin, Tmin = 1.044024, -179.9695933, 1176341492
    GroundQuerySet, interval, record = [], [], {}
    repeat = {}
    for i in range(len(test_query)):
        (x_idx, y_idx, t_idx) = test_query[i]
        if (x_idx, y_idx, t_idx) in repeat:
            continue
        repeat[(x_idx, y_idx, t_idx)] = 1
        x_center, y_center, t_center = Xmin+x_length*(0.5+x_idx), Ymin+y_length*(0.5+y_idx), Tmin+t_length*(0.5+t_idx)
        ref_R = Rtree_ref.range_query((x_center-x_length/2,
                                       y_center-y_length/2,
                                       t_center-t_length/2,
                                       x_center+x_length/2,
                                       y_center+y_length/2,
                                       t_center+t_length/2))
        A = set([item.object for item in ref_R])
        A = list(A)
        if len(A) > 1:
            interval.append((A, test_query[i]))
            ref_DB = get_block_trajs(DB, A, x_center-x_length/2,y_center-y_length/2,t_center-t_length/2,x_center+x_length/2,y_center+y_length/2,t_center+t_length/2)
            GroundSet, QuerySet = [], []
            for q_ in range(len(ref_DB)):
                query = ref_DB[q_]
                ground = []
                for c_ in range(len(ref_DB)):
                    data = ref_DB[c_]
                    if (A[q_], A[c_]) in record:
                        ground.append([record[(A[q_], A[c_])], A[c_]])
                    else:
                        tmp = edr(query, data, eps=0.02)
                        ground.append([tmp, A[c_]])
                        record[(A[q_], A[c_])] = tmp
                ground.sort(key=lambda s:(s[0]))
                GroundSet.append(ground)
                QuerySet.append(query)
            GroundQuerySet.append((GroundSet, QuerySet))
    return GroundQuerySet, interval

def knn_edr_query_online(GroundQuerySet, interval, Rtree_sim, sim_DB, k=3):
    x_length, y_length, t_length = 0.02, 0.02, 3600*24*7
    Xmin, Ymin, Tmin = 1.044024, -179.9695933, 1176341492
    result = []
    for (A, test_query), (GroundSet, QuerySet) in zip(interval, GroundQuerySet):
        (x_idx, y_idx, t_idx) = test_query
        x_center, y_center, t_center = Xmin+x_length*(0.5+x_idx), Ymin+y_length*(0.5+y_idx), Tmin+t_length*(0.5+t_idx)
        sim_R = Rtree_sim.range_query((x_center-x_length/2,
                                       y_center-y_length/2,
                                       t_center-t_length/2,
                                       x_center+x_length/2,
                                       y_center+y_length/2,
                                       t_center+t_length/2))
        B = set([item.object for item in sim_R])
        B = list(B)
        if  len(set(A)) == 0 or len(set(B)) == 0:
            continue
        win_sim_DB = get_block_trajs(sim_DB, B, x_center-x_length/2,y_center-y_length/2,t_center-t_length/2,x_center+x_length/2,y_center+y_length/2,t_center+t_length/2)
        for ground, query in zip(GroundSet, QuerySet):
            predict = []
            for j in range(len(win_sim_DB)):
                predict.append([edr(query, win_sim_DB[j], eps=0.02), B[j]])         
            predict.sort(key=lambda s:(s[0]))              
            predict_tmp, ground_tmp = [], []
            for predict_i in range(0, min(k,len(predict))):
                predict_tmp.append(predict[predict_i][1])
            for ground_i in range(0, min(k,len(ground))):
                ground_tmp.append(ground[ground_i][1])
            result.append(len(set(predict_tmp)&set(ground_tmp))/min(k,len(ground)))
    return sum(result)/len(result)

def knn_t2v_query_offline(DB, Rtree_ref, test_query):
    x_length, y_length, t_length = 0.02, 0.02, 3600*24*7
    Xmin, Ymin, Tmin = 1.044024, -179.9695933, 1176341492
    GroundQuerySet, interval, record = [], [], {}
    repeat = {}
    for i in range(len(test_query)):
        (x_idx, y_idx, t_idx) = test_query[i]
        if (x_idx, y_idx, t_idx) in repeat:
            continue
        repeat[(x_idx, y_idx, t_idx)] = 1
        x_center, y_center, t_center = Xmin+x_length*(0.5+x_idx), Ymin+y_length*(0.5+y_idx), Tmin+t_length*(0.5+t_idx)
        ref_R = Rtree_ref.range_query((x_center-x_length/2,
                                       y_center-y_length/2,
                                       t_center-t_length/2,
                                       x_center+x_length/2,
                                       y_center+y_length/2,
                                       t_center+t_length/2))
        A = set([item.object for item in ref_R])
        A = list(A)
        if len(A) > 1:
            interval.append((A, test_query[i]))
            ref_DB = get_block_trajs(DB, A, x_center-x_length/2,y_center-y_length/2,t_center-t_length/2,x_center+x_length/2,y_center+y_length/2,t_center+t_length/2)
            GroundSet, QuerySet = [], []
            for q_ in range(len(ref_DB)):
                query = ref_DB[q_]
                ground = []
                for c_ in range(len(ref_DB)):
                    data = ref_DB[c_]
                    if (A[q_], A[c_]) in record:
                        ground.append([record[(A[q_], A[c_])], A[c_]])
                    else:
                        tmp = t2vec(query, data)
                        ground.append([tmp, A[c_]])
                        record[(A[q_], A[c_])] = tmp
                ground.sort(key=lambda s:(s[0]))
                GroundSet.append(ground)
                QuerySet.append(query)
            GroundQuerySet.append((GroundSet, QuerySet))
    return GroundQuerySet, interval

def knn_t2v_query_online(GroundQuerySet, interval, Rtree_sim, sim_DB, k=3):
    x_length, y_length, t_length = 0.02, 0.02, 3600*24*7
    Xmin, Ymin, Tmin = 1.044024, -179.9695933, 1176341492
    result = []
    for (A, test_query), (GroundSet, QuerySet) in zip(interval, GroundQuerySet):
        (x_idx, y_idx, t_idx) = test_query
        x_center, y_center, t_center = Xmin+x_length*(0.5+x_idx), Ymin+y_length*(0.5+y_idx), Tmin+t_length*(0.5+t_idx)
        sim_R = Rtree_sim.range_query((x_center-x_length/2,
                                       y_center-y_length/2,
                                       t_center-t_length/2,
                                       x_center+x_length/2,
                                       y_center+y_length/2,
                                       t_center+t_length/2))
        B = set([item.object for item in sim_R])
        B = list(B)
        if  len(set(A)) == 0 or len(set(B)) == 0:
            continue        
        win_sim_DB = get_block_trajs(sim_DB, B, x_center-x_length/2,y_center-y_length/2,t_center-t_length/2,x_center+x_length/2,y_center+y_length/2,t_center+t_length/2)
        for ground, query in zip(GroundSet, QuerySet):
            predict = []
            for j in range(len(win_sim_DB)):
                predict.append([t2vec(query, win_sim_DB[j]), B[j]])         
            predict.sort(key=lambda s:(s[0]))  
            predict_tmp, ground_tmp = [], []
            for predict_i in range(0, min(k,len(predict))):
                predict_tmp.append(predict[predict_i][1])
            for ground_i in range(0, min(k,len(ground))):
                ground_tmp.append(ground[ground_i][1])
            result.append(len(set(predict_tmp)&set(ground_tmp))/min(k,len(ground)))            
    return sum(result)/len(result)

def join(Q_sync,D_sync,Q_start,Q_end,eps=0.01):
    for i in range(Q_start, Q_end):
        if np.linalg.norm(np.array(Q_sync[i])-np.array(D_sync[i]))<eps:
            continue
        else:
            return False
    return True

def sync(traj):
    dict_sync = {}
    for i in range(len(traj)-1):
        ps = traj[i]
        pe = traj[i+1]
        if pe[2] - ps[2] <= 1:
            dict_sync[ps[2]] = ps[0:2]
            dict_sync[pe[2]] = pe[0:2]
            continue
        else:
            dict_sync[ps[2]] = ps[0:2]
            for i in range(ps[2]+1,pe[2]):
                syn_time = i
                time_ratio = 1 if (pe[2]- ps[2]) == 0  else (syn_time-ps[2]) / (pe[2]-ps[2])
                syn_x = ps[0] + (pe[0] - ps[0]) * time_ratio
                syn_y = ps[1] + (pe[1] - ps[1]) * time_ratio
                dict_sync[i] = [syn_x,syn_y]
            dict_sync[pe[2]] = pe[0:2]
    return dict_sync

def join_query_operator(ref_DB, sim_DB, Rtree_ref, Rtree_sim, query):
    x_length, y_length, t_length = 0.02, 0.02, 3600*24*7
    Xmin, Ymin, Tmin = 1.044024, -179.9695933, 1176341492
    F1 = []
    for i in range(len(query)):
        (x_idx, y_idx, t_idx) = query[i]
        x_center, y_center, t_center = Xmin+x_length*(0.5+x_idx), Ymin+y_length*(0.5+y_idx), Tmin+t_length*(0.5+t_idx)
        ref_R = Rtree_ref.range_query((x_center-x_length/2,
                                       y_center-y_length/2,
                                       t_center-t_length/2,
                                       x_center+x_length/2,
                                       y_center+y_length/2,
                                       t_center+t_length/2))
        A = set([item.object for item in ref_R])
        sim_R = Rtree_sim.range_query((x_center-x_length/2,
                                       y_center-y_length/2,
                                       t_center-t_length/2,
                                       x_center+x_length/2,
                                       y_center+y_length/2,
                                       t_center+t_length/2))
        B = set([item.object for item in sim_R])        

        A = list(A)
        B = list(B)
        win_ref_DB = get_block_trajs(ref_DB, A, x_center-x_length/2,y_center-y_length/2,t_center-t_length/2,x_center+x_length/2,y_center+y_length/2,t_center+t_length/2)
        win_sim_DB = get_block_trajs(sim_DB, B, x_center-x_length/2,y_center-y_length/2,t_center-t_length/2,x_center+x_length/2,y_center+y_length/2,t_center+t_length/2)

        for q_ in range(len(win_ref_DB)):
            ground, predict = set(), set()
            Query = win_ref_DB[q_]
            Q_sync = sync(Query)
            Q_start, Q_end = Query[0][2], Query[-1][2]
            for c_ in range(len(win_ref_DB)):
                #for ground
                D1_sync = sync(win_ref_DB[c_])
                D_start, D_end = win_ref_DB[c_][0][2], win_ref_DB[c_][-1][2]
                if (D_start >= Q_start and D_start <= Q_end) or (D_end >= Q_start and D_end <= Q_end):
                    if join(Q_sync,D1_sync,max(Q_start, D_start),min(Q_end,D_end),eps=0.05):
                        ground.add(A[c_])
            for c_ in range(len(win_sim_DB)):
                #for predict
                D2_sync = sync(win_sim_DB[c_])
                D_start, D_end = win_sim_DB[c_][0][2], win_sim_DB[c_][-1][2]
                if (D_start >= Q_start and D_start <= Q_end) or (D_end >= Q_start and D_end <= Q_end):
                    if join(Q_sync,D2_sync,max(Q_start, D_start),min(Q_end,D_end),eps=0.05):
                        predict.add(B[c_])
                    
            if ground == set() and predict == set():
                F1.append(1.0)
            if ground == set() and predict != set():
                F1.append(0.0)
            if ground != set() and predict == set():
                F1.append(0.0)
            if ground != set() and predict != set():
                P = len(ground&predict)/len(predict)
                R = len(ground&predict)/len(ground)
                if (P+R) == 0:
                    F1.append(0.0)
                else:
                    F1.append((2*P*R)/(P+R))
    return sum(F1)/len(F1)

def call_traclus(trajs, A):
    traj_set = []
    for ts in trajs:
        traj_set.append([Point(ts[i:i+2][0], ts[i:i+2][1]) for i in range(0, len(ts), 2)])

    # part 1: partition
    all_segs = approximate_trajectory_partitioning(traj_set[0], theta=5.0, traj_id=A[0])
    for i in range(1, len(traj_set)):
        part = approximate_trajectory_partitioning(traj_set[i], theta=5.0, traj_id=A[i])
        all_segs += part
    
    norm_cluster, remove_cluster = line_segment_clustering(all_segs, min_lines=3, epsilon=0.03)

    return norm_cluster

def get_clusters(norm_cluster):
    clusters = []
    traj_cluster_dict = {}
    for nc in range(len(norm_cluster)):
        cluster=[]
        for segment in norm_cluster[nc]:
            cluster.append(segment.traj_id)
            if segment.traj_id in traj_cluster_dict:
                traj_cluster_dict[segment.traj_id].add(nc)
            else:
                traj_cluster_dict[segment.traj_id] = set()
                traj_cluster_dict[segment.traj_id].add(nc)            
        clusters.append(set(cluster))
    return clusters, traj_cluster_dict

def get_input(traj_db):
    ts=[]
    for traj in traj_db:
        ts.append(np.array(traj)[:,0:2].reshape(1,-1).tolist()[0])
    return ts

def clustering_offline(DB, DB_TREE, query):
    Ts_DB, ID = [], []
    x_length, y_length, t_length = 0.02, 0.02, 3600*24*7
    Xmin, Ymin, Tmin = 1.044024, -179.9695933, 1176341492
    repeat = {}
    for i in range(len(query)):
        (x_idx, y_idx, t_idx) = query[i]
        if (x_idx, y_idx, t_idx) in repeat:
            continue
        x_center, y_center, t_center = Xmin+x_length*(0.5+x_idx), Ymin+y_length*(0.5+y_idx), Tmin+t_length*(0.5+t_idx)
        
        
        ref_R = DB_TREE.range_query((x_center-x_length/2,
                                       y_center-y_length/2,
                                       t_center-t_length/2,
                                       x_center+x_length/2,
                                       y_center+y_length/2,
                                       t_center+t_length/2))
        A = set([item.object for item in ref_R])
        A = list(A)
        if len(A) > 0:
            ref_DB = get_block_trajs(DB, A, x_center-x_length/2,y_center-y_length/2,t_center-t_length/2,x_center+x_length/2,y_center+y_length/2,t_center+t_length/2)
            ts_DB = get_input(ref_DB)
            Ts_DB += ts_DB
            ID += A
    norm_cluster_DB = call_traclus(Ts_DB, ID)
    clusters_DB, traj_cluster_dict_DB = get_clusters(norm_cluster_DB)
    return traj_cluster_dict_DB
    
def clustering_online(traj_cluster_dict_DB, sim_DB, SIMDB_TREE, query):
    F1ALL, Ts_SIMDB, ID = [], [], []
    x_length, y_length, t_length = 0.02, 0.02, 3600*24*7
    Xmin, Ymin, Tmin = 1.044024, -179.9695933, 1176341492
    repeat = {}
    for i in range(len(query)):
        (x_idx, y_idx, t_idx) = query[i]
        if (x_idx, y_idx, t_idx) in repeat:
            continue
        x_center, y_center, t_center = Xmin+x_length*(0.5+x_idx), Ymin+y_length*(0.5+y_idx), Tmin+t_length*(0.5+t_idx)
        sim_R = SIMDB_TREE.range_query((x_center-x_length/2,
                                       y_center-y_length/2,
                                       t_center-t_length/2,
                                       x_center+x_length/2,
                                       y_center+y_length/2,
                                       t_center+t_length/2))
        B = set([item.object for item in sim_R])
        B = list(B)
        if len(B) > 0:
            simDB = get_block_trajs(sim_DB, B, x_center-x_length/2,y_center-y_length/2,t_center-t_length/2,x_center+x_length/2,y_center+y_length/2,t_center+t_length/2)
            ts_SIMDB = get_input(simDB)
            Ts_SIMDB += ts_SIMDB
            ID += B
    
    norm_cluster_simDB = call_traclus(Ts_SIMDB, ID)
    clusters_simDB, traj_cluster_dict_simDB = get_clusters(norm_cluster_simDB)
    
    CO, CS, COS = 0, 0, 0
    for i in range(len(sim_DB)-1):
        for j in range(i+1, len(sim_DB)):
            refind, simind = 0, 0
            if (i in traj_cluster_dict_DB) and (j in traj_cluster_dict_DB):
                if len(traj_cluster_dict_DB[i]&traj_cluster_dict_DB[j])!=0:
                    refind=1
            if (i in traj_cluster_dict_simDB) and (j in traj_cluster_dict_simDB):
                if len(traj_cluster_dict_simDB[i]&traj_cluster_dict_simDB[j])!=0:
                    simind=1
            CO+=refind
            CS+=simind
            if refind==1 and simind==1:
                COS+=1
    if CS == 0 or CO == 0:
        return 0
    P = COS/CS
    R = COS/CO
    F1 = (2*P*R)/(P+R)
    F1ALL.append(F1)
    return sum(F1ALL)/len(F1ALL)

def get_xyt_min(ref_DB):
    X = []
    Y = []
    T = []
    c = 0
    for trajID in range(len(ref_DB)):
        for pointID in range(len(ref_DB[trajID])):
            point = ref_DB[trajID][pointID]
            X.append(point[0])
            Y.append(point[1])
            T.append(point[2])
            c += 1
    xmin, ymin, tmin, xmax, ymax, tmax = min(X), min(Y), min(T), max(X), max(Y), max(T)
    
if __name__ == '__main__':
    print('This is the data util.')

def rdp_simplify(trajectory: np.ndarray, epsilon: float) -> np.ndarray:
    """
    Ramer-Douglas-Peucker (RDP) simplification algorithm.
    
    Args:
        trajectory: Input trajectory [N, 3] (x, y, t)
        epsilon: Distance threshold deviation
        
    Returns:
        Simplified trajectory [M, 3]
    """
    if len(trajectory) < 3:
        return trajectory
        
    # Find the point with the maximum distance
    dmax = 0.0
    index = 0
    end = len(trajectory) - 1
    
    # Vectorized distance calculation
    # Line passing through trajectory[0] and trajectory[-1]
    start_pt = trajectory[0, :2]
    end_pt = trajectory[-1, :2]
    
    # Vector from start to end
    line_vec = end_pt - start_pt
    line_len_sq = np.dot(line_vec, line_vec)
    
    if line_len_sq < 1e-10:
        # Start and end are the same
        dists = np.linalg.norm(trajectory[1:-1, :2] - start_pt, axis=1)
    else:
        # Project points onto the line
        # vectors from start to points
        pts_vec = trajectory[1:-1, :2] - start_pt
        
        # t = dot(pts_vec, line_vec) / line_len_sq
        t = np.dot(pts_vec, line_vec) / line_len_sq
        t = np.clip(t, 0, 1)
        
        # Critical point on the line
        projection = start_pt + np.outer(t, line_vec)
        
        # Distances
        dists = np.linalg.norm(trajectory[1:-1, :2] - projection, axis=1)
        
    if len(dists) > 0:
        index = np.argmax(dists) + 1 # +1 because we skipped the first point
        dmax = dists[index - 1]
    
    # If max distance is greater than epsilon, recursively simplify
    if dmax > epsilon:
        # Recursive call
        rec_results1 = rdp_simplify(trajectory[:index+1], epsilon)
        rec_results2 = rdp_simplify(trajectory[index:], epsilon)
        
        # Build the result list (removing duplicate point at index)
        return np.vstack((rec_results1[:-1], rec_results2))
    else:
        return np.vstack((trajectory[0], trajectory[-1]))
