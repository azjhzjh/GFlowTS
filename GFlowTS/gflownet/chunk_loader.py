# -*- coding: utf-8 -*-
"""
分块轨迹加载器 - 解决内存不足问题

支持按需加载轨迹数据，避免一次性将所有轨迹加载到内存中。
适用于处理超长全局轨迹的 GFlowNet 训练。
"""

import numpy as np
import gc
from typing import List, Tuple, Optional, Iterator, Callable
import sys
import os

# 添加项目路径
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

import data_utils as F


class ChunkedTrajectoryLoader:
    """
    分块轨迹加载器
    
    使用生成器按需加载轨迹，避免一次性加载所有数据到内存。
    支持迭代访问，每次返回一个 chunk 的数据。
    
    用法:
        loader = ChunkedTrajectoryLoader(path, 0, 1000, chunk_size=64)
        for chunk_trajs, chunk_raw in loader:
            # 处理当前 chunk
            process(chunk_trajs)
            # 处理完后自动释放该 chunk 内存
    """
    
    def __init__(
        self,
        path: str,
        start: int,
        end: int,
        chunk_size: int = 64,
        precompute_stats: bool = True,
        normalize_fn: Optional[Callable] = None
    ):
        """
        Args:
            path: 轨迹数据路径前缀
            start: 起始索引
            end: 结束索引
            chunk_size: 每个 chunk 的轨迹数量
            precompute_stats: 是否预计算全局统计量（用于归一化）
            normalize_fn: 归一化函数，接收 (traj, stats) 返回归一化后的轨迹
        """
        self.path = path
        self.start = start
        self.end = end
        self.chunk_size = chunk_size
        self.normalize_fn = normalize_fn
        
        # 预扫描获取有效轨迹索引
        self.valid_indices = self._scan_valid_indices()
        self.total_trajectories = len(self.valid_indices)
        self.num_chunks = (self.total_trajectories + chunk_size - 1) // chunk_size
        
        # 全局统计量（用于归一化）
        self.global_stats = None
        if precompute_stats and self.total_trajectories > 0:
            self.global_stats = self._compute_global_stats_streaming()
        
        print(f"[ChunkedLoader] 发现 {self.total_trajectories} 条有效轨迹，分为 {self.num_chunks} 个 chunk")
    
    def _scan_valid_indices(self) -> List[int]:
        """扫描有效轨迹索引（不加载完整数据）"""
        valid = []
        for i in range(self.start, self.end):
            filepath = self.path + str(i)
            if os.path.exists(filepath) or os.path.exists(filepath + '.npy') or os.path.exists(filepath + '.txt'):
                valid.append(i)
        return valid
    
    def _compute_global_stats_streaming(self) -> dict:
        """流式计算全局统计量，使用采样策略避免内存不足
        
        采样策略：只扫描前 sample_count 条轨迹来估计全局统计量
        """
        # 采样数量：最多扫描 50 条轨迹
        sample_count = min(50, len(self.valid_indices))
        print(f"[ChunkedLoader] 采样 {sample_count} 条轨迹计算统计量...")
        
        x_min, x_max = float('inf'), float('-inf')
        y_min, y_max = float('inf'), float('-inf')
        t_min, t_max = float('inf'), float('-inf')
        
        # 均匀采样索引
        if len(self.valid_indices) <= sample_count:
            sample_indices = self.valid_indices
        else:
            step = len(self.valid_indices) // sample_count
            sample_indices = [self.valid_indices[i * step] for i in range(sample_count)]
        
        loaded_count = 0
        for idx in sample_indices:
            try:
                # 强制垃圾回收
                gc.collect()
                
                traj = F.to_traj(self.path + str(idx))
                if traj is not None and len(traj) > 0:
                    traj = np.array(traj, dtype=np.float32)  # 使用 float32 节省内存
                    
                    x_min = min(x_min, float(traj[:, 0].min()))
                    x_max = max(x_max, float(traj[:, 0].max()))
                    y_min = min(y_min, float(traj[:, 1].min()))
                    y_max = max(y_max, float(traj[:, 1].max()))
                    t_min = min(t_min, float(traj[:, 2].min()))
                    t_max = max(t_max, float(traj[:, 2].max()))
                    
                    loaded_count += 1
                    
                    # 立即释放内存
                    del traj
                    
            except (MemoryError, RuntimeError) as e:
                print(f"[ChunkedLoader] 内存警告：跳过轨迹 {idx}")
                gc.collect()
                continue
            except Exception:
                continue
            
            # 每条轨迹后清理
            gc.collect()
        
        print(f"[ChunkedLoader] 成功加载 {loaded_count}/{sample_count} 条轨迹")
        
        # 如果没有成功加载任何轨迹，使用默认值
        if loaded_count == 0:
            print("[ChunkedLoader] 警告：使用默认统计量")
            return {
                'x_min': 0.0, 'x_max': 1.0,
                'y_min': 0.0, 'y_max': 1.0,
                't_min': 0.0, 't_max': 1.0
            }
        
        return {
            'x_min': x_min, 'x_max': x_max,
            'y_min': y_min, 'y_max': y_max,
            't_min': t_min, 't_max': t_max
        }
    
    def _load_single_trajectory(self, idx: int) -> Optional[np.ndarray]:
        """加载单条轨迹"""
        try:
            traj = F.to_traj(self.path + str(idx))
            if traj is not None and len(traj) > 0:
                return np.array(traj)
        except Exception:
            pass
        return None
    
    def __iter__(self) -> Iterator[Tuple[List[np.ndarray], List[np.ndarray]]]:
        """
        迭代器：按 chunk 返回轨迹数据
        
        Yields:
            (normalized_trajs, raw_trajs): 归一化轨迹列表和原始轨迹列表
        """
        for chunk_idx in range(self.num_chunks):
            start_pos = chunk_idx * self.chunk_size
            end_pos = min(start_pos + self.chunk_size, self.total_trajectories)
            chunk_indices = self.valid_indices[start_pos:end_pos]
            
            raw_trajs = []
            normalized_trajs = []
            
            for idx in chunk_indices:
                traj = self._load_single_trajectory(idx)
                if traj is not None:
                    raw_trajs.append(traj)
                    
                    if self.normalize_fn is not None and self.global_stats is not None:
                        norm_traj = self.normalize_fn(traj, self.global_stats)
                        normalized_trajs.append(norm_traj)
                    else:
                        normalized_trajs.append(traj.copy())
            
            yield normalized_trajs, raw_trajs
            
            # 释放当前 chunk 内存
            del raw_trajs, normalized_trajs
            gc.collect()
    
    def get_chunk(self, chunk_idx: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """获取指定 chunk 的轨迹数据"""
        if chunk_idx < 0 or chunk_idx >= self.num_chunks:
            raise IndexError(f"Chunk index {chunk_idx} out of range [0, {self.num_chunks})")
        
        start_pos = chunk_idx * self.chunk_size
        end_pos = min(start_pos + self.chunk_size, self.total_trajectories)
        chunk_indices = self.valid_indices[start_pos:end_pos]
        
        raw_trajs = []
        normalized_trajs = []
        
        for idx in chunk_indices:
            traj = self._load_single_trajectory(idx)
            if traj is not None:
                raw_trajs.append(traj)
                
                if self.normalize_fn is not None and self.global_stats is not None:
                    norm_traj = self.normalize_fn(traj, self.global_stats)
                    normalized_trajs.append(norm_traj)
                else:
                    normalized_trajs.append(traj.copy())
        
        return normalized_trajs, raw_trajs
    
    def __len__(self) -> int:
        """返回 chunk 数量"""
        return self.num_chunks


class MemoryManager:
    """
    内存管理器 - 监控和管理内存使用
    """
    
    def __init__(self, memory_limit_ratio: float = 0.8, device: str = 'cpu'):
        """
        Args:
            memory_limit_ratio: 内存使用上限比例 (0-1)
            device: 设备类型 ('cpu' 或 'cuda')
        """
        self.memory_limit_ratio = memory_limit_ratio
        self.device = device
        self._check_cuda = device == 'cuda'
        
        try:
            import torch
            self._torch_available = True
        except ImportError:
            self._torch_available = False
    
    def get_memory_usage(self) -> dict:
        """获取当前内存使用情况"""
        result = {
            'cpu_percent': 0.5,  # 默认值
            'cpu_available_gb': 8.0  # 默认值
        }
        
        # 尝试使用 psutil
        try:
            import psutil
            mem = psutil.virtual_memory()
            result['cpu_percent'] = mem.percent / 100.0
            result['cpu_available_gb'] = mem.available / (1024**3)
        except ImportError:
            pass  # psutil 不可用时使用默认值
        
        if self._check_cuda and self._torch_available:
            import torch
            if torch.cuda.is_available():
                result['gpu_allocated_gb'] = torch.cuda.memory_allocated() / (1024**3)
                result['gpu_reserved_gb'] = torch.cuda.memory_reserved() / (1024**3)
                result['gpu_max_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                result['gpu_percent'] = result['gpu_allocated_gb'] / result['gpu_max_memory_gb']
        
        return result
    
    def should_clear_memory(self) -> bool:
        """判断是否需要清理内存"""
        usage = self.get_memory_usage()
        
        if usage['cpu_percent'] > self.memory_limit_ratio:
            return True
        
        if 'gpu_percent' in usage and usage['gpu_percent'] > self.memory_limit_ratio:
            return True
        
        return False
    
    def clear_memory(self, force_cuda: bool = False):
        """清理内存"""
        gc.collect()
        
        if (self._check_cuda or force_cuda) and self._torch_available:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
    
    def log_memory_status(self, prefix: str = ""):
        """打印内存状态"""
        usage = self.get_memory_usage()
        
        msg = f"{prefix}[Memory] CPU: {usage['cpu_percent']*100:.1f}% used, " \
              f"{usage['cpu_available_gb']:.2f} GB available"
        
        if 'gpu_percent' in usage:
            msg += f" | GPU: {usage['gpu_allocated_gb']:.2f}/{usage['gpu_max_memory_gb']:.2f} GB " \
                   f"({usage['gpu_percent']*100:.1f}%)"
        
        print(msg)


def normalize_trajectory(traj: np.ndarray, stats: dict) -> np.ndarray:
    """标准归一化函数"""
    normalized = traj.copy().astype(np.float32)
    normalized[:, 0] = (normalized[:, 0] - stats['x_min']) / (stats['x_max'] - stats['x_min'] + 1e-6)
    normalized[:, 1] = (normalized[:, 1] - stats['y_min']) / (stats['y_max'] - stats['y_min'] + 1e-6)
    normalized[:, 2] = (normalized[:, 2] - stats['t_min']) / (stats['t_max'] - stats['t_min'] + 1e-6)
    return normalized


def compute_gt_hits_for_chunk(
    raw_trajectories: List[np.ndarray],
    queries: List[Tuple]
) -> List[set]:
    """为一个 chunk 的轨迹计算 ground truth hits"""
    batch_hits = []
    
    for traj in raw_trajectories:
        hits = set()
        for q_idx, query in enumerate(queries):
            x_min, x_max, y_min, y_max, t_min, t_max = query
            
            in_x = (traj[:, 0] >= x_min) & (traj[:, 0] <= x_max)
            in_y = (traj[:, 1] >= y_min) & (traj[:, 1] <= y_max)
            in_t = (traj[:, 2] >= t_min) & (traj[:, 2] <= t_max)
            
            if np.any(in_x & in_y & in_t):
                hits.add(q_idx)
        
        batch_hits.append(hits)
    
    return batch_hits
