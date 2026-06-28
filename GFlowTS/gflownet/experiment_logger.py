# -*- coding: utf-8 -*-
"""
GFlowNet 实验日志和可视化模块

记录和可视化：
- 平均保留点数
- F1 ≥ 0.95 达成率
- F1-points Pareto Front
- Forward/Backward Loss 曲线
"""

import os
import json
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from collections import defaultdict


class ExperimentLogger:
    """
    实验日志记录器
    
    记录训练过程中的关键指标
    """
    
    def __init__(
        self,
        log_dir: str = 'logs',
        experiment_name: Optional[str] = None
    ):
        """
        Args:
            log_dir: 日志目录
            experiment_name: 实验名称
        """
        self.log_dir = log_dir
        
        if experiment_name is None:
            experiment_name = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        self.experiment_name = experiment_name
        self.experiment_dir = os.path.join(log_dir, experiment_name)
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        # 日志数据
        self.epoch_logs: List[Dict] = []
        self.step_logs: List[Dict] = []
        self.pareto_points: List[Tuple[float, float, float]] = []  # (f1, sparsity, reward)
        
        # 累计统计
        self.total_trajectories = 0
        self.successful_trajectories = 0
        
    def log_step(
        self,
        step: int,
        loss_forward: float,
        loss_backward: float,
        mean_reward: float,
        mean_f1: float = 0.0,
        mean_points: float = 0.0,
        success_rate: float = 0.0,
        **kwargs
    ):
        """记录单步训练指标"""
        log_entry = {
            'step': step,
            'loss_forward': loss_forward,
            'loss_backward': loss_backward,
            'mean_reward': mean_reward,
            'mean_f1': mean_f1,
            'mean_points': mean_points,
            'success_rate': success_rate,
            **kwargs
        }
        self.step_logs.append(log_entry)
    
    def log_epoch(
        self,
        epoch: int,
        loss_forward: float,
        loss_backward: float,
        mean_reward: float,
        mean_f1: float,
        mean_sparsity: float,
        mean_points: float,
        success_rate: float,
        val_f1: float = 0.0,
        val_sparsity: float = 0.0,
        val_success_rate: float = 0.0,
        replay_buffer_size: int = 0,
        backward_lr: float = 0.0,
        **kwargs
    ):
        """记录 Epoch 指标"""
        log_entry = {
            'epoch': epoch,
            'loss_forward': loss_forward,
            'loss_backward': loss_backward,
            'mean_reward': mean_reward,
            'mean_f1': mean_f1,
            'mean_sparsity': mean_sparsity,
            'mean_points': mean_points,
            'success_rate': success_rate,
            'val_f1': val_f1,
            'val_sparsity': val_sparsity,
            'val_success_rate': val_success_rate,
            'replay_buffer_size': replay_buffer_size,
            'backward_lr': backward_lr,
            **kwargs
        }
        self.epoch_logs.append(log_entry)
        
        # 更新 Pareto Front
        self.pareto_points.append((mean_f1, mean_sparsity, mean_reward))
        
        # 打印日志
        self._print_epoch_summary(log_entry)
    
    def _print_epoch_summary(self, log: Dict):
        """打印 Epoch 摘要"""
        print(f"\n{'='*60}")
        print(f"Epoch {log['epoch']} 摘要")
        print(f"{'='*60}")
        print(f"  平均保留点数: {log['mean_points']:.1f}")
        print(f"  平均压缩率: {log['mean_sparsity']*100:.2f}%")
        print(f"  平均 F1: {log['mean_f1']:.4f}")
        print(f"  F1 ≥ 0.95 达成率: {log['success_rate']*100:.1f}%")
        print(f"  Forward Loss: {log['loss_forward']:.4f}")
        print(f"  Backward Loss: {log['loss_backward']:.4f}")
        print(f"  Backward LR: {log['backward_lr']:.2e}")
        print(f"  Replay Buffer: {log['replay_buffer_size']}")
        
        if log['val_f1'] > 0:
            print(f"\n  [Validation]")
            print(f"    Val F1: {log['val_f1']:.4f}")
            print(f"    Val Sparsity: {log['val_sparsity']*100:.2f}%")
            print(f"    Val Success Rate: {log['val_success_rate']*100:.1f}%")
    
    def log_trajectory_stats(
        self,
        trajectories: List[Dict],
        prefix: str = ''
    ):
        """记录轨迹统计"""
        if not trajectories:
            return
        
        f1_scores = [t.get('f1', 0) for t in trajectories]
        sparsities = [t.get('sparsity', 0) for t in trajectories]
        points = [t.get('num_points', 0) for t in trajectories]
        
        stats = {
            'count': len(trajectories),
            'mean_f1': np.mean(f1_scores),
            'std_f1': np.std(f1_scores),
            'mean_sparsity': np.mean(sparsities),
            'mean_points': np.mean(points),
            'success_count': sum(1 for f1 in f1_scores if f1 >= 0.95),
            'success_rate': sum(1 for f1 in f1_scores if f1 >= 0.95) / len(f1_scores)
        }
        
        print(f"\n{prefix}轨迹统计:")
        print(f"  总数: {stats['count']}")
        print(f"  平均 F1: {stats['mean_f1']:.4f} ± {stats['std_f1']:.4f}")
        print(f"  平均保留点: {stats['mean_points']:.1f}")
        print(f"  成功率: {stats['success_rate']*100:.1f}%")
        
        return stats
    
    def compute_pareto_front(self) -> List[Tuple[float, float, float]]:
        """
        计算 F1-Points Pareto Front
        
        Returns:
            Pareto 前沿点列表 [(f1, sparsity, reward), ...]
        """
        if not self.pareto_points:
            return []
        
        # 转换为数组
        points = np.array(self.pareto_points)
        
        # 找 Pareto 前沿
        # 目标：最大化 F1，最小化 sparsity (等价于最小化点数)
        pareto_mask = np.ones(len(points), dtype=bool)
        
        for i, (f1_i, sp_i, _) in enumerate(points):
            for j, (f1_j, sp_j, _) in enumerate(points):
                if i != j:
                    # j 支配 i: F1_j >= F1_i 且 sp_j <= sp_i，且至少一个严格
                    if f1_j >= f1_i and sp_j <= sp_i and (f1_j > f1_i or sp_j < sp_i):
                        pareto_mask[i] = False
                        break
        
        pareto_front = points[pareto_mask].tolist()
        
        # 按 F1 排序
        pareto_front.sort(key=lambda x: x[0], reverse=True)
        
        return [tuple(p) for p in pareto_front]
    
    def save(self):
        """保存日志到文件"""
        # 保存 epoch 日志
        epoch_path = os.path.join(self.experiment_dir, 'epoch_logs.json')
        with open(epoch_path, 'w') as f:
            json.dump(self.epoch_logs, f, indent=2)
        
        # 保存 step 日志
        step_path = os.path.join(self.experiment_dir, 'step_logs.json')
        with open(step_path, 'w') as f:
            json.dump(self.step_logs, f, indent=2)
        
        # 保存 Pareto Front
        pareto = self.compute_pareto_front()
        pareto_path = os.path.join(self.experiment_dir, 'pareto_front.json')
        with open(pareto_path, 'w') as f:
            json.dump(pareto, f, indent=2)
        
        print(f"\n日志已保存到: {self.experiment_dir}")
    
    def plot_loss_curves(self, save_path: Optional[str] = None):
        """
        绘制 Forward/Backward Loss 曲线
        (如果有 matplotlib)
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("[Warning] matplotlib 未安装，跳过绘图")
            return
        
        if not self.epoch_logs:
            return
        
        epochs = [log['epoch'] for log in self.epoch_logs]
        loss_f = [log['loss_forward'] for log in self.epoch_logs]
        loss_b = [log['loss_backward'] for log in self.epoch_logs]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(epochs, loss_f, 'b-', label='Forward Loss (TB)', linewidth=2)
        ax.plot(epochs, loss_b, 'r-', label='Backward Loss (TLM)', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss Curves')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path is None:
            save_path = os.path.join(self.experiment_dir, 'loss_curves.png')
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Loss 曲线已保存: {save_path}")
    
    def plot_pareto_front(self, save_path: Optional[str] = None):
        """绘制 F1-Sparsity Pareto Front"""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return
        
        pareto = self.compute_pareto_front()
        if not pareto:
            return
        
        f1_vals = [p[0] for p in pareto]
        sp_vals = [p[1] * 100 for p in pareto]  # 转为百分比
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 所有点
        all_f1 = [p[0] for p in self.pareto_points]
        all_sp = [p[1] * 100 for p in self.pareto_points]
        ax.scatter(all_sp, all_f1, c='lightblue', alpha=0.5, label='All Points')
        
        # Pareto 前沿
        ax.scatter(sp_vals, f1_vals, c='red', s=100, label='Pareto Front')
        ax.plot(sp_vals, f1_vals, 'r--', alpha=0.5)
        
        # F1 阈值线
        ax.axhline(y=0.95, color='green', linestyle='--', label='F1 = 0.95')
        
        ax.set_xlabel('Sparsity (%)')
        ax.set_ylabel('F1 Score')
        ax.set_title('F1 vs Sparsity Pareto Front')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path is None:
            save_path = os.path.join(self.experiment_dir, 'pareto_front.png')
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Pareto Front 已保存: {save_path}")
    
    def plot_metrics(self, save_path: Optional[str] = None):
        """绘制综合指标图"""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return
        
        if not self.epoch_logs:
            return
        
        epochs = [log['epoch'] for log in self.epoch_logs]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # F1 曲线
        ax1 = axes[0, 0]
        f1_vals = [log['mean_f1'] for log in self.epoch_logs]
        ax1.plot(epochs, f1_vals, 'b-', linewidth=2)
        ax1.axhline(y=0.95, color='r', linestyle='--', label='Target')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Mean F1')
        ax1.set_title('F1 Score')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 成功率曲线
        ax2 = axes[0, 1]
        success = [log['success_rate'] * 100 for log in self.epoch_logs]
        ax2.plot(epochs, success, 'g-', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Success Rate (%)')
        ax2.set_title('F1 ≥ 0.95 成功率')
        ax2.grid(True, alpha=0.3)
        
        # 平均点数曲线
        ax3 = axes[1, 0]
        points = [log['mean_points'] for log in self.epoch_logs]
        ax3.plot(epochs, points, 'm-', linewidth=2)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Mean Points')
        ax3.set_title('平均保留点数')
        ax3.grid(True, alpha=0.3)
        
        # Reward 曲线
        ax4 = axes[1, 1]
        rewards = [log['mean_reward'] for log in self.epoch_logs]
        ax4.plot(epochs, rewards, 'c-', linewidth=2)
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Mean Reward')
        ax4.set_title('平均奖励')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.experiment_dir, 'metrics.png')
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  指标图已保存: {save_path}")
    
    def generate_report(self) -> str:
        """生成实验报告"""
        if not self.epoch_logs:
            return "无数据"
        
        last = self.epoch_logs[-1]
        best_f1 = max(log['mean_f1'] for log in self.epoch_logs)
        best_success = max(log['success_rate'] for log in self.epoch_logs)
        
        pareto = self.compute_pareto_front()
        
        report = f"""
# GFlowNet + TLM 实验报告

## 实验: {self.experiment_name}

### 最终结果 (Epoch {last['epoch']})
- 平均 F1: {last['mean_f1']:.4f}
- 平均压缩率: {last['mean_sparsity']*100:.2f}%
- 平均保留点数: {last['mean_points']:.1f}
- F1 ≥ 0.95 达成率: {last['success_rate']*100:.1f}%

### 最佳结果
- 最佳 F1: {best_f1:.4f}
- 最佳成功率: {best_success*100:.1f}%

### Pareto Front (Top 5)
"""
        for i, (f1, sp, reward) in enumerate(pareto[:5]):
            report += f"  {i+1}. F1={f1:.4f}, Sparsity={sp*100:.2f}%, Reward={reward:.6f}\n"
        
        # 保存报告
        report_path = os.path.join(self.experiment_dir, 'report.md')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        return report
