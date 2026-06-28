# -*- coding: utf-8 -*-
import os

# 路径配置适配
# 检查是否在 AutoDL 环境 (通常包含 root/autodl-tmp)
if os.path.exists('F:/query/GFlowNet-one'):
    TRAJ_DATA_ROOT = 'F:/query/GFlowNet-one/TrajData'
else:
    # 本地环境
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    TRAJ_DATA_ROOT = os.path.join(BASE_DIR, 'TrajData')

PATHS = {
    'geolife_raw': os.path.join(TRAJ_DATA_ROOT, 'Geolife Trajectories 1.3/Data/'),
    'geolife_out': os.path.join(TRAJ_DATA_ROOT, 'Geolife_out/'),
    'best_model': os.path.join(TRAJ_DATA_ROOT, 'best_model.pt'),
}

"""
轨迹简化系统配置参数
"""

# 预处理参数
PREPROCESS_CONFIG = {
    'vmax': 200.0,  # 最大速度阈值 (m/s)，超过此值视为异常
    'v_stop': 0.5,  # 停留速度阈值 (m/s)
    'tau_stop': 300,  # 停留持续时间阈值 (秒)
    'angle_threshold': 3.14159,  # 角度突变阈值 (弧度)，约180度
    'missing_tolerance': 0.1,  # 缺失数据容忍比例，超过此比例则标记
    'interpolation_enabled': True,  # 是否启用插值修复缺失点
}

# 特征提取参数
FEATURE_CONFIG = {
    'R_thres': 0.85,  # 方向一致性阈值
    'M_time': 100,  # 时间分割的最小点数
    'time_window_days': 10,  # 时间窗口分割的时间跨度阈值（天）
    'density_k': 10,  # KDE密度估计的k值
}

# BallTree分割参数
BALLTREE_CONFIG = {
    'N_leaf': 5,  # 叶节点最小点数
    'maxDepth': 50,  # 最大深度
    'k_radius': 2.0,  # 半径系数 k，在 1.5~3 之间
    'lambda_depth': 0.1,  # 深度衰减系数 λ
    'large_node_threshold': 500,  # 大节点阈值，超过此值进行采样
    'sample_ratio': 0.2,  # 采样比例
    'min_sample_size': 200,  # 最小采样数量
}

# 策略管理参数
STRATEGY_CONFIG = {
    'f1_threshold': 0.5,  # F1分数阈值，低于此值淘汰策略
    'fitness_w1': 0.7,  # 适应度权重 w1 (F1_current)
    'fitness_w2': 0.3,  # 适应度权重 w2 (F1_current - F1_parent)
    'top_k_strategies': 10,  # 每层保留的top-K策略
    'mutation_rate': 0.1,  # 策略变异率
    'selection_ratio': 0.5,  # 每层选择前50%的点
}

# 连续性检测参数
CONTINUITY_CONFIG = {
    'tau_time_gap': 600,  # 时间间隔阈值（秒），10分钟
    'tau_space': 100.0,  # 空间距离阈值（米）
    'tau_dir': 0.3,  # 方向一致性差异阈值
    'merge_f1_improvement': 0.05,  # 合并后F1提升阈值
}

# 语义权重参数
SEMANTIC_WEIGHT_CONFIG = {
    'alpha': 0.4,  # 轨迹重要性权重 α
    'beta': 0.3,  # 时间中心性权重 β
    'gamma': 0.3,  # 空间权重 γ
}

