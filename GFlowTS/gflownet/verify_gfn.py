
import sys
import os
import torch
import numpy as np

# Add parent dir to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("Testing Imports...")
try:
    from gflownet.gfn_env import TrajectorySimplificationEnv
    from gflownet.models import HierarchicalGFlowNet
    from gflownet.train_tlm import TLMTrainer
    print("Imports Successful.")
except ImportError as e:
    print(f"Import Failed: {e}")
    sys.exit(1)

print("Testing Instantiation...")
try:
    # Pseudo data: 50 points, 3 dims (x,y,t)
    data = np.random.rand(50, 3)
    
    # 1. Env
    env = TrajectorySimplificationEnv(data)
    print("Environment Created.")
    
    # 2. Model
    model = HierarchicalGFlowNet(input_dim=3)
    print("Model Created.")
    
    # 3. Trainer
    trainer = TLMTrainer(data, device='cpu')
    print("Trainer Created.")
    
    # 4. Dry Run Train Step
    print("Attempting dry run training step...")
    loss_b, loss_tb = trainer.train_step_tlm(n_backward_updates=1, n_forward_updates=1)
    print(f"Dry Run Successful. Loss B: {loss_b}, Loss TB: {loss_tb}")

except Exception as e:
    print(f"Instantiation/Run Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("Verification Complete: Success")
