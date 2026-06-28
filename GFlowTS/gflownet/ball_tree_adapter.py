
import numpy as np
import sys
import os

# Ensure we can import from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from data_utils import CustomBallTree
    BALLTREE_AVAILABLE = True
except ImportError:
    BALLTREE_AVAILABLE = False
    print("Warning: data_utils.CustomBallTree not found. Using mock for testing.")

class BallTreeAdapter:
    """
    Adapter to interface GFlowNet with the existing CustomBallTree.
    Adds hierarchical error estimation and GFlowNet-specific navigation.
    """
    def __init__(self, trajectory_points):
        """
        Args:
            trajectory_points: (N, D) numpy array
        """
        self.points = np.array(trajectory_points)
        if BALLTREE_AVAILABLE:
            # Re-use existing advanced construction logic
            self.tree = CustomBallTree(self.points, min_leaf_size=10, use_semantic_split=True)
            self.root = self.tree.root
        else:
            self.tree = None
            self.root = None
        
        # Precompute features for GFlowNet
        if self.root:
            self._augment_node_features(self.root)

    def _augment_node_features(self, node):
        """
        Recursively compute error estimates for each node.
        Error Estimate: Maximum distance from any point in the node to the chord connecting start/end of node's time span.
        """
        if node is None:
            return

        # 1. Compute Representative Error (SED-like)
        # If we approximate this whole node by just 1 point (centroid), what's the worst error?
        node_points = np.array(node.points)
        if len(node_points) > 1:
            # Simple approximation: Max distance from centroid
            # A better heuristic for trajectory: PERPENDICULAR distance to the line segment of the node
            # But BallTree nodes aren't necessarily segments.
            # We use radius as a proxy for "Cost of ignoring this node"
            node.error_potential = node.radius
            
            # Density
            node.density = len(node_points) / (node.radius**3 + 1e-6)
        else:
            node.error_potential = 0.0
            node.density = 0.0

        # Recursion
        if node.left_child:
            self._augment_node_features(node.left_child)
        if node.right_child:
            self._augment_node_features(node.right_child)

    def get_root(self):
        return self.root

    def flatten_tree_nodes(self):
        """Returns a flat list of all nodes for embedding (BFS/DFS)"""
        nodes = []
        q = [self.root]
        while q:
            curr = q.pop(0)
            if curr:
                nodes.append(curr)
                if curr.left_child: q.append(curr.left_child)
                if curr.right_child: q.append(curr.right_child)
        return nodes
