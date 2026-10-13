import numpy as np
from tensorflow.keras.models import load_model
import os

def pad_to_shape(matrix, shape, fill=255):
    padded = np.full(shape, fill, dtype=np.float32)
    h, w = matrix.shape
    h_clip, w_clip = min(h, shape[0]), min(w, shape[1])
    padded[:h_clip, :w_clip] = matrix[:h_clip, :w_clip]
    return padded

class Cluster:



    """
    Cluster-level scheduling that uses a shared pretrained Node-DQN.
    """
    _node_dqn = None
    _model_path = os.path.expanduser("__cache__/models/dqn_node_global.keras")

    def __init__(self, nodes, label, queue_size):
        self.nodes = nodes
        self.label = label
        self.queue_size = queue_size
        self.cluster_queue = []

        for node in self.nodes:
            node.queue_size = queue_size

        if Cluster._node_dqn is None:
            print(f"[Cluster] Loading Node-DQN from {Cluster._model_path}")
            Cluster._node_dqn = load_model(Cluster._model_path)
        self.node_dqn = Cluster._node_dqn

        self.expected_shape = (23, 1610)



    def summary(self, bg_shape=None):
        num_nodes = len(self.nodes)
        num_features = len(self.nodes[0].resources) + 3
        mat = np.zeros((num_nodes, num_features), dtype=np.float32)

        # find global maxima across all clusters
        all_res = [r for n in self.nodes for r in n.resources]
        max_res_global = max(all_res) if all_res else 1.0

        for i, node in enumerate(self.nodes):
            mat[i, 0] = node.utilization()
            mat[i, 1] = len(node.scheduled_tasks) / max(1, self.queue_size)
            mat[i, 2:-1] = np.array(node.resources) / max_res_global
            mat[i, -1] = len(self.cluster_queue) / max(1, self.queue_size)

        return mat  # remove mat/np.max(mat)

    # in cluster.py (Cluster.utilization)
    def utilization(self):
        return np.mean([
            n.total_busy_time / max(1, self.environment.timestep_counter)
            for n in self.nodes
        ])

    def schedule(self, task):
        """
        Use only the pretrained Node-DQN to choose the best node for this task.
        Attempts to place the task on one of the cluster's nodes.
        Returns the selected node if successful, or None if no node can take it.
        """
        if self.node_dqn is None:
            raise RuntimeError(f"[FATAL] Node-DQN not initialized for cluster {self.label}")

        # Determine a background shape from node capacities
        bg_col = max(max(node.resources) for node in self.nodes)
        bg_row = max(node.duration for node in self.nodes)
        bg_shape = (bg_row, bg_col)

        # Build input for each node: [node_summary | task_summary]
        task_mat = task.summary(bg_shape)
        inputs = []
        for node in self.nodes:
            node_mat = node.summary(bg_shape)
            pair = np.hstack((node_mat, task_mat))
            inputs.append(pad_to_shape(pair, self.expected_shape))

        # Stack into batch and add channel dim
        X = np.stack(inputs)[..., np.newaxis]  # shape (num_nodes, H, W, 1)

        # Predict Q-values for each node
        qvals = self.node_dqn.predict(X, verbose=0)  # shape (num_nodes, action_dim)
        node_scores = qvals.max(axis=1)

        # Try nodes in descending order of predicted score
        for idx in np.argsort(-node_scores):
            if self.nodes[idx].schedule(task):
                return self.nodes[idx]  # ✅ successfully scheduled

        # ❌ No node could take the task
        # Do NOT increment old `failed_node_schedules` — that attribute no longer exists.
        # Failures are now handled by timeout tracking in env.timestep().
        # You can optionally increment the new counter here for convenience:
        if hasattr(self, 'environment') and self.environment is not None:
            # Optional: record immediate scheduling failure for analysis
            if not hasattr(self.environment, 'instant_node_fail_count'):
                self.environment.instant_node_fail_count = 0
            self.environment.instant_node_fail_count += 1

        return None

    def timestep(self):
        for node in self.nodes:
            node.timestep()
