"""Task Manipulation."""
# -*- coding: utf-8 -*-
# task.py
import numpy as np

class Task:
    """
    Representation of a single scheduling task.
    Each task has:
        - resource requirements (list of ints, e.g., [cpu, mem])
        - duration (int)
        - label / identifier
        - timestamps (arrival, start_exec, completion)
    Used in DeepRM-like scheduling environment.
    """

    def __init__(self, resources, duration, label=None):
        self.resources = resources
        self.duration = duration
        self.label = label if label else "task"
        self.dimension = len(self.resources)

        # --- Time tracking ---
        self.arrival_time = None
        self.start_exec_time = None
        self.completion_time = None

        # --- Deadline (added for deadline-based penalty) ---
        self.deadline = None  # ✅ default; will be set in env.generate_dynamic_tasks()


        # --- Placement coordinates ---
        # These are filled when task is placed onto a node,
        # but we initialize them here to avoid AttributeErrors
        self.start_row = [0 for _ in range(len(self.resources))]
        self.start_col = [0 for _ in range(len(self.resources))]

        # --- State flags ---
        self.finished = False
        self.failed = False

    # ----------------------------------------------------------------------

    def mark_arrival(self, time_now):
        self.arrival_time = time_now

    def mark_start(self, time_now):
        self.start_exec_time = time_now

    def mark_completion(self, time_now):
        self.completion_time = time_now
        self.finished = True

    # ----------------------------------------------------------------------

    def summary(self, bg_shape):
        """
        Return a 2D binary matrix representation of this task
        (DeepRM-style). Each task always outputs a single 2D grid
        of shape bg_shape, regardless of the number of resource
        dimensions.

        bg_shape : tuple (rows, cols)
        """
        # Initialize per-resource 2D matrices
        state_matrices = [np.ones(bg_shape) * 255 for _ in range(len(self.resources))]

        # Defensive fallback if coordinates are missing
        if not hasattr(self, "start_row") or not hasattr(self, "start_col"):
            self.start_row = [0 for _ in range(len(self.resources))]
            self.start_col = [0 for _ in range(len(self.resources))]

        for i in range(len(self.resources)):
            row = int(self.start_row[i])
            col = int(self.start_col[i])
            height = int(self.resources[i])
            width = int(self.duration)

            # Clip to prevent overflow beyond bg_shape
            max_r = min(row + height, bg_shape[0])
            max_c = min(col + width, bg_shape[1])

            for r in range(row, max_r):
                for c in range(col, max_c):
                    state_matrices[i][r, c] = 0

        # --- Collapse any multi-resource (3D) matrices into one 2D ---
        arr = np.array(state_matrices)
        if arr.ndim == 3:
            arr = np.mean(arr, axis=0)  # average over resource channels
        elif arr.ndim == 1:
            arr = arr[0]  # single element list edge case

        return arr

    # ----------------------------------------------------------------------

    def execution_time(self):
        """Return total execution time if completed."""
        if self.start_exec_time is not None and self.completion_time is not None:
            return self.completion_time - self.start_exec_time
        return 0

    def waiting_time(self):
        """Return total waiting time before execution."""
        if self.arrival_time is not None and self.start_exec_time is not None:
            return self.start_exec_time - self.arrival_time
        return 0

    def response_time(self):
        """Return total response time (arrival → completion)."""
        if self.arrival_time is not None and self.completion_time is not None:
            return self.completion_time - self.arrival_time
        return 0

    def __repr__(self):
        return f"<Task {self.label}: res={self.resources}, dur={self.duration}>"
