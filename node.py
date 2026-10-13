"""Node Manipulation."""

import collections
import os
import numpy as np
from PIL import Image

class Node(object):
    """Node."""

    def __init__(self, resources, duration, label):
        self.resources = resources
        self.duration = duration
        self.label = label
        self.dimension = len(resources)
        self.timestep_counter = 0
        self.scheduled_tasks = []
        # State matrices: 0 for occupied, 255 for vacant
        self.state_matrices = [np.full((duration, resource), 255, dtype=np.uint8) for resource in resources]
        self._state_matrices_capacity = [[resource] * duration for resource in resources]
        self.total_busy_time = 0

    def can_run(self, task):
        """Check if this node can run the task (resource and duration fit)."""
        for r_task, r_node in zip(task.resources, self.resources):
            if r_task > r_node:
                return False
        if task.duration > self.duration:
            return False
        return True

    def schedule(self, task):
        """Schedule task on node."""
        start_time = self._satisfy(self._state_matrices_capacity, task.resources, task.duration)
        if start_time == -1:
            return False
        else:
            for i in range(task.dimension):
                self._occupy(self.state_matrices[i], self._state_matrices_capacity[i],
                             task.resources[i], task.duration, start_time)

            # 🔹 Record execution timestamps
            task.start_exec_time = self.timestep_counter + start_time
            task.completion_time = task.start_exec_time + task.duration

            self.scheduled_tasks.append((task, self.timestep_counter + task.duration))
            return True

    def timestep(self):
        """Proceed to the next timestep."""
        self.timestep_counter += 1
        for i in range(self.dimension):
            temp = np.delete(self.state_matrices[i], 0, axis=0)
            temp = np.append(temp, np.array([[255] * temp.shape[1]]), axis=0)
            self.state_matrices[i] = temp
        for i in range(self.dimension):
            self._state_matrices_capacity[i].pop(0)
            self._state_matrices_capacity[i].append(self.resources[i])
        # Remove completed tasks
        self.scheduled_tasks = [(task, end_time) for task, end_time in self.scheduled_tasks if self.timestep_counter < end_time]
        if len(self.scheduled_tasks) > 0:
            self.total_busy_time += 1

    def utilization(self):
        """Calculate utilization."""
        total_slots = sum(self.resources) * self.duration
        occupied_slots = sum([collections.Counter(matrix.flatten()).get(0, 0) for matrix in self.state_matrices])
        return occupied_slots / total_slots

    def calculate_load(self):
        """Calculate total load as the sum of durations of all scheduled tasks."""
        return sum(task.duration for task, _ in self.scheduled_tasks)

    def calculate_resource_utilization(self):
        """Calculate the current resource utilization for each resource type."""
        utilization = [0] * self.dimension
        for task, _ in self.scheduled_tasks:
            for i in range(self.dimension):
                utilization[i] += task.resources[i]
        return utilization

    def summary(self, bg_shape=None):
        """Generate node state representation with utilization and task load."""
        if self.dimension > 0:
            temp = self._expand(self.state_matrices[0], bg_shape)
            for i in range(1, self.dimension):
                temp = np.concatenate((temp, self._expand(self.state_matrices[i], bg_shape)), axis=1)

            utilization_matrix = np.full((temp.shape[0], 1), self.utilization(), dtype=np.float32)
            temp = np.concatenate((temp, utilization_matrix), axis=1)

            task_count_matrix = np.full((temp.shape[0], 1),
                                        len(self.scheduled_tasks) / max(1, self.queue_size), dtype=np.float32)
            temp = np.concatenate((temp, task_count_matrix), axis=1)

            # 🔹 Normalize pixel values
            temp = temp.astype(np.float32)
            if np.max(temp) > 0:
                temp = temp / 255.0

            return temp

    def _satisfy(self, capacity_matrix, required_resources, required_duration):
        """Find the earliest time to schedule task with required resources for required duration."""
        p1, p2 = 0, 0
        duration_bound = min(len(capacity) for capacity in capacity_matrix)
        while p1 < duration_bound and p2 < required_duration:
            if any(capacity_matrix[i][p1] < required_resources[i] for i in range(len(required_resources))):
                p1 += 1
                p2 = 0
            else:
                p1 += 1
                p2 += 1
        return p1 - required_duration if p2 == required_duration else -1

    def _occupy(self, state_matrix, state_matrix_capacity, required_resource, required_duration, start_time):
        """Occupy state matrix slots for required resources and duration."""
        for i in range(start_time, start_time + required_duration):
            for j in range(required_resource):
                state_matrix[i, len(state_matrix[i]) - state_matrix_capacity[i] + j] = 0
            state_matrix_capacity[i] -= required_resource

    def _expand(self, matrix, bg_shape=None):
        """Expand state matrix to specified background shape."""
        if bg_shape and bg_shape[0] >= matrix.shape[0] and bg_shape[1] >= matrix.shape[1]:
            temp = matrix
            if bg_shape[0] > matrix.shape[0]:
                temp = np.concatenate((temp, np.full((bg_shape[0] - matrix.shape[0], matrix.shape[1]), 255, dtype=np.uint8)), axis=0)
            if bg_shape[1] > matrix.shape[1]:
                temp = np.concatenate((temp, np.full((bg_shape[0], bg_shape[1] - matrix.shape[1]), 255, dtype=np.uint8)), axis=1)
            return temp
        else:
            return matrix

    def __repr__(self):
        return f'Node(state_matrices={self.state_matrices}, label={self.label})'
