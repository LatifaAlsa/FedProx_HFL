"""Environment Manipulation."""
import json
import os
import random
import numpy as np
from cluster import Cluster
from node import Node
from task import Task
#from schedule import DeepRMScheduler, CompactScheduler, SpreadScheduler, RandomScheduler, FIFOScheduler


class Environment(object):
    def __init__(self, clusters, queue_size, backlog_size, tasks,
                 lambda_rate=5, move_rate=3, generation_rate=3):
        # … your existing topology init …
        self.clusters      = clusters
        self.queue_size    = queue_size
        self.backlog_size  = backlog_size

        # Task pools
        self.queue         = []
        self.backlog       = []
        self.tasks         = tasks    # static CSV list
        self._task_index   = 0        # next CSV task

        # Rates
        self.lambda_rate       = lambda_rate     # Poisson λ
        self.move_rate         = move_rate       # backlog→queue cap
        self.generation_rate   = generation_rate # arrivals cap per step

        # Internal clock & Poisson clock
        self.timestep_counter  = 0
        self.next_task_time    = np.random.exponential(1.0 / self.lambda_rate)

        # ---- Failure tracking (timeouts) ----
        for task in self.tasks:
            task.max_queue_wait = task.duration * 15  # or any multiplier you prefer
            task.max_cluster_wait = task.duration * 10  # shorter for cluster-level
        self.dispatch_fail_count = 0  # failed to leave global queue → cluster in time
        self.node_place_fail_count = 0  # failed to leave cluster queue → node in time
        self.deadline_fail_count = 0
        self.dispatch_failed_ids = set()
        self.node_failed_ids = set()
        self.deadline_fail_ids = set()

        # ---- NEW: shaping knobs (penalty weights) ----
        self.fail_penalty_dispatch = 1.0  # per task dropped from global ready queue
        self.fail_penalty_node = 0.5  # per task dropped from cluster queue
        self.fail_penalty_deadline = 2.0  # per task that misses its deadline

        # ---- NEW: per-step failure deltas (reset every timestep) ----
        self.failures_this_step = {'dispatch': 0, 'node': 0, 'deadline': 0}
        # Logging accumulator for penalty over a full round
        self.cumulative_penalty = 0.0


    def generate_dynamic_tasks(self):
        """
        Drip in up to `generation_rate` tasks when the Poisson clock fires.
        """
        tasks_generated = 0

        while (self.timestep_counter >= self.next_task_time
               and self._task_index < len(self.tasks)
               and tasks_generated < self.generation_rate
               and len(self.backlog) < self.backlog_size):
            # 1) Enqueue one static CSV task
            task = self.tasks[self._task_index]
            task.arrival_time = self.timestep_counter  # 🔹 new line
            task.deadline = task.arrival_time + max(100, 5 * task.duration)

            task.queue_entry_time = None  # will be set when moved to queue
            self.backlog.append(task)
            self._task_index += 1
            tasks_generated += 1

            # 2) Draw next inter‐arrival
            inter = np.random.exponential(1.0 / self.lambda_rate)
            self.next_task_time += inter


    def timestep(self):
        """
        1) Advance all nodes,
        2) Poisson arrivals → backlog,
        3) move up to move_rate tasks backlog→queue
        """
        # advance clock
        self.timestep_counter += 1
        # NEW: reset per-step failure deltas
        self.failures_this_step = {'dispatch': 0, 'node': 0, 'deadline': 0}
        # 1) tick every node
        for cluster in self.clusters:
            for node in cluster.nodes:
                node.timestep()

        # 2) drip in new tasks
        self.generate_dynamic_tasks()

        # 3) admit tasks into the ready queue
        moved = 0
        while (moved < self.move_rate
               and len(self.queue) < self.queue_size
               and self.backlog):
            task = self.backlog.pop(0)
            task.queue_entry_time = self.timestep_counter  # 🔹 record time of queue entry
            self.queue.append(task)
            moved += 1
        # ==================== TIMEOUT FAILURE CHECKS ====================

        # ---- 1) Global ready queue → Cluster dispatch failure ----
        keep_q = []
        for t in self.queue:
            ref = t.queue_entry_time if t.queue_entry_time is not None else t.arrival_time
            if ref is not None and (self.timestep_counter - ref) > t.max_queue_wait:
                if t.label not in self.dispatch_failed_ids:
                    self.dispatch_failed_ids.add(t.label)
                    self.dispatch_fail_count += 1
                    self.failures_this_step['dispatch'] += 1
                # drop (do not re-queue)
            else:
                keep_q.append(t)
        self.queue = keep_q

        # ---- 2) Cluster queue → Node scheduling failure ----
        for c in self.clusters:
            keep_c = []
            for t in c.cluster_queue:
                ref = t.enter_cluster_time if t.enter_cluster_time is not None else t.queue_entry_time or t.arrival_time
                if ref is not None and (self.timestep_counter - ref) > t.max_cluster_wait:
                    if t.label not in self.node_failed_ids:
                        self.node_failed_ids.add(t.label)
                        self.node_place_fail_count += 1
                        self.failures_this_step['node'] += 1
                    # drop task
                else:
                    keep_c.append(t)
            c.cluster_queue = keep_c

        # ---- 3) Deadline failure: completion after deadline ----
        for task in self.tasks:
            if task.completion_time is not None and task.deadline is not None:
                if task.completion_time > task.deadline:
                    if not hasattr(task, 'missed_deadline'):
                        task.missed_deadline = True
                        self.deadline_fail_count += 1
                        # NEW:
                        self.failures_this_step['deadline'] += 1





    def terminated(self):
        """
        Return True when no tasks remain in backlog, queue, or running on any node.
        """
        # check active nodes
        for cluster in self.clusters:
            for node in cluster.nodes:
                if node.utilization() > 0:
                    return False
        # check pending tasks
        if self.queue or self.backlog:
            return False
        # check pending tasks in cluster-level queues
        for cluster in self.clusters:
            if cluster.cluster_queue:
                return False
        return True






    def reward(self):
        """
        Compute reward: negative sum of inverse wait and service metrics,
        including waiting in ready queue, backlog, and cluster-level queues.
        """
        r = 0
        # penalty for tasks waiting in ready queue
        if self.queue:
            r += 1 / sum(task.duration for task in self.queue)
        # penalty for tasks waiting in backlog
        if self.backlog:
            r += 1 / sum(task.duration for task in self.backlog)
        # penalty for tasks waiting in cluster-level queues
        for cluster in self.clusters:
            if cluster.cluster_queue:
                r += 1 / sum(task.duration for task in cluster.cluster_queue)
        # reward for tasks currently running (inverse of duration)
        for cluster in self.clusters:
            for node in cluster.nodes:
                if node.scheduled_tasks:
                    r += 1 / sum(t[0].duration for t in node.scheduled_tasks)
        return -r
        




    def summary(self, bg_shape=None):
        # determine background dimensions
        if bg_shape is None:
            max_res = max(max(node.resources) for c in self.clusters for node in c.nodes)
            max_dur = max(node.duration for c in self.clusters for node in c.nodes)
            bg_shape = (max_dur, max_res)
        rows, cols = bg_shape

        # 🔸 STEP 1: aggregate and normalise cluster summaries
        cluster_mats = [c.summary(bg_shape) for c in self.clusters]
        cs_stack = np.stack(cluster_mats)
        cs_stack = cs_stack / np.max(cs_stack, axis=(1, 2), keepdims=True).clip(min=1e-6)
        cs_mean = np.mean(cs_stack, axis=0)
        cs_max = np.max(cs_stack, axis=0)
        cs_std = np.std(cs_stack, axis=0)
        cs = np.hstack([cs_mean, cs_max, cs_std])

        # 🔸 STEP 2: pad/trim rows
        if cs.shape[0] < rows:
            pad = np.zeros((rows - cs.shape[0], cs.shape[1]), dtype=cs.dtype)
            cs = np.vstack([cs, pad])
        else:
            cs = cs[:rows, :]

        # 🔸 STEP 3: queue and backlog (unchanged)
        q_tiles = [t.summary(bg_shape) for t in self.queue]
        empty_tile = np.zeros((rows, cols), dtype=np.float32)
        while len(q_tiles) < self.queue_size:
            q_tiles.append(empty_tile)
        q_mat = np.hstack(q_tiles[:self.queue_size])

        b_mat = np.zeros((rows, cols), dtype=np.float32)
        n_back = min(len(self.backlog), rows * cols)
        for i in range(n_back):
            r = i // cols
            c = i % cols
            b_mat[r, c] = 1.0

        return np.hstack([cs, q_mat, b_mat])






def load_env(lambda_rate=5, move_rate=3, regenerate_tasks=True, seed=None):
    """
    Loads the environment and optionally regenerates tasks with a new seed.
    """
    import random
    import numpy as np

    if regenerate_tasks:
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        tasks = _generate_tasks(seed=seed)  # ✅ generate fresh tasks
        #print(f"[Env] 🔁 Regenerated tasks with seed={seed}")
    else:
        tasks = _load_tasks()  # ✅ load from saved pickle / trace
        #print("[Env] 📦 Loaded static task trace from disk")

    # read configuration
    with open('conf/env.conf.json', 'r') as fr:
        data = json.load(fr)

    # build clusters with per-cluster queue sizes
    clusters = []
    for cj in data['clusters']:
        nodes = [Node(nj['resource_capacity'], nj['duration_capacity'], nj['label'])
                 for nj in cj['nodes']]
        qsz = cj.get('queue_size', data['queue_size'])
        clusters.append(Cluster(nodes, cj['label'], qsz))

    # create environment
    env = Environment(clusters,
                      data['queue_size'],
                      data['backlog_size'],
                      tasks,
                      lambda_rate=lambda_rate,
                      move_rate=move_rate)

    # 🔹 Link each cluster to the environment for failure tracking
    for c in clusters:
        c.environment = env

    # seed the first Poisson arrivals
    env.timestep()
    return env






def load_scheduler(environment, idx):
    from schedule import DeepRMScheduler  # local import avoids circularity
    return DeepRMScheduler(environment, idx)

# Task file helpers






def _load_tasks():
    if not os.path.isdir('__cache__/tasks'):
        os.makedirs('__cache__/tasks')
    path = '__cache__/tasks/tasks.csv'
    if not os.path.isfile(path):
        _generate_tasks()
    tasks = []
    with open(path, 'r') as fr:
        header = fr.readline().strip().split(',')
        res_idx = [i for i,col in enumerate(header) if col.startswith('resource')]
        dur_idx = header.index('duration')
        lbl_idx = header.index('label')
        for line in fr:
            parts = line.strip().split(',')
            res = [int(parts[i]) for i in res_idx]
            dur = int(parts[dur_idx])
            lbl = parts[lbl_idx]
            tasks.append(Task(res, dur, lbl))
    return tasks








def _generate_tasks(seed=None):
    """
    Generate a new set of heterogeneous tasks and save them to __cache__/tasks/tasks.csv.
    Adds stochastic variation in resource and duration ranges to promote non-uniform workloads.
    """
    import os, json, random, numpy as np

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    if not os.path.isdir('__cache__/tasks'):
        os.makedirs('__cache__/tasks')

    path = '__cache__/tasks/tasks.csv'
    with open('conf/task.pattern.conf.json', 'r') as fr:
        patterns = json.load(fr)

    with open(path, 'w') as fw:
        # Write CSV header
        if patterns:
            for i in range(len(patterns[0]['resource_range'])):
                fw.write(f'resource{i + 1},')
            fw.write('duration,label\n')

        lbl = 0

        # --------------------------------------------
        # 🔹 Generate tasks with heterogeneity
        # --------------------------------------------
        for pat in patterns:
            for _ in range(pat['batch_size']):
                lbl += 1

                # Add mild randomization to resource range
                res = []
                for r in pat['resource_range']:
                    base_low, base_high = r['lowerLimit'], r['upperLimit']
                    # Add ±10% random noise around defined limits
                    noisy_low = max(1, int(base_low * random.uniform(0.9, 1.1)))
                    noisy_high = max(noisy_low + 1, int(base_high * random.uniform(0.9, 1.2)))
                    res_val = random.randint(noisy_low, noisy_high)
                    res.append(str(res_val))

                # Add random variation to duration
                dur_low = pat['duration_range']['lowerLimit']
                dur_high = pat['duration_range']['upperLimit']
                # Add ±20% noise for heterogeneity
                noisy_dur_low = max(1, int(dur_low * random.uniform(0.8, 1.2)))
                noisy_dur_high = max(noisy_dur_low + 1, int(dur_high * random.uniform(0.8, 1.5)))
                dur = str(random.randint(noisy_dur_low, noisy_dur_high))

                # Write to file
                fw.write(','.join(res) + ',' + dur + ',task' + str(lbl) + '\n')

    # Optional debug output
    #print(f"[Task Generator] ✅ Generated {lbl} heterogeneous tasks → {path}")

    # ✅ Return reloaded tasks
    return _load_tasks()



