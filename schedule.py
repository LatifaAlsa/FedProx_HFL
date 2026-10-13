"""Schedulers with two-stage DQN: cluster-level + node-level."""
import os
from abc import ABC, abstractmethod
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, GlobalAveragePooling2D
import env
import matplotlib.pyplot as plt
import random

class Action(object):
    """Schedule Action."""

    def __init__(self, task, node, cluster):
        self.task = task
        self.node = node
        self.cluster = cluster

    def __repr__(self):
        return f"Action(task={self.task.label} -> node={self.node.label}, cluster={self.cluster.label})"

class Scheduler(ABC):
    """Scheduler Interface."""
    @abstractmethod
    def schedule(self):
        pass






class FIFOScheduler(Scheduler):
    def __init__(self, environment):
        self.environment = environment
        self.label = "FIFOScheduler"

    def schedule(self):
        actions = []
        rewards = 0
        env = self.environment
        tasks = sorted(env.queue, key=lambda t: t.arrival_time or 0)

        for task in tasks:
            scheduled = False
            for cluster in env.clusters:
                for node in cluster.nodes:
                    if node.can_run(task):
                        if node.schedule(task):
                            env.queue.remove(task)
                            task.start_exec_time = env.timestep_counter
                            task.completion_time = env.timestep_counter + task.duration
                            actions.append(Action(task, cluster, node))
                            scheduled = True
                            break
                if scheduled:
                    break

        env.timestep()
        return actions, rewards



class RandomScheduler(Scheduler):
    def __init__(self, environment):
        self.environment = environment
        self.label = "RandomScheduler"

    def schedule(self):
        actions = []
        rewards = 0
        env = self.environment
        tasks = list(env.queue)
        random.shuffle(tasks)

        for task in tasks:
            clusters = list(env.clusters)
            random.shuffle(clusters)
            scheduled = False
            for cluster in clusters:
                nodes = list(cluster.nodes)
                random.shuffle(nodes)
                for node in nodes:
                    if node.can_run(task):
                        if node.schedule(task):
                            env.queue.remove(task)
                            task.start_exec_time = env.timestep_counter
                            task.completion_time = env.timestep_counter + task.duration
                            actions.append(Action(task, cluster, node))
                            scheduled = True
                            break
                if scheduled:
                    break

        env.timestep()
        return actions, rewards



class SpreadScheduler(Scheduler):
    def __init__(self, environment):
        self.environment = environment
        self.label = "SpreadScheduler"

    def schedule(self):
        actions = []
        rewards = 0
        env = self.environment
        tasks = list(env.queue)

        for task in tasks:
            best = None
            min_util = float('inf')
            for cluster in env.clusters:
                for node in cluster.nodes:
                    if node.can_run(task):
                        util = node.utilization()
                        if util < min_util:
                            best = (cluster, node)
                            min_util = util
            if best:
                cluster, node = best
                if node.schedule(task):
                    env.queue.remove(task)
                    task.start_exec_time = env.timestep_counter
                    task.completion_time = env.timestep_counter + task.duration
                    actions.append(Action(task, cluster, node))

        env.timestep()
        return actions, rewards






class CompactScheduler(Scheduler):
    def __init__(self, environment):
        self.environment = environment
        self.label = "CompactScheduler"

    def schedule(self):
        actions = []
        rewards = 0
        env = self.environment
        tasks = list(env.queue)

        for task in tasks:
            best = None
            max_util = -1
            for cluster in env.clusters:
                for node in cluster.nodes:
                    if node.can_run(task):
                        util = node.utilization()
                        if util > max_util:
                            best = (cluster, node)
                            max_util = util
            if best:
                cluster, node = best
                if node.schedule(task):
                    env.queue.remove(task)
                    task.start_exec_time = env.timestep_counter
                    task.completion_time = env.timestep_counter + task.duration
                    actions.append(Action(task, cluster, node))

        env.timestep()
        return actions, rewards














class CNNModel(tf.keras.Model):
    """CNN Model for variable-sized input summaries."""
    def __init__(self, input_shape, output_shape, idx):
        super(CNNModel, self).__init__()
        self.idx = idx
        model_path = f'__cache__/model/dqn{idx}.keras'
        if os.path.isfile(model_path):
            self.model = load_model(model_path)
            self.compile_model()
        else:
            self.model = Sequential([
                Conv2D(16, (3,3), padding='same', activation='relu', input_shape=(None,None,1)),
                MaxPooling2D(), Dropout(0.2),
                Conv2D(32, (3,3), padding='same', activation='relu'),
                MaxPooling2D(), Dropout(0.2),
                GlobalAveragePooling2D(),
                Dense(256, activation='relu'),
                Dense(output_shape, activation='linear'),
            ])
            self.compile_model()

    def compile_model(self):
        self.model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.01), loss='mse')

    @tf.function(experimental_relax_shapes=True)
    def call(self, input_data):
        return self.model(input_data)

    def save(self):
        os.makedirs('__cache__/model', exist_ok=True)
        self.model.save(f'__cache__/model/dqn{self.idx}.keras')

class DQN(object):
    """DQN Implementation."""
    def __init__(self, input_shape, output_shape, idx):
        self.lr = 0.01
        self.gamma = 0.99
        self.batch_size = 32
        self.min_experiences = 100
        self.max_experiences = 10000
        self.optimizer = tf.optimizers.Adam(self.lr)
        self.num_actions = output_shape
        self.model = CNNModel(input_shape, output_shape, idx)
        self.experience = {'s': [], 'a': [], 'r': [], 's2': [], 'done': []}


        # --- FedProx additions ---
        self.prox_mu = 0.0  # 0 disables FedProx
        self._prox_center = None  # list[Tensor] with same shapes as trainable vars

    # Enable/disable FedProx
    def enable_prox(self, mu: float):
        self.prox_mu = float(mu)
    def set_prox_center_from_current(self):
        # Capture the *current* trainable weights as w^(t) for this round
        self._prox_center = [tf.identity(v) for v in self.model.trainable_variables]

    def predict(self, input_data):
        x = input_data.astype('float32').reshape(
            input_data.shape[0], input_data.shape[1], input_data.shape[2], 1)
        return self.model(x)

    @tf.function(experimental_relax_shapes=True)
    def train(self, dqn_target):
        if len(self.experience['s']) < self.min_experiences:
            return
        idxs = np.random.randint(0, len(self.experience['s']), self.batch_size)
        states = np.asarray([self.experience['s'][i] for i in idxs])
        actions = np.asarray([self.experience['a'][i] for i in idxs])
        rewards = np.asarray([self.experience['r'][i] for i in idxs])
        next_states = np.asarray([self.experience['s2'][i] for i in idxs])
        dones = np.asarray([self.experience['done'][i] for i in idxs])

        future_q = np.max(dqn_target.predict(next_states), axis=1)
        target_q = np.where(dones, rewards, rewards + self.gamma * future_q)

        with tf.GradientTape() as tape:
            pred_q = tf.reduce_sum(
                self.predict(states) * tf.one_hot(actions, self.num_actions), axis=1
            )
            base_loss = tf.reduce_sum(tf.square(target_q - pred_q))
            tf.debugging.assert_all_finite(base_loss, "base_loss has NaN/Inf")

            # ---------- FedProx penalty ----------
            prox_loss = tf.zeros((), dtype=base_loss.dtype)
            if self.prox_mu > 0.0 and self._prox_center is not None:
                diffs = []
                for w, w0 in zip(self.model.trainable_variables, self._prox_center):
                    diffs.append(tf.reduce_sum(tf.square(w - w0)))
                prox_loss = 0.5 * self.prox_mu * tf.add_n(diffs)
                tf.debugging.assert_all_finite(prox_loss, "prox_loss has NaN/Inf")

                # Debug print (optional)
                if tf.math.is_finite(prox_loss):
                    tf.print("FedProx penalty:", prox_loss)
            loss = base_loss + prox_loss
            # ------------------------------------

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

    def get_action(self, state, epsilon):
        if np.random.random() < epsilon:
            return np.random.choice(self.num_actions)
        return int(np.argmax(self.predict(np.array([state]))[0]))

    def add_experience(self, exp):
        if len(self.experience['s']) >= self.max_experiences:
            for k in self.experience:
                self.experience[k].pop(0)
        for k, v in exp.items():
            self.experience[k].append(v)

    def copy_weights(self, src):
        for tgt_var, src_var in zip(self.model.trainable_variables, src.model.trainable_variables):
            tgt_var.assign(src_var.numpy())

    def save_weights(self):
        self.model.save()






class DeepRMScheduler(Scheduler):
    """DeepRM scheduler with two-level dispatch."""
    def __init__(self, environment, idx):
        #self.idx = idx
        self.max_cluster_utilization = 0.85
        self.environment = environment
        self.episodes = 500
        self.copy_steps = 32
        self.save_steps = 32
        self.epsilon = 0.99
        self.decay = 0.99
        self.min_epsilon = 0.1

        inp = (*environment.summary().shape, 1)
        out = (environment.queue_size * len(environment.clusters)) + 1

        self.dqn_train = DQN(inp, out, idx)
        self.dqn_target = DQN(inp, out, idx)

        self.total_rewards = np.empty(self.episodes)
        self.task_slowdown = np.empty(self.episodes)
        self.cum_rewards = []
        self.slowdowns = []
        self.smoothed_rewards = []  # 🔹 Added for stability

    def train(self):
        smoothed_rewards = []  # store smoothed reward per episode
        per_episode_rewards = []  # full reward history for plotting

        for ep in range(self.episodes):
            self.epsilon = max(self.min_epsilon, self.epsilon * self.decay)

            self.environment.queue.clear()
            self.environment.backlog.clear()
            self.environment._task_index = 0
            self.environment.timestep_counter = 0
            self.environment.next_task_time = np.random.exponential(1.0 / self.environment.lambda_rate)

            for cluster in self.environment.clusters:
                cluster.cluster_queue.clear()
                for node in cluster.nodes:
                    node.__init__(node.resources, node.duration, node.label)

            self.environment.dispatch_fail_count = 0
            self.environment.node_place_fail_count = 0
            self.environment.dispatch_failed_ids = set()
            self.environment.node_failed_ids = set()
            self.environment.failures_this_step = {'dispatch': 0, 'node': 0, 'deadline': 0}
            self.environment.cumulative_penalty = 0.0

            reward, smoothed_ep = self.train_episode()
            self.total_rewards[ep] = reward
            self.task_slowdown[ep] = -reward
            self.cum_rewards.append(reward)
            self.slowdowns.append(self.task_slowdown[ep])
            smoothed_rewards.append(smoothed_ep)
            per_episode_rewards.append(reward)

            print(f"Episode {ep}, reward={reward:.2f}, smoothed={smoothed_ep:.2f}")

        self.smoothed_rewards = smoothed_rewards
        self.per_episode_rewards = per_episode_rewards

    def train_episode(self):
        env_obj = self.environment

        env_obj.queue.clear()
        env_obj.backlog.clear()
        env_obj._task_index = 0
        env_obj.timestep_counter = 0
        env_obj.next_task_time = np.random.exponential(1.0 / env_obj.lambda_rate)

        env_obj.dispatch_fail_count = 0
        env_obj.node_place_fail_count = 0
        env_obj.dispatch_failed_ids = set()
        env_obj.node_failed_ids = set()
        env_obj.failures_this_step = {'dispatch': 0, 'node': 0, 'deadline': 0}
        env_obj.cumulative_penalty = 0.0

        for cluster in env_obj.clusters:
            cluster.cluster_queue.clear()
            for node in cluster.nodes:
                node.__init__(node.resources, node.duration, node.label)

        if not env_obj.backlog and not env_obj.queue:
            t = env_obj.tasks[env_obj._task_index]
            env_obj._task_index += 1
            env_obj.backlog.append(t)

        total_reward = 0
        smoothed_reward = 0
        step = 0

        while not env_obj.terminated():
            state = env_obj.summary().astype('float32')[..., np.newaxis]
            ai = self.dqn_train.get_action(state, epsilon=self.epsilon)

            acts, rews = self.schedule(ai)
            reward = sum(rews)
            total_reward += reward
            smoothed_reward = 0.9 * smoothed_reward + 0.1 * reward

            next_state = env_obj.summary().astype('float32')[..., np.newaxis]
            done = env_obj.terminated()

            self.dqn_train.add_experience({
                's': state, 'a': ai, 'r': reward, 's2': next_state, 'done': done
            })
            self.dqn_train.train(self.dqn_target)

            step += 1
            if step % self.copy_steps == 0:
                self.dqn_target.copy_weights(self.dqn_train)
            if step % self.save_steps == 0:
                self.dqn_train.model.save()

        return total_reward, smoothed_reward

    def schedule(self, first_action=None):
        actions, rewards = [], []

        for cluster in self.environment.clusters:
            new_q = []
            for task in cluster.cluster_queue:
                node = cluster.schedule(task)
                if node:
                    actions.append(Action(task, node, cluster))
                    rewards.append(self.environment.reward())
                else:
                    new_q.append(task)
            cluster.cluster_queue = new_q

        self.environment.timestep()
        rewards.append(self.environment.reward())

        if first_action is not None:
            ai = first_action
        else:
            state = self.environment.summary().astype('float32')[..., np.newaxis]
            ai = self.dqn_train.get_action(state, epsilon=self.epsilon)

        ti, ci = self._explain(ai)
        if ti < 0 or ci < 0 or ti >= len(self.environment.queue) or ci >= len(self.environment.clusters):
            return actions, rewards

        cluster = self.environment.clusters[ci]

        if (len(cluster.cluster_queue) < cluster.queue_size and
                cluster.utilization() < self.max_cluster_utilization):
            task = self.environment.queue.pop(ti)
            task.enter_cluster_time = self.environment.timestep_counter
            cluster.cluster_queue.append(task)
            rewards.append(self.environment.reward())
        else:
            if not hasattr(self.environment, "blocked_dispatch_steps"):
                self.environment.blocked_dispatch_steps = 0
            self.environment.blocked_dispatch_steps += 1
            self.environment.timestep()
            rewards.append(self.environment.reward())

        return actions, rewards

    def _explain(self, action_index):
        tl = self.environment.queue_size
        nc = len(self.environment.clusters)
        if action_index == tl * nc:
            return -1, -1
        ti = action_index % tl
        ci = action_index // tl
        if ti >= len(self.environment.queue):
            return -1, -1
        return ti, ci

    def plot_metrics(self, agent_type):
        os.makedirs(f'__cache__/plots/{agent_type}', exist_ok=True)
        os.makedirs('__cache__/metrics', exist_ok=True)

        # 🔹 Plot 1: Raw Total Rewards vs Smoothed Rewards
        plt.figure(figsize=(10, 6))
        plt.plot(self.per_episode_rewards, label='Raw Reward', alpha=0.6)
        plt.plot(self.smoothed_rewards, label='Smoothed Reward', linewidth=2)
        plt.title(f'{agent_type} Reward per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.legend()
        plt.grid()
        plt.savefig(f'__cache__/plots/{agent_type}/reward_smoothed.png')

        # 🔹 Plot 2: Task Slowdown
        plt.figure(figsize=(10, 6))
        plt.plot(self.task_slowdown, label='Slowdown')
        plt.title(f'{agent_type} Task Slowdown')
        plt.xlabel('Episode')
        plt.ylabel('Slowdown')
        plt.legend()
        plt.grid()
        plt.savefig(f'__cache__/plots/{agent_type}/task_slowdown.png')

        # 🔹 Save all to CSV
        csv_path = f'__cache__/metrics/metrics_data_{agent_type}.csv'
        with open(csv_path, 'w') as f:
            f.write('Episode,Reward,SmoothedReward,Slowdown\n')
            for i in range(len(self.per_episode_rewards)):
                r = self.per_episode_rewards[i]
                sr = self.smoothed_rewards[i] if i < len(self.smoothed_rewards) else ''
                s = self.task_slowdown[i] if i < len(self.task_slowdown) else ''
                f.write(f"{i},{r},{sr},{s}\n")
