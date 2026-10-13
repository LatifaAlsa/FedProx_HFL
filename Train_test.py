import env
import pickle
import numpy as np
import pandas as pd
from collections import Counter
import os
from FL_AVG import *
from plot import (scheduling_plot, plot_waiting_time, plot_execution_time, plot_wait_vs_exec, plot_response_time, plot_throughput, plot_waiting_time_total, plot_makespan, plot_avg_exec_per_task)
import random
import tensorflow as tf

# 🔹 Make results reproducible
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)




def train_agent(agent, environment):
    agent.environment = environment
    agent.is_training = True
    agent.train()
    return agent


if __name__ == '__main__':
    print('Training and Testing Cycle Start...')

    test_environment = env.load_env()
    global_agent_environment = env.load_env()
    single_agent_environment = env.load_env()
    global_agent = env.load_scheduler(global_agent_environment, 4)
    single_agent = env.load_scheduler(single_agent_environment, 5)

    ##########################################################################
    #                         Training part                                  #
    ##########################################################################

    # Load the global agent parameters
    print('Loading global agent...')
    with open('__cache__/model/global_agent.pkl', 'rb') as f:
        global_agent_pram = pickle.load(f)
    global_agent = env.load_scheduler(global_agent_environment, 4)
    global_agent = copy_pram(global_agent, global_agent_pram)
    # Compile the global agent model after loading
    print('Compiling global agent...')
    global_agent.dqn_train.model.compile_model()
    global_agent.dqn_target.model.compile_model()
    # Train the global agent
    print('Training the global agent')
    global_agent.is_training = True  # Set training mode
    global_agent.train()  # Call the training method
    # persist your retrained global agent
    with open('__cache__/model/global_agent.pkl', 'wb') as f:
        pickle.dump([v.numpy() for v in global_agent.dqn_train.model.trainable_variables], f)
    # Plot cumulative rewards and task slowdown
    print('Plotting cumulative rewards and task slowdown...')
    global_agent.plot_metrics('global_agent')
    # 5) Evaluate via your existing test_agent()
    # score = test_agent(test_environment, global_agent)
    # print(f"Retrained Global Agent Test Score: {score}")

    ###################### Training single agent ############################
    # Load the parameters for the single agent
    print('Loading single agent...')
    print('[Init] Single agent created fresh (no checkpoint reload).')

    single_agent = env.load_scheduler(single_agent_environment, 5)
    # Compile the single agent model after loading
    print('Compiling single agent...')
    single_agent.dqn_train.model.compile_model()
    single_agent.dqn_target.model.compile_model()
    # Train the single DQN
    print('Training the single DQN')
    single_agent.is_training = True  # Set training mode
    single_agent.train()  # Call the training method
    # Plot cumulative rewards and task slowdown
    print('Plotting cumulative rewards and task slowdown...')
    single_agent.plot_metrics('single_agent')

    ############################################################################
    #                              Scheduling Stage                            #
    ############################################################################

    # 1) Create fresh envs for scheduling runs
    sched_global_env = env.load_env()
    sched_single_env = env.load_env()

    # 2) Attach trained agents to those envs
    global_agent.environment = sched_global_env
    single_agent.environment = sched_single_env

    # 3) (Re-)compile if needed
    global_agent.dqn_train.model.compile_model()
    single_agent.dqn_train.model.compile_model()

    # 4) Initialize trackers
    backlog_wait_global = queue_wait_global = cluster_queue_wait_global = 0
    backlog_wait_single = queue_wait_single = cluster_queue_wait_single = 0

    # --- Global agent scheduling run ---
    print("Scheduling with global agent…")
    global_actions = []
    while not sched_global_env.terminated():
        global_agent.epsilon = 0.1
        acts, _ = global_agent.schedule()
        global_actions.extend(acts)

        # debug print
        print(f"\nTimestep {sched_global_env.timestep_counter}: "
              f"backlog={len(sched_global_env.backlog)}, "
              f"queue={len(sched_global_env.queue)}")
        for c in sched_global_env.clusters:
            print(f"Cluster {c.label} queue size = {len(c.cluster_queue)}")
        print("Actions taken:" if acts else "No scheduler actions!", acts)

        # accumulate waits
        backlog_wait_global       += len(sched_global_env.backlog)
        queue_wait_global         += len(sched_global_env.queue)
        for c in sched_global_env.clusters:
            cluster_queue_wait_global += len(c.cluster_queue)

    # 5) Dump global actions
    os.makedirs('__cache__/arrays', exist_ok=True)
    pd.DataFrame([{
        'cluster': a.cluster.label,
        'node':    a.node.label,
        'task':    a.task.label
    } for a in global_actions])\
      .to_csv('__cache__/arrays/global_actions.csv', index=False)


    # --- Single agent scheduling run ---
    print("\nScheduling with single agent…")
    single_actions = []
    while not sched_single_env.terminated():
        single_agent.epsilon = 0.1
        acts, _ = single_agent.schedule()
        single_actions.extend(acts)

        # debug print
        print(f"\nTimestep {sched_single_env.timestep_counter}: "
              f"backlog={len(sched_single_env.backlog)}, "
              f"queue={len(sched_single_env.queue)}")
        for c in sched_single_env.clusters:
            print(f"Cluster {c.label} queue size = {len(c.cluster_queue)}")
        print("Actions taken:" if acts else "No scheduler actions!", acts)

        # accumulate waits
        backlog_wait_single       += len(sched_single_env.backlog)
        queue_wait_single         += len(sched_single_env.queue)
        for c in sched_single_env.clusters:
            cluster_queue_wait_single += len(c.cluster_queue)

    # 6) Dump single actions
    pd.DataFrame([{
        'cluster': a.cluster.label,
        'node':    a.node.label,
        'task':    a.task.label
    } for a in single_actions])\
      .to_csv('__cache__/arrays/single_actions.csv', index=False)

    # 7) Build per-node counts & save
    global_counts = Counter(a.node.label for a in global_actions)
    single_counts = Counter(a.node.label for a in single_actions)
    rows = []
    for c in sched_global_env.clusters:
        for n in c.nodes:
            rows.append({
                'cluster':       c.label,
                'node':          n.label,
                'global_count':  global_counts.get(n.label, 0),
                'single_count':  single_counts.get(n.label, 0),
            })
    pd.DataFrame(rows)\
      .to_csv('__cache__/arrays/scheduling_counts.csv', index=False)

    # 8) Execution & waiting time metrics
    exec_time_global = sched_global_env.timestep_counter
    exec_time_single = sched_single_env.timestep_counter

    metrics = [
        ('backlog_wait', backlog_wait_global, backlog_wait_single),
        ('queue_wait', queue_wait_global, queue_wait_single),
        ('cluster_queue_wait', cluster_queue_wait_global, cluster_queue_wait_single),
        ('execution_time', exec_time_global, exec_time_single),
    ]
    pd.DataFrame(metrics, columns=['metric', 'global', 'single']) \
        .to_csv('__cache__/arrays/wait_and_exec_metrics.csv', index=False)

    # after your scheduling loops in Train_test.py
    tasks_global = len(global_actions)
    tasks_single = len(single_actions)

    avg_exec_global = exec_time_global / tasks_global  # ms per task
    avg_exec_single = exec_time_single / tasks_single  # ms per task







    ############################################################################
    #                              Plotting Stage                            #
    ############################################################################
    # 9) Combined summary & plot
    plot_wait_vs_exec()
    # 10) Individual waiting & execution plots
    plot_waiting_time(
        backlog_wait_global, backlog_wait_single,
        queue_wait_global, queue_wait_single,
        cluster_queue_wait_global, cluster_queue_wait_single,
        exec_time_global, exec_time_single
    )
    # 11) Compute & plot
    plot_execution_time(exec_time_global, exec_time_single)
    # 12) Average response time per task
    plot_response_time()
    # 13) Compute & plot throughput
    plot_throughput()
    # 14) plot scheduling actions
    scheduling_plot(global_actions, single_actions, sched_global_env.clusters)
    # 15) Compute & plot Makspan
    plot_makespan()
    # 16) Compute & plot
    plot_waiting_time_total(
        backlog_wait_global, backlog_wait_single,
        queue_wait_global, queue_wait_single,
        cluster_queue_wait_global, cluster_queue_wait_single
    )
    # 17) Compute & plot Average execution time per task
    plot_avg_exec_per_task()


print("END of Scheduling Stage")


