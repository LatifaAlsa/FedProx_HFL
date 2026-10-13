import os
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd
import numpy as np


def scores_plot(steps,
                scores_global_total,
                scores_single_total,
                scores_global_per_task,
                scores_single_per_task,
                scores_global_per_time,
                scores_single_per_time):
    """Plots Global vs Single Agent's score metrics over training rounds."""
    import os, matplotlib.pyplot as plt

    os.makedirs('__cache__/plots', exist_ok=True)

    # ------------------------------
    # 1. Plot total raw scores
    plt.figure(figsize=(12, 6))
    plt.plot(steps, scores_global_total, linestyle='-', marker='o', label='Global Agent')
    plt.plot(steps, scores_single_total, linestyle='--', marker='s', label='Single Agent')
    plt.xlabel('Rounds')
    plt.ylabel('Total Score')
    plt.title('Total Score: Global vs Single Agent')
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig('__cache__/plots/Scores_Total.png')
    plt.close()

    # ------------------------------
    # 2. Plot normalized score per task
    plt.figure(figsize=(12, 6))
    plt.plot(steps, scores_global_per_task, linestyle='-', marker='o', label='Global Agent')
    plt.plot(steps, scores_single_per_task, linestyle='--', marker='s', label='Single Agent')
    plt.xlabel('Rounds')
    plt.ylabel('Score per Task')
    plt.title('Normalized Score per Task')
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig('__cache__/plots/Scores_Per_Task.png')
    plt.close()

    # ------------------------------
    # 3. Plot normalized score per time
    plt.figure(figsize=(12, 6))
    plt.plot(steps, scores_global_per_time, linestyle='-', marker='o', label='Global Agent')
    plt.plot(steps, scores_single_per_time, linestyle='--', marker='s', label='Single Agent')
    plt.xlabel('Rounds')
    plt.ylabel('Score per Timestep')
    plt.title('Normalized Score per Timestep')
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig('__cache__/plots/Scores_Per_Time.png')
    plt.close()



def scheduling_plot(global_actions, single_actions, clusters):
    """Plots the number of tasks scheduled per node, grouped by cluster."""
    if not os.path.exists('__cache__/plots'):
        os.makedirs('__cache__/plots')

    # Count tasks per node
    global_counts = Counter(act.node.label for act in global_actions)
    single_counts = Counter(act.node.label for act in single_actions)

    # Prepare labels in cluster order
    labels = []
    global_vals = []
    single_vals = []
    for cluster in clusters:
        for node in cluster.nodes:
            labels.append(f"{cluster.label}.{node.label}")
            global_vals.append(global_counts.get(node.label, 0))
            single_vals.append(single_counts.get(node.label, 0))

    x = range(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x, global_vals, width, label='Global Agent')
    ax.bar([i + width for i in x], single_vals, width, label='Single Agent')

    ax.set_xticks([i + width/2 for i in x])
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_xlabel('Cluster.Node')
    ax.set_ylabel('Scheduled Task Count')
    ax.set_title('Scheduled Tasks per Node')
    ax.legend()
    plt.tight_layout()
    plt.savefig('__cache__/plots/Scheduling_comparison.png')
    plt.close()


def plot_waiting_time(backlog_wait_global, backlog_wait_single,
                      queue_wait_global, queue_wait_single,
                      cluster_queue_wait_global, cluster_queue_wait_single,
                      exec_time_global, exec_time_single):
    """
    Plots average waiting time per scheduling step for:
      - backlog queue
      - ready queue
      - cluster queues
    for both Global and Single agents.
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    os.makedirs('__cache__/plots', exist_ok=True)

    # Compute avg wait per step (1 step = 1 ms)
    avg_backlog_global = backlog_wait_global       / exec_time_global
    avg_queue_global   = queue_wait_global         / exec_time_global
    avg_cluster_global = cluster_queue_wait_global / exec_time_global

    avg_backlog_single = backlog_wait_single       / exec_time_single
    avg_queue_single   = queue_wait_single         / exec_time_single
    avg_cluster_single = cluster_queue_wait_single / exec_time_single

    # Prepare data for plotting
    labels = ['Backlog', 'Ready Queue', 'Cluster Queues']
    global_times = [avg_backlog_global, avg_queue_global, avg_cluster_global]
    single_times = [avg_backlog_single, avg_queue_single, avg_cluster_single]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width/2, global_times, width, label='Global Agent')
    ax.bar(x + width/2, single_times, width, label='Single Agent')

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel('Queue Type')
    ax.set_ylabel('Avg Wait per Step (ms)')
    ax.set_title('Average Waiting Time per Step Comparison')
    ax.legend()
    plt.tight_layout()
    plt.savefig('__cache__/plots/Avg_Wait_per_Step.png')
    plt.close()



def plot_execution_time(exec_time_global, exec_time_single):
    """Plots the total execution time for both agents."""
    if not os.path.exists('__cache__/plots'):
        os.makedirs('__cache__/plots')

    labels = ['Global Agent', 'Single Agent']
    times = [exec_time_global, exec_time_single]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(labels, times)
    ax.set_xlabel('Agent Type')
    ax.set_ylabel('Execution Time')
    ax.set_title('Total Execution Time Comparison')
    plt.tight_layout()
    plt.savefig('__cache__/plots/Execution_Time.png')
    plt.close()


def plot_round_metrics(rounds,
                       # Global metrics
                       backlog_waits, queue_waits, cluster_queue_waits, total_tasks,
                       exec_times_global,
                       # Single-agent metrics
                       backlog_waits_single, queue_waits_single, cluster_queue_waits_single, total_tasks_single,
                       exec_times_single=None):
    """
    Plot per-round comparison of:
      1) Average waiting time per task (global vs single)
      2) Execution time (global vs single)
    And save combined metrics to CSV.
    """
    if not os.path.exists('__cache__/plots'):
        os.makedirs('__cache__/plots')

    # --- Compute average waits per task ---
    avg_waits_global = [
        (bw + qw + cw) / t
        for bw, qw, cw, t in zip(backlog_waits, queue_waits, cluster_queue_waits, total_tasks)
    ]
    avg_waits_single = [
        (bw + qw + cw) / t
        for bw, qw, cw, t in zip(backlog_waits_single, queue_waits_single, cluster_queue_waits_single, total_tasks_single)
    ]

    # --- Save to CSV ---
    df = pd.DataFrame({
        'round': rounds,
        'avg_wait_global_ms': avg_waits_global,
        'avg_wait_single_ms': avg_waits_single,
        'exec_time_global_ms': exec_times_global,
        'exec_time_single_ms': exec_times_single or [None]*len(rounds)
    })
    df.to_csv('__cache__/arrays/round_metrics.csv', index=False)

    # --- Average Waiting Time per Round ---
    plt.figure(figsize=(10, 6))
    plt.plot(rounds, avg_waits_global, marker='o', linestyle='-',  label='Global Avg Wait')
    plt.plot(rounds, avg_waits_single, marker='s', linestyle='--', label='Single-Agent Avg Wait')
    plt.xlabel('Federated Round')
    plt.ylabel('Average Waiting Time (ms)')
    plt.title('Average Waiting Time per Round')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('__cache__/plots/Avg_Wait_per_Round.png')
    plt.close()

    # --- Execution Time per Round ---
    plt.figure(figsize=(10, 6))
    plt.plot(rounds, exec_times_global, marker='^', linestyle='-',  label='Global Exec Time')
    if exec_times_single is not None:
        plt.plot(rounds, exec_times_single, marker='v', linestyle='--', label='Single-Agent Exec Time')
    plt.xlabel('Federated Round')
    plt.ylabel('Execution Time (ms)')
    plt.title('Execution Time per Round')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('__cache__/plots/Exec_Time_per_Round.png')
    plt.close()



def plot_wait_vs_exec(metrics_csv='__cache__/arrays/wait_and_exec_metrics.csv',
                      summary_csv='__cache__/arrays/wait_exec_summary.csv',
                      plot_path='__cache__/plots/wait_vs_exec.png'):
    """Reads raw wait/exec metrics, computes average waiting time per step, saves summary CSV, and plots bar chart."""
    # ensure directories exist
    if not os.path.exists(os.path.dirname(summary_csv)):
        os.makedirs(os.path.dirname(summary_csv))
    if not os.path.exists(os.path.dirname(plot_path)):
        os.makedirs(os.path.dirname(plot_path))

    # read existing metrics
    df = pd.read_csv(metrics_csv)

    # sum waiting metrics (all except execution_time)
    wait_df = df[df['metric'] != 'execution_time']
    total_wait_global = wait_df['global'].sum()
    total_wait_single = wait_df['single'].sum()

    # get execution times
    exec_global = df.loc[df['metric'] == 'execution_time', 'global'].iloc[0]
    exec_single = df.loc[df['metric'] == 'execution_time', 'single'].iloc[0]

    # compute average wait per step (since 1 step = 1 ms)
    avg_wait_global = total_wait_global / exec_global
    avg_wait_single = total_wait_single / exec_single

    # save summary CSV
    summary = pd.DataFrame({
        'metric': ['avg_wait_per_step', 'execution_time'],
        'global_agent': [avg_wait_global, exec_global],
        'single_agent': [avg_wait_single, exec_single]
    })
    summary.to_csv(summary_csv, index=False)

    # plot
    labels = ['Global', 'Single']
    avg_waits = [avg_wait_global, avg_wait_single]
    execution = [exec_global, exec_single]
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(x - width/2, avg_waits, width, label='Avg Waiting Time (ms per step)')
    ax.bar(x + width/2, execution, width, label='Execution Time (ms)')

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Time (ms)')
    ax.set_title('Scheduling: Avg Wait per Step vs Execution Time')
    ax.legend()
    plt.tight_layout()
    fig.savefig(plot_path)
    plt.close()



def plot_response_time(
        delays_csv='__cache__/arrays/average_task_delays.csv',
        global_actions_csv='__cache__/arrays/global_actions.csv',
        single_actions_csv='__cache__/arrays/single_actions.csv',
        plot_path='__cache__/plots/response_time.png'):
    """
    Compute true average response time per task using timestamps:
        R_i = completion_time - arrival_time
    and plot comparison between global and single agents.
    """
    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np

    os.makedirs(os.path.dirname(plot_path), exist_ok=True)

    # Try loading timestamped task data if available
    # (for now we rely on per-task logs created in test.py)
    if not os.path.exists(delays_csv):
        print("[Warning] average_task_delays.csv not found — cannot plot response time.")
        return

    # Load per-task delays first (for reference)
    df_delay = pd.read_csv(delays_csv)

    # Try reading response timestamps if created in test.py
    resp_path = '__cache__/arrays/average_task_responses.csv'
    if os.path.exists(resp_path):
        df_resp = pd.read_csv(resp_path)
        global_responses = df_resp['global_response'].dropna()
        single_responses = df_resp['single_response'].dropna()
        resp_global_mean = np.nanmean(global_responses)
        resp_single_mean = np.nanmean(single_responses)
    else:
        # Fallback: approximate if response CSV not available
        print("[Info] Using fallback approximate response time calculation.")
        resp_global_mean = np.nanmean(df_delay['global_delay'])
        resp_single_mean = np.nanmean(df_delay['single_delay'])

    # Plot bar chart
    labels = ['Global Agent', 'Single Agent']
    values = [resp_global_mean, resp_single_mean]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(labels, values, color=['tab:blue', 'tab:orange'])
    ax.set_ylabel('Average Response Time (timesteps per task)')
    ax.set_title('True Average Response Time per Task')
    for i, v in enumerate(values):
        ax.text(i, v + 0.5, f"{v:.2f}", ha='center')
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    # Save numeric summary
    pd.DataFrame({
        'metric': ['avg_response_time_true'],
        'global_agent': [resp_global_mean],
        'single_agent': [resp_single_mean]
    }).to_csv('__cache__/arrays/response_time_summary_true.csv', index=False)

    print(f"Average Response Time (Global Agent): {resp_global_mean:.2f}")
    print(f"Average Response Time (Single Agent): {resp_single_mean:.2f}")


def plot_throughput(global_actions_csv='__cache__/arrays/global_actions.csv',
                    single_actions_csv='__cache__/arrays/single_actions.csv',
                    exec_csv='__cache__/arrays/wait_and_exec_metrics.csv',
                    plot_csv='__cache__/arrays/throughput_summary.csv',
                    plot_path='__cache__/plots/throughput.png'):
    """Plots throughput (#tasks/s), saves summary CSV."""
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)

    g_tasks=len(pd.read_csv(global_actions_csv))
    s_tasks=len(pd.read_csv(single_actions_csv))
    df=pd.read_csv(exec_csv)
    g_makespan=df.loc[df.metric=='execution_time','global'].iloc[0]
    s_makespan=df.loc[df.metric=='execution_time','single'].iloc[0]

    tp_global=g_tasks/(g_makespan/1000.0)
    tp_single=s_tasks/(s_makespan/1000.0)

    summary=pd.DataFrame({
        'metric':['throughput_tasks_per_s'],
        'global_agent':[tp_global],
        'single_agent':[tp_single]
    })
    summary.to_csv(plot_csv,index=False)

    labels=['Global','Single']; values=[tp_global,tp_single]
    fig,ax=plt.subplots(figsize=(8,6))
    ax.bar(labels,values)
    ax.set_ylabel('Throughput (tasks/s)');ax.set_title('Throughput Comparison')
    plt.tight_layout();fig.savefig(plot_path);plt.close()


def plot_makespan(metrics_csv='__cache__/arrays/wait_and_exec_metrics.csv',
                  plot_csv='__cache__/arrays/makespan_summary.csv',
                  plot_path='__cache__/plots/makespan.png'):
    """Plot makespan comparison, saves summary CSV."""
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    df=pd.read_csv(metrics_csv)
    makespan_global=df.loc[df.metric=='execution_time','global'].iloc[0]
    makespan_single=df.loc[df.metric=='execution_time','single'].iloc[0]

    summary=pd.DataFrame({
        'metric':['makespan_ms'],
        'global_agent':[makespan_global],
        'single_agent':[makespan_single]
    })
    summary.to_csv(plot_csv,index=False)

    labels=['Global','Single']; values=[makespan_global,makespan_single]
    fig,ax=plt.subplots(figsize=(8,6))
    ax.bar(labels,values)
    ax.set_ylabel('Makespan (ms)');ax.set_title('Makespan Comparison')
    plt.tight_layout();fig.savefig(plot_path);plt.close()

def plot_avg_exec_per_task(global_actions_csv='__cache__/arrays/global_actions.csv',
                           single_actions_csv='__cache__/arrays/single_actions.csv',
                           exec_csv='__cache__/arrays/wait_and_exec_metrics.csv',
                           plot_path='__cache__/plots/Avg_Exec_per_Task.png',
                           csv_path='__cache__/arrays/Avg_Exec_per_Task.csv'):
    """
    Computes average execution time per task (makespan / #tasks) for Global vs Single,
    saves a bar chart and a CSV file with the values.
    """
    import os, pandas as pd, matplotlib.pyplot as plt, numpy as np

    # Ensure output directories exist
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    # 1) Load #tasks completed
    g_tasks = len(pd.read_csv(global_actions_csv))
    s_tasks = len(pd.read_csv(single_actions_csv))

    # 2) Load makespan (execution_time in ms)
    df = pd.read_csv(exec_csv)
    g_makespan = df.loc[df.metric == 'execution_time', 'global'].iloc[0]
    s_makespan = df.loc[df.metric == 'execution_time', 'single'].iloc[0]

    # 3) Compute avg exec time per task
    avg_exec_global = g_makespan / g_tasks if g_tasks > 0 else np.nan
    avg_exec_single = s_makespan / s_tasks if s_tasks > 0 else np.nan

    # 4) Save to CSV
    out_df = pd.DataFrame({
        'agent': ['global', 'single'],
        'avg_exec_per_task_ms': [avg_exec_global, avg_exec_single]
    })
    out_df.to_csv(csv_path, index=False)

    # 5) Plot
    labels = ['Global', 'Single']
    values = [avg_exec_global, avg_exec_single]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, values)
    ax.set_ylabel('Avg Execution Time (ms per task)')
    ax.set_title('Average Execution Time per Task')
    plt.tight_layout()
    fig.savefig(plot_path)
    plt.close()




def plot_waiting_time_total(backlog_wait_global, backlog_wait_single,
                      queue_wait_global, queue_wait_single,
                      cluster_queue_wait_global, cluster_queue_wait_single):
    """Plots waiting times in backlog, queue, and cluster queue for both agents."""
    if not os.path.exists('__cache__/plots'):
        os.makedirs('__cache__/plots')

    labels = ['Backlog Wait', 'Queue Wait', 'Cluster-Queue Wait']
    global_times = [backlog_wait_global, queue_wait_global, cluster_queue_wait_global]
    single_times = [backlog_wait_single, queue_wait_single, cluster_queue_wait_single]

    x = range(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x, global_times, width, label='Global Agent')
    ax.bar([i + width for i in x], single_times, width, label='Single Agent')

    ax.set_xticks([i + width/2 for i in x])
    ax.set_xticklabels(labels)
    ax.set_xlabel('Waiting Metric')
    ax.set_ylabel('Total Wait Time')
    ax.set_title('Waiting Time Comparison')
    ax.legend()
    plt.tight_layout()
    plt.savefig('__cache__/plots/Waiting_Time.png')
    plt.close()


def plot_failure_rates(csv_path="__cache__/arrays/failure_types_all_schedulers.csv"):
    import pandas as pd, matplotlib.pyplot as plt, os
    os.makedirs("__cache__/plots", exist_ok=True)

    # Load failure summary
    df = pd.read_csv(csv_path)

    if "scheduler" not in df.columns:
        print("[Error] Expected 'scheduler' column not found in failure CSV.")
        print("CSV columns:", df.columns.tolist())
        return

    schedulers = df["scheduler"]
    dispatch = df["dispatch_fail_rate_percent"]
    node = df["node_place_fail_rate_percent"]
    deadline = df["deadline_fail_rate_percent"]

    # --- Bar Plot (side-by-side) ---
    x = range(len(schedulers))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar([i - width for i in x], dispatch, width, label="Dispatch Failures")
    ax.bar(x, node, width, label="Node Placement Failures")
    ax.bar([i + width for i in x], deadline, width, label="Deadline Failures")

    ax.set_xticks(x)
    ax.set_xticklabels(schedulers, rotation=20, ha="right")
    ax.set_ylabel("Failure Rate (%)")
    ax.set_title("Failure Rate Comparison Across All Schedulers")
    ax.legend()

    plt.tight_layout()
    plt.savefig("__cache__/plots/failure_rates_all_schedulers.png")
    plt.close()
    print("[Plot Saved] __cache__/plots/failure_rates_all_schedulers.png")



def plot_average_task_delay():
    import pandas as pd, matplotlib.pyplot as plt, numpy as np, os
    df = pd.read_csv('__cache__/arrays/average_task_delays.csv')

    avg_global = np.nanmean(df['global_delay'])
    avg_single = np.nanmean(df['single_delay'])

    print(f"Average Delay (Global Agent): {avg_global:.2f}")
    print(f"Average Delay (Single Agent): {avg_single:.2f}")

    df['global_smooth'] = df['global_delay'].rolling(window=10).mean()
    df['single_smooth'] = df['single_delay'].rolling(window=10).mean()

    os.makedirs('__cache__/plots', exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(df['global_delay'], alpha=0.3, label='Global Agent (Raw)')
    plt.plot(df['single_delay'], alpha=0.3, label='Single Agent (Raw)')
    plt.plot(df['global_smooth'], label='Global Agent (Smoothed)')
    plt.plot(df['single_smooth'], label='Single Agent (Smoothed)')
    plt.xlabel('Task Index')
    plt.ylabel('Delay (timesteps)')
    plt.title('Average Task Delay Comparison')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('__cache__/plots/average_task_delay.png')
    plt.close()


def plot_failure_rates_per_round(csv_path="__cache__/arrays/failure_rates_per_round.csv",
                                 plot_path="__cache__/plots/failure_rates_per_round.png"):
    import pandas as pd, matplotlib.pyplot as plt, os

    if not os.path.exists(csv_path):
        print("[plot_failure_rates_per_round] Missing failure CSV. Skipping plot.")
        return

    df = pd.read_csv(csv_path)
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.plot(df['round'], df['total_fail_rate_global'], label='Global Agent', marker='o')
    plt.plot(df['round'], df['total_fail_rate_single'], label='Single Agent', marker='s', linestyle='--')
    plt.xlabel("Federated Round")
    plt.ylabel("Failure Rate (%)")
    plt.title("Task Failure Rate Per Round")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    print(f"[✓] Failure rate per round plot saved → {plot_path}")


def plot_failure_types_stacked(csv_path="__cache__/arrays/failure_rates_per_round.csv",
                               save_path="__cache__/plots/failure_types_stacked.png"):
    import pandas as pd
    import matplotlib.pyplot as plt
    import os

    if not os.path.exists(csv_path):
        print("[plot_failure_types_stacked] CSV not found. Skipping.")
        return

    df = pd.read_csv(csv_path)
    rounds = df['round']

    # Global agent
    gd = df['dispatch_fail_global']
    gn = df['node_fail_global']
    gl = df.get('deadline_fail_global', pd.Series([0] * len(df)))

    # Single agent
    sd = df['dispatch_fail_single']
    sn = df['node_fail_single']
    sl = df.get('deadline_fail_single', pd.Series([0] * len(df)))

    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    axs[0].bar(rounds, gd, label='Dispatch', color='skyblue')
    axs[0].bar(rounds, gn, bottom=gd, label='Node', color='orange')
    axs[0].bar(rounds, gl, bottom=gd+gn, label='Deadline', color='red')
    axs[0].set_title("Global Agent Failures")
    axs[0].legend()

    axs[1].bar(rounds, sd, label='Dispatch', color='skyblue')
    axs[1].bar(rounds, sn, bottom=sd, label='Node', color='orange')
    axs[1].bar(rounds, sl, bottom=sd+sn, label='Deadline', color='red')
    axs[1].set_title("Single Agent Failures")
    axs[1].legend()
    axs[1].set_xlabel("Round")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[✓] Saved stacked failure plot → {save_path}")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[✓] Saved stacked failure type plot → {save_path}")


def plot_failure_rates_bar(csv_path="__cache__/arrays/timeout_failures.csv",
                           save_path="__cache__/plots/failure_breakdown_bar.png"):
    import pandas as pd
    import matplotlib.pyplot as plt
    import os

    if not os.path.exists(csv_path):
        print("[plot_failure_rates_bar] CSV not found. Skipping plot.")
        return

    df = pd.read_csv(csv_path)

    # Read values from latest CSV (single row expected)
    if df.empty:
        print("[plot_failure_rates_bar] CSV is empty. Skipping plot.")
        return

    failures = ["dispatch_fail", "node_place_fail", "deadline_fail"]
    global_vals = [df.loc[df['metric'] == f"{ft}_count", "global_agent"].values[0] for ft in failures]
    single_vals = [df.loc[df['metric'] == f"{ft}_count", "single_agent"].values[0] for ft in failures]

    labels = ["Dispatch", "Node", "Deadline"]
    x = range(len(labels))

    plt.figure(figsize=(8, 5))
    bar_w = 0.35

    plt.bar([i - bar_w/2 for i in x], global_vals, width=bar_w, label="Global Agent")
    plt.bar([i + bar_w/2 for i in x], single_vals, width=bar_w, label="Single Agent")

    plt.xticks(x, labels)
    plt.ylabel("Failure Count")
    plt.title("Failure Type Breakdown")
    plt.legend()
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

    print(f"[✓] Saved failure breakdown bar plot → {save_path}")


def plot_failure_rates_stacked_all_schedulers(csv_path="__cache__/arrays/failure_types_all_schedulers.csv",
                                              save_path="__cache__/plots/failure_rates_stacked_all_schedulers.png"):
    import pandas as pd
    import matplotlib.pyplot as plt
    import os

    if not os.path.exists(csv_path):
        print("[plot_failure_rates_stacked_all_schedulers] CSV not found.")
        return

    df = pd.read_csv(csv_path)

    schedulers = df["scheduler"]
    dispatch = df["dispatch_fail_rate_percent"]
    node = df["node_place_fail_rate_percent"]
    deadline = df["deadline_fail_rate_percent"]

    x = range(len(schedulers))

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot stacked bars
    bar_dispatch = ax.bar(x, dispatch, label='Dispatch Failures', color='skyblue')
    bar_node = ax.bar(x, node, bottom=dispatch, label='Node Failures', color='orange')
    bar_deadline = ax.bar(x, deadline, bottom=dispatch + node, label='Deadline Failures', color='red')

    # Annotate total failure rate percentages on top of each stack
    for i, (d, n, l) in enumerate(zip(dispatch, node, deadline)):
        total = d + n + l
        ax.text(i, total + 0.5, f"{total:.1f}%", ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Styling
    ax.set_xticks(x)
    ax.set_xticklabels(schedulers, rotation=20, ha='right')
    ax.set_ylabel("Failure Rate (%)")
    ax.set_title("Stacked Failure Rates Across All Schedulers")
    ax.legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"[✓] Saved stacked bar plot with labels → {save_path}")




def plot_penalty_per_round(csv_path="__cache__/arrays/penalty_per_round.csv",
                           save_path="__cache__/plots/penalty_per_round.png"):
    import pandas as pd
    import matplotlib.pyplot as plt
    if not os.path.exists(csv_path):
        print("[Plot] No penalty_per_round.csv found.")
        return
    df = pd.read_csv(csv_path)
    plt.figure(figsize=(6,4))
    plt.plot(df["round"], df["avg_penalty"], marker="o", label="Average penalty")
    plt.title("Average Penalty per Round")
    plt.xlabel("Round")
    plt.ylabel("Penalty (negative reward)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"[Plot Saved] → {save_path}")




def plot_node_utilization(csv_path="__cache__/arrays/node_utilization.csv",
                          save_path="__cache__/plots/node_utilization.png"):
    import pandas as pd, matplotlib.pyplot as plt, seaborn as sns, os
    if not os.path.exists(csv_path):
        print("[Plot] No node_utilization.csv found.")
        return
    df = pd.read_csv(csv_path)
    plt.figure(figsize=(10,6))
    sns.boxplot(x="scheduler", y="utilization", data=df)
    plt.ylabel("Node Utilization")
    plt.title("Per-Node Utilization across Schedulers")
    plt.grid(True)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"[✓] Saved node utilization plot → {save_path}")
    plt.close()


def plot_cluster_utilization(csv_path="__cache__/arrays/cluster_utilization.csv",
                             save_path="__cache__/plots/cluster_utilization.png"):
    import pandas as pd, matplotlib.pyplot as plt, seaborn as sns, os
    if not os.path.exists(csv_path):
        print("[Plot] No cluster_utilization.csv found.")
        return
    df = pd.read_csv(csv_path)
    plt.figure(figsize=(8,5))
    sns.barplot(x="cluster", y="utilization", hue="scheduler", data=df)
    plt.ylabel("Cluster Utilization")
    plt.title("Average Cluster Utilization by Scheduler")
    plt.legend()
    plt.grid(True)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"[✓] Saved cluster utilization plot → {save_path}")
    plt.close()

