# __main__.py
import os
import pickle
import numpy as np
import pandas as pd
import env
import random
from plot import scores_plot, plot_round_metrics, plot_failure_rates_per_round, plot_failure_types_stacked, plot_penalty_per_round
from FL_AVG import *
import tensorflow as tf

# 0) Timestep → real‐time mapping (1 timestep = 1 ms)
TIME_STEP_MS = 1.0
USE_FEDPROX = True     # set False to run plain FedAvg (baseline)
FEDPROX_MU  = 0.5    # try 1e-4 … 1e-2; larger = stronger pull to global (mu from {0.001, 0.01, 0.1, 0.5, 1})
# 🔹 Make results reproducible
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

def test_agent(make_env, agent, test_run=3, base_seed=SEED):
    """
    Evaluate an agent across `test_run` episodes on fresh environments.

    Returns:
        avg_total: average total reward across episodes
        avg_task:  average reward per scheduled task
        avg_time:  average reward per timestep
    """
    agent.is_training = False
    agent.epsilon = 0.1   # greedy policy during testing

    total_scores, scores_per_task, scores_per_time = [], [], []

    for run in range(test_run):
        # create a fresh environment each run (heterogeneous per seed)
        env_run = make_env(regenerate_tasks=True, seed=base_seed + run)
        agent.environment = env_run

        total_reward = 0
        tasks_scheduled = 0

        while not env_run.terminated():
            acts, rewards = agent.schedule()
            if isinstance(rewards, list):
                total_reward += sum(rewards)
            else:
                total_reward += rewards
            tasks_scheduled += len(acts)

        exec_time = env_run.timestep_counter
        norm_task = total_reward / tasks_scheduled if tasks_scheduled > 0 else 0
        norm_time = total_reward / exec_time if exec_time > 0 else 0

        total_scores.append(total_reward)
        scores_per_task.append(norm_task)
        scores_per_time.append(norm_time)

        print(f"[Test Run {run+1}] "
              f"Total={total_reward:.2f}, Tasks={tasks_scheduled}, "
              f"PerTask={norm_task:.4f}, PerTime={norm_time:.4f}")

    # --- Compute averages across test runs ---
    avg_total = sum(total_scores) / len(total_scores)
    avg_task  = sum(scores_per_task) / len(scores_per_task)
    avg_time  = sum(scores_per_time) / len(scores_per_time)

    return avg_total, avg_task, avg_time


def train_agent(agent, environment):
    agent.environment = environment
    agent.is_training = True
    agent.train()
    return agent


if __name__ == '__main__':
    print('Training and Testing Cycle Start...')

    ### Configuration
    number_of_agents = 4
    additional_rounds = 2  # how many new rounds to add

    ### Instantiate test & scheduler environments
    test_env = env.load_env(regenerate_tasks=True)
    global_env = env.load_env(regenerate_tasks=True)
    single_env = env.load_env(regenerate_tasks=True)

    global_agent    = env.load_scheduler(global_env,   4)
    single_agent    = env.load_scheduler(single_env,    5)



    # --- Load existing scores & steps ---
    scores_csv = '__cache__/arrays/scores.csv'
    if os.path.exists(scores_csv):
        df_scores = pd.read_csv(scores_csv)
        steps = df_scores['round'].tolist()
        scores_global_agent = df_scores['score_global_total'].tolist()
        scores_single_agent = df_scores['score_single_total'].tolist()
        scores_per_task_global = df_scores['score_global_per_task'].tolist()
        scores_per_task_single = df_scores['score_single_per_task'].tolist()
        scores_per_time_global = df_scores['score_global_per_time'].tolist()
        scores_per_time_single = df_scores['score_single_per_time'].tolist()
    else:
        steps = []
        scores_global_agent = []
        scores_single_agent = []
        scores_per_task_global = []
        scores_per_task_single = []
        scores_per_time_global = []
        scores_per_time_single = []

    # --- Load existing wait & exec metrics ---
    metrics_csv = '__cache__/arrays/wait_and_exec_metrics.csv'
    expected_cols = {
        'round': ['round', 'Unnamed: 0'],
        'backlog_wait_global_ms': ['backlog_wait_global_ms', 'backlog_waits', 'backlog_wait'],
        'queue_wait_global_ms': ['queue_wait_global_ms', 'queue_waits', 'queue_wait'],
        'cluster_wait_global_ms': ['cluster_wait_global_ms', 'cluster_queue_waits', 'cluster_waits'],
        'backlog_wait_single_ms': ['backlog_wait_single_ms', 'backlog_waits_single'],
        'queue_wait_single_ms': ['queue_wait_single_ms', 'queue_waits_single'],
        'cluster_wait_single_ms': ['cluster_wait_single_ms', 'cluster_queue_waits_single'],
        'total_tasks_global': ['total_tasks_global', 'total_scheduled_tasks'],
        'total_tasks_single': ['total_tasks_single', 'total_scheduled_tasks_single'],
        'exec_time_global_ms': ['exec_time_global_ms', 'exec_times_global'],
        'exec_time_single_ms': ['exec_time_single_ms', 'exec_times_single'],
    }

    if os.path.exists(metrics_csv):
        df_m = pd.read_csv(metrics_csv)

        # Ensure we have a 'round' column
        if 'round' not in df_m.columns:
            if 'Unnamed: 0' in df_m.columns:
                df_m = df_m.rename(columns={'Unnamed: 0': 'round'})
            else:
                df_m.insert(0, 'round', df_m.index.astype(int))

        # For every metric, detect or initialize
        for canonical, variants in expected_cols.items():
            if canonical not in df_m.columns:
                for old in variants:
                    if old in df_m.columns:
                        df_m = df_m.rename(columns={old: canonical})
                        break
                else:
                    # still not found: initialize an empty column so .tolist() yields []
                    #df_m[canonical] = []
                    df_m[canonical] = [0] * len(df_m)


        # Finally extract lists
        rounds = df_m['round'].tolist()
        backlog_waits = df_m['backlog_wait_global_ms'].tolist()
        queue_waits = df_m['queue_wait_global_ms'].tolist()
        cluster_queue_waits = df_m['cluster_wait_global_ms'].tolist()
        backlog_waits_single = df_m['backlog_wait_single_ms'].tolist()
        queue_waits_single = df_m['queue_wait_single_ms'].tolist()
        cluster_queue_waits_single = df_m['cluster_wait_single_ms'].tolist()
        total_scheduled_tasks = df_m['total_tasks_global'].tolist()
        total_scheduled_tasks_single = df_m['total_tasks_single'].tolist()
        exec_times_global = df_m['exec_time_global_ms'].tolist()
        exec_times_single = df_m['exec_time_single_ms'].tolist()
    else:
        rounds = []
        backlog_waits = []
        queue_waits = []
        cluster_queue_waits = []
        backlog_waits_single = []
        queue_waits_single = []
        cluster_queue_waits_single = []
        total_scheduled_tasks = []
        total_scheduled_tasks_single = []
        exec_times_global = []
        exec_times_single = []

    print(f"[Info] Starting from round={len(steps)}, loaded {len(rounds)} rounds of metrics.")

    ### Load existing agent checkpoints if any
    if os.path.exists('__cache__/model/global_agent.pkl'):
        with open('__cache__/model/global_agent.pkl','rb') as f:
            global_agent = copy_pram(global_agent, pickle.load(f))
    

    # Determine new total rounds
    start_round  = len(steps)
    total_rounds = start_round + additional_rounds

    ######################################################################
    #                    Training & Aggregation Loop                      #
    ######################################################################
    for round_idx in range(start_round, total_rounds):
        print(f"\n[Round {round_idx}] ------------------------------")
        print(f"[Round {round_idx}] Task seed used for all agents: {SEED}")

        # 1) Initialize/load local agents
        envs = [env.load_env(regenerate_tasks=True, seed=SEED) for i in range(number_of_agents)]
        # ✅ Reset penalty logs for all envs at the start of the round
        for e in envs:
            e.cumulative_penalty = 0.0
        agents = []
        for i in range(number_of_agents):
            sched = env.load_scheduler(envs[i], i)
            ckpt = f'__cache__/model/agent_{i}.pkl'
            if os.path.exists(ckpt):
                with open(ckpt,'rb') as f:
                    sched = copy_pram(sched, pickle.load(f))
                print(f"[Checkpoint] Local agent {i} parameters loaded for round {round_idx}.")
            else:
                print(f"[Init] Local agent {i} has fresh parameters (no checkpoint found).")
            agents.append(sched)

#########################################################################################
        # 2) Train local agents (no scoring)
        ##########################################################################

        for i, agent in enumerate(agents):
            print(f"[Training] Local agent {i} in round {round_idx}")
            agent.is_training = True
            agent.environment = envs[i]
            # --- FedProx: pin centre to current (global) weights and enable penalty ---
            if USE_FEDPROX:
                agent.dqn_train.enable_prox(FEDPROX_MU)
                agent.dqn_train.set_prox_center_from_current()
            else:
                agent.dqn_train.enable_prox(0.0)  # ensure disabled for baseline
            # --------------------------------------------------------------------------
            agent.train()
            agent.dqn_train.model.save()
###########################################################################################
        # 3) Train single-agent (no testing yet)
        ##########################################################################
        print(f"[Training] Single agent (DQN) in round {round_idx}")
        single_env = env.load_env()
        single_agent = env.load_scheduler(single_env, 5)
        single_agent.is_training = True
        single_agent.environment = single_env
        single_agent.dqn_train.enable_prox(0.0)
        single_agent.train()
        single_agent.dqn_train.model.save()

##############################################################################################
        # 4) FedAvg aggregate and test global
        #############################################################################
        print(f"[Aggregation] Combining local agents via FedAvg/FedProx...")
        global_agent = combine_agents(global_agent, agents)
        agents = distribute_agents(global_agent, agents)

        # Evaluate single-agent baseline
        avg_total_s, avg_task_s, avg_time_s = test_agent(
            env.load_env, single_agent, base_seed=SEED
        )
        scores_single_agent.append(avg_total_s)
        scores_per_task_single.append(avg_task_s)
        scores_per_time_single.append(avg_time_s)

        # Evaluate federated global agent
        avg_total_g, avg_task_g, avg_time_g = test_agent(
            env.load_env, global_agent, base_seed=SEED
        )
        scores_global_agent.append(avg_total_g)
        scores_per_task_global.append(avg_task_g)
        scores_per_time_global.append(avg_time_g)

        # 5) Collect waiting & execution metrics with new task sets
        eg = env.load_env(regenerate_tasks=True, seed=SEED)
        es = env.load_env(regenerate_tasks=True, seed=SEED)
        global_agent.environment = eg
        single_agent.environment = es
        global_agent.is_training = single_agent.is_training = False

        bw_g = qw_g = cq_g = 0; tasks_g = 0
        bw_s = qw_s = cq_s = 0; tasks_s = 0
        while not eg.terminated():
            global_agent.schedule()
            bw_g += len(eg.backlog)
            qw_g += len(eg.queue)
            cq_g += sum(len(c.cluster_queue) for c in eg.clusters)
            tasks_g += 1
        while not es.terminated():
            single_agent.schedule()
            bw_s += len(es.backlog)
            qw_s += len(es.queue)
            cq_s += sum(len(c.cluster_queue) for c in es.clusters)
            tasks_s += 1

        et_g = eg.timestep_counter
        et_s = es.timestep_counter


        # ============================== [Failure tracking per round] =======================================
        def timeout_failures_summary(env_obj):
            """
            Collect all failure types (dispatch, node, deadline) per environment.
            """
            total = len(env_obj.tasks)
            dispatch_cnt = env_obj.dispatch_fail_count
            node_cnt = env_obj.node_place_fail_count
            deadline_cnt = getattr(env_obj, "deadline_fail_count", 0)

            return {
                "dispatch_fail_count": dispatch_cnt,
                "node_place_fail_count": node_cnt,
                "deadline_fail_count": deadline_cnt,
                "total_fail_rate_percent": 100.0 * (dispatch_cnt + node_cnt + deadline_cnt) / max(1, total)
            }


        g_timeout = timeout_failures_summary(eg)
        s_timeout = timeout_failures_summary(es)

        fail_df_path = '__cache__/arrays/failure_rates_per_round.csv'
        os.makedirs('__cache__/arrays', exist_ok=True)

        if os.path.exists(fail_df_path):
            df_fail = pd.read_csv(fail_df_path)
        else:
            df_fail = pd.DataFrame(columns=[
                'round',
                'dispatch_fail_global', 'node_fail_global', 'deadline_fail_global', 'total_fail_rate_global',
                'dispatch_fail_single', 'node_fail_single', 'deadline_fail_single', 'total_fail_rate_single'
            ])

        df_fail = pd.concat([df_fail, pd.DataFrame([{
            'round': round_idx,
            'dispatch_fail_global': g_timeout["dispatch_fail_count"],
            'node_fail_global': g_timeout["node_place_fail_count"],
            'deadline_fail_global': g_timeout["deadline_fail_count"],
            'total_fail_rate_global': g_timeout["total_fail_rate_percent"],
            'dispatch_fail_single': s_timeout["dispatch_fail_count"],
            'node_fail_single': s_timeout["node_place_fail_count"],
            'deadline_fail_single': s_timeout["deadline_fail_count"],
            'total_fail_rate_single': s_timeout["total_fail_rate_percent"]
        }])], ignore_index=True)

        # ✅ Save failure rates cumulatively across runs
        if os.path.exists(fail_df_path):
            df_old_fail = pd.read_csv(fail_df_path)
            df_fail = pd.concat([df_old_fail, df_fail]).drop_duplicates(subset=['round'], keep='last')
        df_fail.to_csv(fail_df_path, index=False)
        print(f"[Save] ✅ Updated failure log → {fail_df_path}")

        # ================================================================
        # ✅ [Penalty tracking per round]
        # ================================================================
        penalty_csv = "__cache__/arrays/penalty_per_round.csv"
        os.makedirs("__cache__/arrays", exist_ok=True)

        total_penalties = [e.cumulative_penalty for e in [eg, es]]
        avg_penalty = sum(total_penalties) / len(total_penalties)

        df_pen = pd.DataFrame([{
            "round": round_idx,
            "avg_penalty": avg_penalty,
            "penalty_global": total_penalties[0],
            "penalty_single": total_penalties[1]
        }])

        # ✅ Save penalty log cumulatively across runs
        if os.path.exists(penalty_csv):
            df_old_pen = pd.read_csv(penalty_csv)
            df_pen = pd.concat([df_old_pen, df_pen]).drop_duplicates(subset=['round'], keep='last')
        df_pen.to_csv(penalty_csv, index=False)
        print(f"[Save] ✅ Updated penalty log → {penalty_csv}")
        print(
            f"[Penalty Log] Round {round_idx} → avg_penalty={avg_penalty:.3f} (global={total_penalties[0]:.3f}, single={total_penalties[1]:.3f})")

        # ========================================================================

        # 6) Append round metrics
        steps.append(round_idx)
        rounds.append(round_idx)
        backlog_waits.append(bw_g * TIME_STEP_MS)
        queue_waits.append(qw_g * TIME_STEP_MS)
        cluster_queue_waits.append(cq_g * TIME_STEP_MS)
        backlog_waits_single.append(bw_s * TIME_STEP_MS)
        queue_waits_single.append(qw_s * TIME_STEP_MS)
        cluster_queue_waits_single.append(cq_s * TIME_STEP_MS)
        total_scheduled_tasks.append(tasks_g)
        total_scheduled_tasks_single.append(tasks_s)
        exec_times_global.append(et_g * TIME_STEP_MS)
        exec_times_single.append(et_s * TIME_STEP_MS)

        # 7.8) Save the local agents' updated parameters for next round's initialization
        for i, agent in enumerate(agents):
            agent_path = f'__cache__/model/agent_{i}.pkl'
            agent_params_current = [var.numpy() for var in agent.dqn_train.model.trainable_variables]
            if not os.path.exists('__cache__/model'):
                os.makedirs('__cache__/model')
            with open(agent_path, 'wb') as f:
                pickle.dump(agent_params_current, f)


        # --- Ensure all metric lists have the same length ---
        lists_to_check = [
            rounds, backlog_waits, queue_waits, cluster_queue_waits,
            backlog_waits_single, queue_waits_single, cluster_queue_waits_single,
            total_scheduled_tasks, total_scheduled_tasks_single,
            exec_times_global, exec_times_single
        ]

        min_len = min(len(lst) for lst in lists_to_check)


        # Helper to align list sizes
        def align_length(lst, target_len, fill_value=0):
            if len(lst) > target_len:
                return lst[:target_len]
            elif len(lst) < target_len:
                return lst + [fill_value] * (target_len - len(lst))
            return lst


        # Truncate/pad all lists to the same length
        rounds = align_length(rounds, min_len)
        backlog_waits = align_length(backlog_waits, min_len)
        queue_waits = align_length(queue_waits, min_len)
        cluster_queue_waits = align_length(cluster_queue_waits, min_len)
        backlog_waits_single = align_length(backlog_waits_single, min_len)
        queue_waits_single = align_length(queue_waits_single, min_len)
        cluster_queue_waits_single = align_length(cluster_queue_waits_single, min_len)
        total_scheduled_tasks = align_length(total_scheduled_tasks, min_len)
        total_scheduled_tasks_single = align_length(total_scheduled_tasks_single, min_len)
        exec_times_global = align_length(exec_times_global, min_len)
        exec_times_single = align_length(exec_times_single, min_len)

        # --- Save the metrics safely ---
        os.makedirs('__cache__/arrays', exist_ok=True)

        np.savetxt('__cache__/arrays/steps.csv', steps, fmt='%d')
        # wait metrics
        np.savetxt('__cache__/arrays/backlog_waits.csv', backlog_waits, fmt='%.6f')
        np.savetxt('__cache__/arrays/queue_waits.csv', queue_waits, fmt='%.6f')
        np.savetxt('__cache__/arrays/cluster_queue_waits.csv', cluster_queue_waits, fmt='%.6f')
        np.savetxt('__cache__/arrays/backlog_waits_single.csv', backlog_waits_single, fmt='%.6f')
        np.savetxt('__cache__/arrays/queue_waits_single.csv', queue_waits_single, fmt='%.6f')
        np.savetxt('__cache__/arrays/cluster_queue_waits_single.csv', cluster_queue_waits_single, fmt='%.6f')
        # task counts & exec times
        np.savetxt('__cache__/arrays/total_tasks.csv', total_scheduled_tasks, fmt='%d')
        np.savetxt('__cache__/arrays/total_tasks_single.csv', total_scheduled_tasks_single, fmt='%d')
        np.savetxt('__cache__/arrays/exec_times_global.csv', exec_times_global, fmt='%.6f')
        np.savetxt('__cache__/arrays/exec_times_single.csv', exec_times_single, fmt='%.6f')



    # =======================================================
    # ✅ Save or update waiting/execution metrics cumulatively
    # =======================================================
    df_metrics_new = pd.DataFrame({
        'round': rounds,
        'backlog_wait_global_ms': backlog_waits,
        'queue_wait_global_ms': queue_waits,
        'cluster_wait_global_ms': cluster_queue_waits,
        'backlog_wait_single_ms': backlog_waits_single,
        'queue_wait_single_ms': queue_waits_single,
        'cluster_wait_single_ms': cluster_queue_waits_single,
        'total_tasks_global': total_scheduled_tasks,
        'total_tasks_single': total_scheduled_tasks_single,
        'exec_time_global_ms': exec_times_global,
        'exec_time_single_ms': exec_times_single
    })

    if os.path.exists(metrics_csv):
        df_old_metrics = pd.read_csv(metrics_csv)
        df_combined_metrics = pd.concat([df_old_metrics, df_metrics_new]) \
                                 .drop_duplicates(subset=['round'], keep='last')
    else:
        df_combined_metrics = df_metrics_new

    df_combined_metrics.to_csv(metrics_csv, index=False)
    print(f"[Save] ✅ Updated metrics → {metrics_csv}")


    # =======================================================
    # ✅ Save or update scores after all training rounds
    # =======================================================
    os.makedirs('__cache__/arrays', exist_ok=True)

    df_scores_new = pd.DataFrame({
        'round': steps,
        'score_global_total': scores_global_agent,
        'score_single_total': scores_single_agent,
        'score_global_per_task': scores_per_task_global,
        'score_single_per_task': scores_per_task_single,
        'score_global_per_time': scores_per_time_global,
        'score_single_per_time': scores_per_time_single
    })

    # If old file exists, merge old + new safely (avoid duplicates)
    if os.path.exists(scores_csv):
        df_old = pd.read_csv(scores_csv)
        # combine & keep last record per round (avoid duplicates)
        df_combined = pd.concat([df_old, df_scores_new]).drop_duplicates(subset=['round'], keep='last')
    else:
        df_combined = df_scores_new

    df_combined.to_csv(scores_csv, index=False)
    print(f"[Save] ✅ Updated scores → {scores_csv}")




    #################### Plotting after all rounds##########################
    ########################################################################
    print('Plotting scores and metrics...')

    scores_plot(
        steps,
        scores_global_agent, scores_single_agent,
        scores_per_task_global, scores_per_task_single,
        scores_per_time_global, scores_per_time_single
    )


    plot_round_metrics(
        rounds,
        backlog_waits,
        queue_waits,
        cluster_queue_waits,
        total_scheduled_tasks,
        exec_times_global,
        backlog_waits_single,
        queue_waits_single,
        cluster_queue_waits_single,
        total_scheduled_tasks_single,
        exec_times_single
    )

    plot_failure_rates_per_round()

    plot_failure_types_stacked()

    plot_penalty_per_round()

    # 8) Save final agent parameters
    os.makedirs('__cache__/model', exist_ok=True)
    with open('__cache__/model/global_agent.pkl', 'wb') as f:
        pickle.dump([v.numpy() for v in global_agent.dqn_train.model.trainable_variables], f)
   