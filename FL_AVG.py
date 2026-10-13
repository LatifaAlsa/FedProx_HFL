# FL_AVG.py

def combine_agents(main_agent, agents):
    for i in range(len(agents)):
        for main_vars, agent_var in zip(main_agent.dqn_train.model.trainable_variables, agents[i].dqn_train.model.trainable_variables):
            if i == 0:
                main_vars.assign(agent_var)
            else:
                main_vars.assign(main_vars * (i / (i + 1)) + agent_var * (1 / (i + 1)))

        for main_vars, agent_var in zip(main_agent.dqn_target.model.trainable_variables, agents[i].dqn_target.model.trainable_variables):
            if i == 0:
                main_vars.assign(agent_var)
            else:
                main_vars.assign(main_vars * (i / (i + 1)) + agent_var * (1 / (i + 1)))

    return main_agent


def distribute_agents(main_agent, agents):
    for i in range(len(agents)):
        for main_agent_param, agent_param in zip(main_agent.dqn_target.model.trainable_variables, agents[i].dqn_target.model.trainable_variables):
            agent_param.assign(main_agent_param)
        for main_agent_param, agent_param in zip(main_agent.dqn_train.model.trainable_variables, agents[i].dqn_train.model.trainable_variables):
            agent_param.assign(main_agent_param)
    return agents


def combine_agents_reward_based(main_agent, agents, scores):
    total_reward = sum(scores)

    for i in range(len(agents)):
        for main_param, agent_param in zip(main_agent.dqn_train.model.trainable_variables, agents[i].dqn_train.model.trainable_variables):
            if i == 0:
                main_param.assign(agent_param * (scores[i] / total_reward))
            else:
                main_param.assign(main_param + agent_param * (scores[i] / total_reward))


        for main_param, agent_param in zip(main_agent.dqn_target.model.trainable_variables, agents[i].dqn_target.model.trainable_variables):
            if i == 0:
                main_param.assign(agent_param * (scores[i] / total_reward))
            else:
                main_param.assign(main_param + agent_param * (scores[i] / total_reward))

    return main_agent


def copy_pram(agent, agent_pram):
    """
    Copy a list of numpy weight arrays into BOTH the train and target DQNs.
    """
    # 1) Copy into the train network
    for var, pram in zip(agent.dqn_train.model.trainable_variables, agent_pram):
        var.assign(pram)

    # 2) Copy into the target network
    for var, pram in zip(agent.dqn_target.model.trainable_variables, agent_pram):
        var.assign(pram)

    return agent


def copy_single_pram(agent, agent_pram):
    """
    Copy parameters into both the train and target networks.
    """
    for var, pram in zip(agent.dqn_train.model.trainable_variables, agent_pram):
        var.assign(pram)
    for var, pram in zip(agent.dqn_target.model.trainable_variables, agent_pram):
        var.assign(pram)
    return agent
