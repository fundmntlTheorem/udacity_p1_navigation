from collections import deque
import numpy as np
import torch
from sys import float_info
from dqn_agent import Agent

SCORE_WINDOW = 100

def dqn(config, env, agent, brain_name, save_name):
    """Deep Q-Learning.
    Params
    ======
        num_episodes (int): maximum number of training episodes
        max_time (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=SCORE_WINDOW)  # last 100 scores
    eps = config['eps_start']          # initialize epsilon
    max_score = float_info.min
    for i_episode in range(1, config['num_episodes']+1):        
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0] 
        score = 0
        for _ in range(config['max_time']):
            # get an action from the agent
            action = agent.act(state, eps)
            # update the environment with this action
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations[0]   # get the next state
            reward = env_info.rewards[0]                   # get the reward
            done = env_info.local_done[0]                  # see if episode has finished
            # update the agent
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break 
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score

        eps = max(config['eps_end'], config['eps_decay']*eps) # decrease epsilon
        mean_score = np.mean(scores_window)
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, mean_score), end="")
        if i_episode % SCORE_WINDOW == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, mean_score))
        
        if mean_score > max_score:
            max_score = mean_score
            torch.save({
                    'net': agent.qnetwork_local.state_dict(), 
                    'config': config,
                    'scores': scores,
                },
                save_name)
    return scores


def run(env, agent, brain_name):
    '''
        Run an agent on the environment
        This is the loop from the lab demo, with the action line
        replaced with the agent acting on the environment
    '''
    # run the agent with the network    
    env_info = env.reset(train_mode=False)[brain_name] # reset the environment
    state = env_info.vector_observations[0]            # get the current state
    score = 0                                          # initialize the score
    while True:
        action = agent.act(state)
        env_info = env.step(action)[brain_name]        # send the action to the environment
        next_state = env_info.vector_observations[0]   # get the next state
        reward = env_info.rewards[0]                   # get the reward
        done = env_info.local_done[0]                  # see if episode has finished
        score += reward                                # update the score
        state = next_state                             # roll over the state to next time step
        if done:                                       # exit loop if episode finished
            break
    print('Final Score {}'.format(score))