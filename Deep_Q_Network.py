import torch
import numpy as np
import matplotlib.pyplot as plt

from dqn_agent import Agent
from collections import deque
from unityagents import UnityEnvironment

def new_unity_environment():
    env = UnityEnvironment(file_name=".\\Banana_Windows_x86_64\\Banana.exe")
    
    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    
    env_info = env.reset(train_mode=False)[brain_name] # reset the environment
    state = env_info.vector_observations[0]            # get the current state
    
    # look up the size of the action and state spaces
    state_size = env_info.vector_observations[0].shape[0]
    action_size = brain.vector_action_space_size
    
    return (brain_name, env, env_info, state, state_size, action_size)

def dqn_train(agent, env, brain_name, state_size, n_episodes=1200, max_t=300, eps_start=1.0, eps_end=0.005, eps_decay=0.995):
    """Deep Q-Learning.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    state = deque(maxlen=state_size)   # <- init with all zeros (below at start of episode)
    
    for i_episode in range(1, n_episodes+1):
        # reset the environment for the start of a new episode
        env_info = env.reset(train_mode=True)[brain_name] # reset the environment
        
        # get the current state
        _state = env_info.vector_observations[0]           
        
        # reset and fill with zeros on a new episode
        [state.append((0)) for _ in range(state_size)]
        
        #encoded actions can be zero to start
        one_hot_actions = [0 for i in range(agent.action_size)]
        
        # since the project was 'open ended' I had a little fun with this part
        # here I'm giving the network the last few states as well as the current one, plus the 
        # last reward and last action taken for those prior states.
        state.extend(_state) # state
        state.append((0)) # reward
        state.extend(one_hot_actions) #actions
        
        score = 0
        for t in range(max_t):
            state_np = np.asarray(state)
            action = int(agent.act(state_np, eps))
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]
            
            one_hot_actions[action] = 1  # let the network know what action was just taken
            state.extend(next_state)     # add the next environment state to the training state
            state.append((reward))       # add the reward to the training state
            one_hot_actions[action] = 0  # reset action vector done
            state.extend(one_hot_actions)# add the action to the training state
            
            # state_np is the original state for this step, and state has already been updated with next_state
            agent.step(state_np, action, reward, np.asarray(state), done)
            score += reward
            if done:
                break 
            
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        mean_score = np.mean(scores_window)
        
        print('\rEpisode {}\tAverage Score: {:.2f}\tActual Score:{:.2f}'.format(i_episode, mean_score, score), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}\tActual Score:{:.2f}'.format(i_episode, mean_score, score))
        if np.mean(scores_window)>=15.0:  # solved is 13 but this agent is able to perform better, so lets go until 15
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}\tActual Score:{:.2f}'.format(i_episode-100, mean_score, score))
            torch.save(agent.qnetwork_local.state_dict(), './{:.2f}_checkpoint.pth'.format(mean_score))
            break  # or not and just keep on keepin on
    return scores

