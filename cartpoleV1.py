import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import numpy as np
import gymnasium as gym

def network(layerSizes, activation=nn.Tanh, output_activation=nn.Identity):
    # Build a feedforward neural network.
    layers = []
    for i in range(len(layerSizes)-1):
        #activation function of middle layers = activation, else output activation = output_activation
        act = activation if i < len(layerSizes)-2 else output_activation
        #appends an array of [linear transformations y = wx + b, activation function (y)]
        layers += [nn.Linear(layerSizes[i], layerSizes[i+1]), act()]

    # nn.Sequential takes any amount of arguments, and basically returns a simple neural network 
    return nn.Sequential(*layers)

def train(env_name='CartPole-v1', hidden_sizes=50, learning_rate=1e-2, epochs=50, batch_size=2000):
    
    env = gym.make(env_name, render_mode = 'human')
    input_size = env.observation_space.shape[0]
    output_size = env.action_space.n

    logits_network = network(layerSizes=[input_size, hidden_sizes, output_size])

    def get_policy(observation):
        observation = torch.as_tensor(observation)
        logits = logits_network(observation)
        return Categorical(logits=logits)
    
    def get_action(observation):
        return get_policy(observation).sample().item()
    
    def get_loss(observation, act, episode_returns):
        log_prob = get_policy(observation).log_prob(act)
        return -(log_prob * episode_returns).mean()
    
    optimizer = Adam(logits_network.parameters(), lr=learning_rate)

    def train_one_epoch():
        observations_list = []
        actions_list = []
        episode_returns = []
        episode_rewards = []
        batch_returns = []
        episode_lens = []

        obs = env.reset()[0]
        done = False
        truncated = False
    
        while(True):
            # original code appends a copy of the observation which is obsolete: obs will never be modified in place.
            observations_list.append(obs)

            action = get_action(obs)
            obs,reward,done,failed,_ = env.step(action)
            episode_rewards.append(reward)
            actions_list.append(action)

            if done or failed:
                episode_lens.append(len(episode_rewards))
                sampled_return = np.sum(episode_rewards)
                batch_returns.append(sampled_return)
                episode_returns.extend([sampled_return] * len(episode_rewards))
                obs,done,failed,episode_rewards = env.reset()[0], False, False, []
                if len(actions_list) > batch_size:
                    break
        
        optimizer.zero_grad()
        # print(episode_returns)
        loss = get_loss(observation=torch.as_tensor(observations_list, dtype=torch.float32),
                        act = torch.as_tensor(actions_list, dtype=torch.float32),
                        episode_returns = torch.as_tensor(episode_returns, dtype=torch.float32))        
        loss.backward()
        optimizer.step()

        return batch_returns, episode_lens

    for i in range(epochs):
        returns, lens = train_one_epoch()
        # for cartpole, the len of an episode = return of that episode
        print(f"epoch #{i}: average epoch return: {np.mean(returns)}, average epoch lengths: {np.mean(lens)}")

train()