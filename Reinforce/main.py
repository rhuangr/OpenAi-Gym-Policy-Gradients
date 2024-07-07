import torch
import torch.nn as nn
from torch.optim import Adam
import gymnasium as gym
from numpy import sum

class simple_nn(nn.Module):
    def __init__(self,input_size, output_size, middle_size = 64, activation = nn.ReLU()) -> None:
        super(simple_nn, self).__init__()

        self.layer1 = nn.Linear(input_size, middle_size)
        self.layer2 = nn.Linear(middle_size, middle_size )
        self.layer3 = nn.Linear(middle_size, output_size)
        self.activation = activation
        self.optimizer = Adam(self.parameters(), lr = 0.01)

    def forward(self, input):
        input = torch.as_tensor(input, dtype=torch.float32)
        x = self.layer1(input)
        x = self.activation(x)
        x = self.layer2(x)
        x = self.activation(x)
        x = self.layer3(x)
        return x
    

def train(env_name='CartPole-v1', hidden_sizes=50, learning_rate=1e-2, epochs=50, batch_size=2000):
    
    env = gym.make(env_name, render_mode = 'human')
    obs_space = env.observation_space.shape[0]
    action_space = env.action_space.n

    actor = simple_nn(obs_space, action_space)
    critic = simple_nn(obs_space, 1)

    def get_action(obs):
        logits = actor(obs)
        distribution = torch.distributions.Categorical(logits=logits)
        action = distribution.sample()
        return action, distribution.log_prob(action)
    
    def get_state_values(batch_obs):
        state_value = critic(batch_obs)
        return state_value.squeeze()

    def train_one_epoch():
        batch_obs = []
        batch_rew = []
        batch_act = []
        batch_log_probs = []
        batch_returns = []

        episode_rew = []

        obs = env.reset()[0]

        while (True):
            batch_obs.append(0)
            action, log_prob = get_action(obs)
            obs, reward, done, failed, _ = env.step()
            episode_rew.append(reward)
            batch_act.append(action)
            batch_log_probs.append(log_prob)
            batch_obs.append(obs)

            if done or failed:
                obs = env.reset()[0]
                batch_rew.append(episode_rew)
                episode_return = sum(episode_rew)
                batch_returns.extend([episode_return] * len(episode_rew))
                obs = env.reset()[0]
                episode_rew = []

                if len(batch_act) > batch_size:
                    break
    
        actor.zero_grad()
        state_values = get_state_values(batch_obs)
        actor_loss = -(torch.as_tensor(batch_log_probs, dtpe=torch.float32)*
                       (torch.as_tensor(batch_returns, dtype=torch.float32) - state_values.detach())).mean()
        
        actor_loss.backward()
        actor.optimizer.step()


        critic.zero_grad()
        critic_loss = ((state_values - torch.as_tensor(batch_returns, dtype=torch.float32)))**2
        critic_loss.backward()
        critic.optimizer.step()

train()




