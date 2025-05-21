import gym
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import copy
from tqdm import tqdm
import random
import collections
from torch.utils.tensorboard import SummaryWriter
import os

from multiprocessing import Process, Pipe
from pettingzoo.mpe import simple_spread_v3
from value_norm import ValueNorm
os.environ['SDL_VIDEODRIVER'] = 'dummy'
def huber_loss(e, d):
    a = (abs(e) <= d).float()
    b = (e > d).float()
    return a*e**2/2 + b*d*(abs(e)-d/2)
class EnvWrapper:
    def __init__(self):
        self.env = simple_spread_v3.parallel_env()
        self.N=3
    def reset(self,seed=None):
        obs,infos=self.env.reset(seed=seed)
        return self.dict_to_list(obs),self.dict_to_list(infos)
    def step(self,action):
        obs,rewards,dones,tructs,infos=self.env.step(self.list_to_dict(action))
        return self.dict_to_list(obs),self.dict_to_list(rewards),self.dict_to_list(dones),self.dict_to_list(tructs),self.dict_to_list(infos)
    def list_to_dict(self,infos):
        result={}
        for i in range(3):
            result["agent_%d"%i]=infos[i]
        return result
    def dict_to_list(self,infos):
        result=[]
        for i in range(3):
            result.append(infos["agent_%d"%i])
        return result
def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper()
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            obs, reward, done, truct,info = env.step(data)
            remote.send((obs, reward, done,truct, info))
        elif cmd == 'reset':
            remote.send(env.reset()[0])
        elif cmd == 'close':
            remote.close()
            break

class ParallelEnv:
    def __init__(self, n_envs):
        self.n_envs = n_envs
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(n_envs)])
        self.ps = [Process(target=worker, args=(work_remote, remote, EnvWrapper))
                   for (work_remote, remote) in zip(self.work_remotes, self.remotes)]
        for p in self.ps:
            p.daemon = True
            p.start()
        for remote in self.work_remotes:
            remote.close()
    def sample_action(self):
        return np.random.randint(0,4,size=(self.n_envs,3))
    def step(self, actions):
        #n_env n_agent
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        results = [remote.recv() for remote in self.remotes]
        obs, rewards, dones, tructs,infos = zip(*results)
        #n_env n_agent d
        return obs, rewards, dones, tructs,infos

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        return [remote.recv() for remote in self.remotes]

    def close(self):
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()



def train_on_policy_agent(env, agent, num_episodes,batch_size):
    exist_exp_id=[int(exp) for exp in os.listdir("logs")]
    if len(exist_exp_id)>0:
        exist_exp_id=sorted(exist_exp_id)
        exp_id=exist_exp_id[-1]+1
    else:
        exp_id=0
    exp_dir="logs/%d"%exp_id
    os.makedirs(exp_dir)
    writer=SummaryWriter(exp_dir)
    return_list = []
    transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
    i_eps=0
    num_update=0
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                state = env.reset()
                done = False
                episode_return = 0.0
                while True:
                    action = agent.take_action(state)
                    next_state, reward, done,truct, _ = env.step(action)
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(truct)
                    state = next_state
                    episode_return += np.array(reward).sum()/len(reward)
                    # if truct:
                    #     break
                    if done[0][0] or truct[0][0]:
                        break
                assert len(transition_dict["rewards"])<=batch_size
                if len(transition_dict["rewards"])==batch_size:
                    a_loss,c_loss,e_loss=agent.update(transition_dict)
                    writer.add_scalar('actor_loss',a_loss,i_eps)
                    writer.add_scalar('critic_loss',c_loss,i_eps)
                    writer.add_scalar('entropy_loss',e_loss,i_eps)
                    writer.add_scalar('individual_reward',np.array(transition_dict['rewards']).mean(),i_eps)
                    writer.add_scalar('rewards',episode_return,num_update)
                    num_update+=1
                    transition_dict ={'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
                return_list.append(episode_return)
                i_eps+=1
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                                      'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
        agent.save(i)
    return return_list
def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module
class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        gain = nn.init.calculate_gain(['tanh', 'relu'][True])
        def init_(m): 
            return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0),gain=gain)
        self.layer1 = nn.Sequential(init_(torch.nn.Linear(state_dim, hidden_dim)),nn.ReLU(),nn.LayerNorm(hidden_dim))
        self.layer2= nn.Sequential(init_(torch.nn.Linear(hidden_dim,hidden_dim)),nn.ReLU(),nn.LayerNorm(hidden_dim))
        def init_2(m):
            return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0),gain=0.01)
        self.fc = init_2(torch.nn.Linear(hidden_dim, action_dim))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return F.softmax(self.fc(x), dim=-1)


class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        gain = nn.init.calculate_gain(['tanh', 'relu'][True])
        def init_(m): 
            return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0),gain=gain)
        self.layer1 = nn.Sequential(init_(torch.nn.Linear(state_dim, hidden_dim)),nn.ReLU(),nn.LayerNorm(hidden_dim))
        self.layer2= nn.Sequential(init_(torch.nn.Linear(hidden_dim,hidden_dim)),nn.ReLU(),nn.LayerNorm(hidden_dim))  
        def init_2(m):
            return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))  
        self.fc = init_2(torch.nn.Linear(hidden_dim, 1))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return self.fc(x)
def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.cpu().detach().numpy()
    T=td_delta.shape[0]
    advantage_list = []
    advantage = 0.0
    for i in reversed(range(T)):
        advantage = gamma * lmbda * advantage + td_delta[i]
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(np.array(advantage_list), dtype=torch.float)
class PPO:
    ''' PPO算法,采用截断方式 '''
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,N,
                 lmbda, epochs, eps, gamma, device):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim*N, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr,eps=1e-5)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr,eps=1e-5)
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs  # 一条序列的数据用来训练轮数
        self.eps = eps  # PPO中截断范围的参数
        self.device = device
        self.value_norm=ValueNorm(1,device=device,enable=True)
        self.episode_limit=25
    def cal_value_loss(self, old_values, new_values, targets):
        self.value_norm.update(targets)
        value_clipped = old_values + (new_values - old_values).clamp(-0.2, 0.2)
        error_clipped = self.value_norm.normalize(targets) - value_clipped
        error_original = self.value_norm.normalize(targets) - new_values

        value_loss_clipped = huber_loss(error_clipped, 10.0)
        value_loss_original = huber_loss(error_original, 10.0)

        value_loss = torch.max(value_loss_original, value_loss_clipped)
        value_loss = value_loss.mean()
        return value_loss

    def take_action(self, state):
        self.actor.eval()
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.cpu().numpy() #bs

    def update(self, states,shared_states,actions,rewards,shared_next_states,dones):
        self.critic.eval()
        self.actor.eval()
        td_target = rewards + self.gamma * self.value_norm.denormalize(self.critic(shared_next_states).detach()) * (1 -
                                                                       dones)
        td_delta = td_target - self.value_norm.denormalize(self.critic(shared_states).detach())
        advantage = compute_advantage(self.gamma, self.lmbda,
                                               td_delta.reshape(-1,self.episode_limit).transpose(0,1)).to(self.device)
        advantage = advantage.transpose(0,1).reshape(-1,1)
        old_values=self.critic(shared_states).detach()
        td_target = (self.value_norm.denormalize(old_values)+advantage).detach()
        advantage= (advantage-advantage.mean())/(advantage.std()+1e-9)
        old_log_probs = torch.log(self.actor(states).gather(1,
                                                            actions)).detach()
        
        total_actor_loss,total_critic_loss,total_entropy,total_ratio=0.0,0.0,0.0,0.0
        self.critic.train()
        self.actor.train()
        for _ in range(self.epochs):
            probs=self.actor(states)
            entropy=-torch.sum(probs*probs.log(),dim=-1).mean()
            log_probs = torch.log(probs.gather(1, actions))
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps,
                                1 + self.eps) * advantage  # 截断
            actor_loss = torch.mean(-torch.min(surr1, surr2))  # PPO损失函数
            critic_loss = self.cal_value_loss(old_values,self.critic(shared_states), td_target.detach())
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            (actor_loss-0.01*entropy).backward()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
            nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
            self.actor_optimizer.step()
            self.critic_optimizer.step()
            total_actor_loss+=actor_loss.item()
            total_critic_loss+=critic_loss.item()
            total_entropy+=entropy.item()
            total_ratio+=ratio.mean().item()
        return total_actor_loss/self.epochs,total_critic_loss/self.epochs,total_entropy/self.epochs
    def save(self,prefix):
        torch.save(self.actor.state_dict(),"models/"+str(prefix)+"_actor.pth")
        torch.save(self.critic.state_dict(),"models/"+str(prefix)+"_critic.pth")
class MAPPO:
    ''' PPO算法,采用截断方式 '''
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,N,
                 lmbda, epochs, eps, gamma, device):
        self.agents=[PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr,N,
                 lmbda, epochs, eps, gamma, device) for i in range(N)]
        self.device = device
        self.N=N
    def take_action(self, states):
        #n_env n_agent d
        states = torch.tensor(np.array(states), dtype=torch.float).to(self.device)
        with torch.no_grad():
            #n_agent n_e->n_e n_a
            action=[]
            for i in range(self.N):
                action.append(self.agents[i].take_action(states[:,i,:]))
            return np.array(action).transpose((1,0))
    

    def update(self, transition_dict):
        #steps n_e n_a d->n_e steps n_a d->bs n_a d->na bs d
        states=torch.tensor(np.array(transition_dict['states']),
                            dtype=torch.float).to(self.device).transpose(0,1).flatten(0,1).transpose(0,1)
        actions=torch.tensor(np.array(transition_dict['actions'])).unsqueeze(-1).to(
            self.device).transpose(0,1).flatten(0,1).transpose(0,1)
        next_states=torch.tensor(np.array(transition_dict['next_states']),
                            dtype=torch.float).to(self.device).transpose(0,1).flatten(0,1).transpose(0,1)
        rewards=torch.tensor(np.array(transition_dict['rewards']),
                            dtype=torch.float).unsqueeze(-1).to(self.device).transpose(0,1).flatten(0,1).transpose(0,1)
        dones=torch.tensor(np.array(transition_dict['dones']),
                            dtype=torch.float).unsqueeze(-1).to(self.device).transpose(0,1).flatten(0,1).transpose(0,1)
        #n_a bs d -> bs n_a d -> bs d*
        shared_states=states.transpose(0,1).flatten(-2,-1) #bs n*d
        shared_next_states=next_states.transpose(0,1).flatten(-2,-1)#bs n*d
        total_actor_loss,total_critic_loss,total_entropy=0.0,0.0,0.0
        for i in range(self.N):
            actor_loss,critic_loss,entropy=self.agents[i].update(states[i],shared_states,actions[i],rewards[i],shared_next_states,dones[i])
            total_actor_loss+=actor_loss
            total_critic_loss+=critic_loss
            total_entropy+=entropy
        return total_actor_loss/self.N,total_critic_loss/self.N,total_entropy/self.N
    def save(self,i):
        for j in range(self.N):
            self.agents[j].save("{}_{}".format(i,j))
        # torch.save(self.actor.state_dict(),"actor%d.pth"%i)
        # torch.save(self.critic.state_dict(),"critic%d.pth"%i)
    def load(self,i):
        self.actor.load_state_dict(torch.load("actor%d.pth"%i))
        self.critic.load_state_dict(torch.load("critic%d.pth"%i))
class SharedPPO:
    ''' PPO算法,采用截断方式 '''
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,N,
                 lmbda, epochs, eps, gamma, device):
        self.actors =[PolicyNet(state_dim, hidden_dim, action_dim).to(device) for i in range(N)]
        self.critic = ValueNet(state_dim*N, hidden_dim).to(device)
        self.actor_optimizers= [torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)for i in range(N)]
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs  # 一条序列的数据用来训练轮数
        self.eps = eps  # PPO中截断范围的参数
        self.device = device
        self.N=N

    def take_action(self, states):
        actions={}
        for i in range(self.N):
            agent='agent_%d'%i
            state = torch.tensor([states[agent]], dtype=torch.float).to(self.device)
            probs = self.actors[i](state)
            action_dist = torch.distributions.Categorical(probs)
            action = action_dist.sample()
            actions[agent]=action
        return actions

    def update(self, transition_dict):
        states=[]
        actions=[]
        next_states=[]
        rewards=[]
        dones=[]
        td_target=[]
        advantage=[]
        old_log_probs=[]
        for i in range(self.N):
            agent='agent_%d'%i
            states.append(torch.tensor(transition_dict['states'][agent],
                                dtype=torch.float).to(self.device))
            actions.append(torch.tensor(transition_dict['actions'][agent]).view(-1, 1).to(
                self.device))
            next_states.append(torch.tensor(transition_dict['next_states'][agent],
                                   dtype=torch.float).to(self.device))
            rewards.append(torch.tensor(transition_dict['rewards'][agent],
                                dtype=torch.float).view(-1, 1).to(self.device))
            
            dones.append(torch.tensor(transition_dict['dones'][agent],
                             dtype=torch.float).view(-1, 1).to(self.device))
        shared_states=torch.cat(states,dim=-1)
        shared_next_states=torch.cat(next_states,dim=-1)

        td_target = torch.cat(rewards,dim=-1).sum(dim=-1,keepdim=True) + self.gamma * self.critic(shared_next_states) * (1 -
                                                                       dones[0])
        td_delta = td_target - self.critic(shared_states)
        advantage = compute_advantage(self.gamma, self.lmbda,
                                               td_delta.cpu()).to(self.device)
        old_log_probs = [torch.log(self.actors[i](states[i]).gather(1,
                                                            actions[i])).detach() for i in range(self.N)]

        for _ in range(self.epochs):
            total_actor_loss=0.0
            total_entropy=0.0
            for i in range(self.N):
                probs=self.actors[i](states[i])
                entropy=-torch.sum(probs*probs.log(),dim=-1).mean()
                log_probs = torch.log(probs.gather(1, actions[i]))
                ratio = torch.exp(log_probs - old_log_probs[i])
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.eps,
                                    1 + self.eps) * advantage  # 截断
                actor_loss = torch.mean(-torch.min(surr1, surr2))  # PPO损失函数

                total_actor_loss+=actor_loss
                total_entropy_loss+=entropy
            actor_loss=total_actor_loss/self.N
            entropy=total_entropy/self.N
            critic_loss = torch.mean(
                F.mse_loss(self.critic(shared_states), td_target.detach()))
            for i in range(self.N):
                self.actor_optimizers[i].zero_grad()
            self.critic_optimizer.zero_grad()
            (actor_loss-0.01*entropy).backward()
            critic_loss.backward()
            for i in range(self.N):
                self.actor_optimizers[i].step()
            self.critic_optimizer.step()
            return actor_loss.item(),critic_loss.item(),entropy.item()
class SinglePPO:
    ''' PPO算法,采用截断方式 '''
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,N,
                 lmbda, epochs, eps, gamma, device):
        self.actor =PolicyNet(state_dim, hidden_dim, action_dim).to(device) 
        self.critic = ValueNet(state_dim*N, hidden_dim).to(device)
        self.actor_optimizers= torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs  # 一条序列的数据用来训练轮数
        self.eps = eps  # PPO中截断范围的参数
        self.device = device
        self.N=N

    def take_action(self, states):
        actions={}
        for i in range(self.N):
            agent='agent_%d'%i
            state = torch.tensor([states[agent]], dtype=torch.float).to(self.device)
            probs = self.actor(state)
            action_dist = torch.distributions.Categorical(probs)
            action = action_dist.sample()
            actions[agent]=action
        return actions

    def update(self, transition_dict):
        states=[]
        actions=[]
        next_states=[]
        rewards=[]
        dones=[]
        td_target=[]
        advantage=[]
        old_log_probs=[]
        for i in range(self.N):
            agent='agent_%d'%i
            states.append(torch.tensor(transition_dict['states'][agent],
                                dtype=torch.float).to(self.device))
            actions.append(torch.tensor(transition_dict['actions'][agent]).view(-1, 1).to(
                self.device))
            next_states.append(torch.tensor(transition_dict['next_states'][agent],
                                   dtype=torch.float).to(self.device))
            rewards.append(torch.tensor(transition_dict['rewards'][agent],
                                dtype=torch.float).view(-1, 1).to(self.device))
            
            dones.append(torch.tensor(transition_dict['dones'][agent],
                             dtype=torch.float).view(-1, 1).to(self.device))
        shared_states=torch.cat(states,dim=-1)
        shared_next_states=torch.cat(next_states,dim=-1)

        td_target = torch.cat(rewards,dim=-1).sum(dim=-1,keepdim=True) + self.gamma * self.critic(shared_next_states) * (1 -
                                                                       dones[0])
        td_delta = td_target - self.critic(shared_states)
        advantage = compute_advantage(self.gamma, self.lmbda,
                                               td_delta.cpu()).to(self.device)
        old_log_probs = [torch.log(self.actor(states[i]).gather(1,
                                                            actions[i])).detach() for i in range(self.N)]

        for _ in range(self.epochs):
            total_actor_loss=0.0
            total_entropy=0.0
            for i in range(self.N):
                probs=self.actor(states[i])
                entropy=-torch.sum(probs*probs.log(),dim=-1).mean()
                log_probs = torch.log(probs.gather(1, actions[i]))
                ratio = torch.exp(log_probs - old_log_probs[i])
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.eps,
                                    1 + self.eps) * advantage  # 截断
                actor_loss = torch.mean(-torch.min(surr1, surr2))  # PPO损失函数

                total_actor_loss+=actor_loss
                total_entropy_loss+=entropy
            actor_loss=total_actor_loss/self.N
            entropy=total_entropy/self.N
            critic_loss = torch.mean(
                F.mse_loss(self.critic(shared_states), td_target.detach()))
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            (actor_loss-0.01*entropy).backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()
            return actor_loss.item(),critic_loss.item(),entropy.item()

actor_lr = 5e-4
critic_lr = 5e-4
n_envs=4
num_episodes = 5000000//n_envs
hidden_dim = 64
gamma = 0.99
lmbda = 0.95
epochs = 15
eps = 0.2

batch_size=1000//n_envs
device = torch.device("cuda:3") 

torch.manual_seed(0)
state_dim = 18
action_dim = 5
N=3
agent = MAPPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, N,lmbda,
            epochs, eps, gamma, device)
# agent.load(0)
if __name__ =="__main__":
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.set_num_threads(2)
    np.random.seed(1)
    env=ParallelEnv(n_envs)
    return_list = train_on_policy_agent(env, agent, num_episodes,batch_size)
