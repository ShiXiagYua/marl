import torch
import gym
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter
import os
from tqdm import tqdm
import copy
from multiprocessing import Process, Pipe
from pettingzoo.mpe import simple_spread_v3
os.environ['SDL_VIDEODRIVER'] = 'dummy'
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
class Replay:
    def __init__(self,min_size,max_size,batch_size):
        self.max_size=max_size
        self.min_size=min_size
        self.buffer=[] #num_episode episode_len num_agent dim
        self.batch_size=batch_size
    def add(self,states,prev_actions,actions,rewards,next_states,dones):
        for i in range(len(rewards)):
            self.buffer.append((states[i],prev_actions[i],actions[i],rewards[i],next_states[i],dones[i]))
    def sample(self):
        transitions=random.sample(self.buffer,self.batch_size)
        states,prev_actions,actions,rewards,next_states,dones=zip(*transitions)
        return states,prev_actions,actions,rewards,next_states,dones
    def __len__(self):
        return len(self.buffer)

class RNN(nn.Module):
    # Because all the agents share the same network, input_shape=obs_shape+n_actions+n_agents
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(RNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        # self.rnn = nn.GRU(hidden_dim, hidden_dim)
        self.norm1=nn.LayerNorm(hidden_dim)
        self.fc2=nn.Linear(hidden_dim, hidden_dim)
        self.norm2=nn.LayerNorm(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, inputs, hidden_state=None):
        #time bs dim
        # x = F.relu(self.fc1(inputs))

        # x , hidden= self.rnn(x, hidden_state)
        # qs = self.fc2(x)
        # return qs, hidden
        x=self.norm1(F.relu(self.fc1(inputs)))
        x=self.norm2(F.relu(self.fc2(x)))
        return self.fc3(x),None
class Qmix(nn.Module):
    def __init__(self,state_dim,hidden_dim,num_agent):
        super(Qmix,self).__init__()
        self.hyper_w1 = nn.Sequential(nn.Linear(state_dim, hidden_dim ),nn.ReLU(),nn.Linear(hidden_dim , num_agent))
        self.hyper_b1 = nn.Sequential(nn.Linear(state_dim,hidden_dim),nn.ReLU(),nn.Linear(hidden_dim,1))
        self.trans_fn=nn.Softplus(beta=1, threshold=20)
    def forward(self,qs,states):#states: num_episode  episode_len state_dim ; qs: num_episode episode_len num_agent 
        weight=self.trans_fn(self.hyper_w1(states))  #num_episode  episode_len  num_agent
        bias=self.hyper_b1(states)          #num_episode  episode_len 1
        return torch.sum(weight*qs,dim=-1,keepdim=True)+bias

class QmixAgent:
    def __init__(self,state_dim,action_dim,hidden_dim,num_agent,num_env,q_net_lr,qmix_lr,gamma,explore_rate,explore_rate_decay,min_explore_rate,update_gap,device):
        self.q_net=RNN(state_dim+action_dim,hidden_dim,action_dim).to(device)#只有一个动作，多个动作state_dim+num*action_dim
        self.qmix=Qmix(state_dim*num_agent,hidden_dim,num_agent).to(device)
        self.target_q_net=copy.deepcopy(self.q_net)
        self.target_qmix=copy.deepcopy(self.qmix)
        self.q_net_optimizer=torch.optim.Adam(self.q_net.parameters(),lr=q_net_lr)
        self.qmix_optimizer=torch.optim.Adam(self.qmix.parameters(),lr=qmix_lr)
        self.action_dim=action_dim
        self.hidden_dim=hidden_dim
        self.num_agent=num_agent
        self.num_env=num_env
        self.explore_rate=explore_rate
        self.explore_rate_decay=explore_rate_decay
        self.min_explore_rate=min_explore_rate
        self.device=device
        self.episode_len=25
        self.gamma=gamma
        self.update_step=0
        self.update_gap=update_gap
    def generate_hidden_state(self):
        return torch.zeros(1,self.num_env*self.num_agent,self.hidden_dim).to(self.device)
    def tdv(self,x):
        return torch.tensor(np.array(x),dtype=torch.float32).to(self.device)
    def generate_inputs(self,states,prev_actions):
        one_hot_action=F.one_hot(prev_actions.to(torch.long).squeeze(-1),num_classes=self.action_dim).to(torch.float32)
        return torch.cat([states,one_hot_action],dim=-1)
    def take_action(self,states,prev_action,prev_hidden_state):
        #num_env num_ag dim -> num_env num_ag
        with torch.no_grad():
            #states: num_envs num_agent state_dim
            states=self.tdv(states).reshape(1,self.num_env*self.num_agent,-1)
            prev_action=self.tdv(prev_action).reshape(1,self.num_env*self.num_agent,-1)
            inputs=self.generate_inputs(states,prev_action)
            qs,hs=self.q_net(inputs,prev_hidden_state)
            if np.random.rand()<self.explore_rate:
                return np.random.randint(0,self.action_dim,size=(self.num_env,self.num_agent)),hs
            actions=qs.argmax(dim=-1).reshape(self.num_env,self.num_agent) #num_envs num_agent
            return actions.cpu().numpy(),hs
    def update(self, states,prev_actions,actions,rewards,next_states,dones):
        #bs e_l n_a d->e_l bs n_a d -> e_l bs*n_a d
        bs=len(rewards)
        states=self.tdv(states).transpose(0,1).flatten(1,2)
        prev_actions=self.tdv(prev_actions).unsqueeze(-1).transpose(0,1).flatten(1,2)
        actions=self.tdv(actions).unsqueeze(-1).transpose(0,1).flatten(1,2).to(torch.long)
        next_states=self.tdv(next_states).transpose(0,1).flatten(1,2)

        #bs e_l n_a ->e_l bs n_a -> e_l bs 1
        rewards=self.tdv(rewards).transpose(0,1).sum(dim=-1,keepdim=True)
        dones=self.tdv(dones).transpose(0,1)[:,:,:1]
        #e_l bs*n_a d -> el bs d*
        shared_states=states.reshape(self.episode_len,bs,-1)
        shared_next_states=next_states.reshape(self.episode_len,bs,-1)

        #e_l bs*n_a d => el bs*na d_a
        qs,_=self.q_net(self.generate_inputs(states,prev_actions)) 
        next_qs,_=self.target_q_net(self.generate_inputs(next_states,actions))
        next_qs_,_=self.q_net(self.generate_inputs(next_states,actions))

        #el bs*na d_a -> el bs*na 1 -> el bs na
        qs=qs.gather(-1,actions).reshape(self.episode_len,bs,self.num_agent)
        next_qs=next_qs.gather(-1,next_qs_.argmax(dim=-1,keepdim=True)).reshape(self.episode_len,bs,self.num_agent)
        
        #e_l bs d -> e_l bs 1
        values=self.qmix(qs,shared_states)
        next_values=self.target_qmix(next_qs,shared_next_states)

        target_values=rewards+self.gamma*next_values*(1-dones)
        loss=F.mse_loss(values,target_values.detach())
        self.q_net_optimizer.zero_grad()
        self.qmix_optimizer.zero_grad()
        loss.backward()
        # nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        # nn.utils.clip_grad_norm_(self.qmix.parameters(), 10.0)
        self.q_net_optimizer.step()
        self.qmix_optimizer.step()
        self.update_step+=1
        if self.update_step%self.update_gap==0:
            self.copy_target()
        self.explore_rate*=self.explore_rate_decay
        self.explore_rate=max(self.explore_rate,self.min_explore_rate)
        return loss.item(),self.explore_rate
    def copy_target(self):
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.target_qmix.load_state_dict(self.qmix.state_dict())
    def save(self,i):
        torch.save(self.q_net.state_dict(),'models/q_net%d'%i)
        torch.save(self.qmix.state_dict(),'models/qmix%d'%i)
def train_off_policy_agent(env, agent, replay,num_episodes,update_iter):
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
    i_eps=0
    loss=0.0
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                state = env.reset()
                prev_action=env.sample_action()
                prev_hidden_state=None
                done = False
                transition_dict = {'states': [], 'prev_actions':[],'actions': [], 'next_states': [], 'rewards': [], 'dones': []}#ep_l num_e num_a dim
                while True:
                    action ,hidden_state= agent.take_action(state,prev_action,prev_hidden_state)
                    next_state, reward, done,truct, _ = env.step(action)
                    transition_dict['states'].append(state)
                    transition_dict['prev_actions'].append(prev_action)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(truct)
                    state = next_state
                    prev_action=action
                    prev_hidden_state=hidden_state
                    episode_return += np.array(reward).sum()/len(reward)
                    if done[0][0] or truct[0][0]:
                        break
                for key,value in transition_dict.items():
                    transition_dict[key]=np.array(value).swapaxes(0,1)  #episode_len num_env num_agent dim->num_env episode_len num_agent dim
                replay.add(transition_dict['states'],transition_dict['prev_actions'],transition_dict['actions'],transition_dict['rewards'],transition_dict['next_states'],transition_dict['dones'])
                if len(replay)>=replay.min_size:
                    for _ in range(update_iter):
                        states,prev_actions,actions,rewards,next_states,dones=replay.sample()
                        loss,explore_rate=agent.update(states,prev_actions,actions,rewards,next_states,dones)
                    writer.add_scalar('loss',loss,i_eps)
                    writer.add_scalar('explore_rate',explore_rate,i_eps)
                    writer.add_scalar('individual_reward',np.array(transition_dict['rewards']).mean(),i_eps)
                return_list.append(episode_return)
                writer.add_scalar('rewards',episode_return,i_eps)
                i_eps+=1
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                                      'return': '%.3f' % np.mean(return_list[-10:]),'lr': '%.3f'%agent.explore_rate,'loss':'%.3f' %loss})
                pbar.update(1)
        agent.save(i)
    return return_list
min_size=100
max_size=100000
batch_size=64

state_dim = 18
action_dim = 5
hidden_dim=128
num_agent=3
num_env=20

q_net_lr = 5e-4
qmix_lr=5e-5
gamma = 0.99
explore_rate=1.0
explore_rate_decay=0.99995
min_explore_rate=0.01
update_gap=100
device = torch.device("cuda:0") 

num_episodes = 2000000//num_env
update_iter=1

# agent.load(0)
if __name__ =="__main__":
    
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    np.random.seed(1)
    replay=Replay(min_size,max_size,batch_size)
    env=ParallelEnv(num_env)
    agent = QmixAgent(state_dim,action_dim,hidden_dim,num_agent,num_env,q_net_lr,qmix_lr,gamma,explore_rate,explore_rate_decay,min_explore_rate,update_gap,device)

    return_list = train_off_policy_agent(env, agent, replay,num_episodes,update_iter)
