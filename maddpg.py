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
from pettingzoo.mpe import simple_adversary_v3
class EnvWrapper:
    def __init__(self,visible=False):
        self.padding=np.zeros(2)
        if visible:
            self.env = simple_adversary_v3.parallel_env(render_mode="human",continuous_actions=True)
        else:
            self.env = simple_adversary_v3.parallel_env(continuous_actions=True)
    def process_obs(self,obs):
        obs['adversary_0']=np.concatenate([self.padding,obs['adversary_0']],axis=0)
        return obs
    def reset(self,seed=None):
        obs,infos=self.env.reset(seed=seed)
        obs=self.process_obs(obs)
        return self.dict_to_list(obs),self.dict_to_list(infos)
    def step(self,action):
        #action -1,1 -> 0,1
        action=action.astype(np.float32)#3,2
        full_action=np.zeros((3,5)).astype(np.float32)
        full_action[:,1]=-action[:,0]
        full_action[:,2]=action[:,0]
        full_action[:,3]=-action[:,1]
        full_action[:,4]=action[:,1]
        full_action=np.where(full_action<0,0,full_action)
        obs,rewards,dones,tructs,infos=self.env.step(self.list_to_dict(full_action))
        obs=self.process_obs(obs)
        return self.dict_to_list(obs),self.dict_to_list(rewards),self.dict_to_list(dones),self.dict_to_list(tructs),self.dict_to_list(infos)
    def list_to_dict(self,infos):
        result={}
        result['adversary_0']=infos[0]
        result['agent_0']=infos[1]
        result['agent_1']=infos[2]
        return result
    def dict_to_list(self,infos):
        result=[infos['adversary_0'],infos['agent_0'],infos['agent_1']]
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
    def add(self,states,actions,rewards,next_states,dones):
        #num_env episode_len num_agent dim
        for i in range(len(rewards)):
            self.buffer.append((states[i],actions[i],rewards[i],next_states[i],dones[i]))
    def sample(self):
        transitions=random.sample(self.buffer,self.batch_size)
        states,actions,rewards,next_states,dones=zip(*transitions)
        return states,actions,rewards,next_states,dones
    def __len__(self):
        return len(self.buffer)

class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.layer1 = nn.Sequential(torch.nn.Linear(state_dim, hidden_dim),nn.ReLU(),nn.LayerNorm(hidden_dim))
        self.layer2= nn.Sequential(torch.nn.Linear(hidden_dim,hidden_dim),nn.ReLU(),nn.LayerNorm(hidden_dim))
        self.fc = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return F.tanh(self.fc(x))


class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.layer1 = nn.Sequential(torch.nn.Linear(state_dim, hidden_dim),nn.ReLU(),nn.LayerNorm(hidden_dim))
        self.layer2= nn.Sequential(torch.nn.Linear(hidden_dim,hidden_dim),nn.ReLU(),nn.LayerNorm(hidden_dim))  
        self.fc = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return self.fc(x)
class DDPG:
    def __init__(self,state_dim,hidden_dim,action_dim,agent_id,num_agent,num_env,actor_lr,critic_lr,gamma,explore_rate,explore_rate_decay,min_explore_rate,update_gap,device):
        self.actor=PolicyNet(state_dim,hidden_dim,action_dim).to(device)#只有一个动作，多个动作state_dim+num*action_dim
        self.critic=ValueNet((state_dim+action_dim)*num_agent,hidden_dim).to(device)
        self.target_actor=copy.deepcopy(self.actor)
        self.target_critic=copy.deepcopy(self.critic)
        self.actor_optimizer=torch.optim.Adam(self.actor.parameters(),lr=actor_lr)
        self.critic_optimizer=torch.optim.Adam(self.critic.parameters(),lr=critic_lr)
        self.device=device
        self.gamma=gamma
        self.agent_id=agent_id
        self.update_step=0
        self.update_gap=update_gap
        self.num_env=num_env
        self.action_dim=action_dim
        self.explore_rate=explore_rate
        self.explore_rate_decay=explore_rate_decay
        self.min_explore_rate=min_explore_rate
    def take_action(self,states):
        if np.random.rand()<self.explore_rate:
            return np.random.uniform(-1,1,size=(self.num_env,self.action_dim))
        with torch.no_grad():
            #num_envs state_dim-> num_envs action_dim
            actions=self.actor(states)
            return actions.cpu().numpy()
    def update(self, states,shared_states,actions,online_actions,rewards,shared_next_states,online_next_actions,dones):
        #shared_states 包含所有agent的动作和状态
        #shared_next_states 包含所有agent的next状态和他们会采取的动作
        #actions 是一个list, 包含一个元素，是所有actions拼接后结果
        #online_actions 是一个list,包含每个agent当前将采取的动作 bs action_dim detached
        #bs d->bs 1

        qs=self.critic(torch.cat([shared_states,actions],-1))
        next_qs=self.target_critic(torch.cat([shared_next_states,online_next_actions],-1))
        #bs 1
        target_qs=rewards+self.gamma*next_qs*(1-dones)
        critic_loss=F.mse_loss(qs,target_qs.detach())
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
        self.critic_optimizer.step()

        online_actions=copy.deepcopy(online_actions)
        online_actions[self.agent_id]=self.actor(states)
        online_actions=torch.cat(online_actions,-1)
        actor_loss=-torch.mean(self.critic(torch.cat([shared_states,online_actions],-1)))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
        self.actor_optimizer.step()
        self.update_step+=1
        if self.update_step%self.update_gap==0:
            self.copy_target()
        self.explore_rate*=self.explore_rate_decay
        self.explore_rate=max(self.explore_rate,self.min_explore_rate)
        return actor_loss.item(),critic_loss.item()
    def copy_target(self):
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
    def save(self,prefix):
        torch.save(self.actor.state_dict(),"models/"+str(prefix)+"_actor.pth")
        torch.save(self.critic.state_dict(),"models/"+str(prefix)+"_critic.pth")
    def load(self,prefix):
        self.actor.load_state_dict(torch.load("models/"+str(prefix)+"_actor.pth"))
        self.critic.load_state_dict(torch.load("models/"+str(prefix)+"_critic.pth"))
        self.copy_target()
class MADDPG:
    def __init__(self,state_dim,hidden_dim,action_dim,num_agent,num_env,actor_lr,critic_lr,gamma,explore_rate,explore_rate_decay,min_explore_rate,update_gap,device):
        self.agents=[DDPG(state_dim,hidden_dim,action_dim,agent_id,num_agent,num_env,actor_lr,critic_lr,
                    gamma,explore_rate,explore_rate_decay,min_explore_rate,update_gap,device)for agent_id in range(num_agent)]
        self.num_agent=num_agent
        self.device=device
        self.build_writer()
    def build_writer(self):
        os.makedirs('logs',exist_ok=True)
        exist_exp_id=[int(exp) for exp in os.listdir("logs")]
        if len(exist_exp_id)>0:
            exist_exp_id=sorted(exist_exp_id)
            exp_id=exist_exp_id[-1]+1
        else:
            exp_id=0
        exp_dir="logs/%d"%exp_id
        os.makedirs(exp_dir)
        ad_dir=exp_dir+'/adversary'
        f1_dir=exp_dir+'/friend1'
        f2_dir=exp_dir+'/friend2'
        os.makedirs(ad_dir)
        os.makedirs(f1_dir)
        os.makedirs(f2_dir)
        self.writers=[SummaryWriter(ad_dir),SummaryWriter(f1_dir),SummaryWriter(f2_dir)]
    def tdv(self,x):
        return torch.tensor(np.array(x),dtype=torch.float32).to(self.device)
    def save(self,i):
        for j in range(self.num_agent):
            self.agents[j].save("{}_{}".format(i,j))
    def load(self,i):
        for j in range(self.num_agent):
            self.agents[j].load("{}_{}".format(i,j))
    def take_action(self,states):
        #num_env num_agent dim
        #num_agent num_env action_dim 
        states=self.tdv(states)
        actions=[]
        for i in range(self.num_agent):
            action=self.agents[i].take_action(states[:,i,:])
            actions.append(action)
        return np.array(actions).swapaxes(0,1)
    def update(self,states,actions,rewards,next_states,dones,i_eps):
        #bs num_agent dim
        bs=len(rewards)
        states=self.tdv(states)# bs n_a dim
        next_states=self.tdv(next_states)#bs n_a dim
        shared_states=states.reshape(bs,-1)#bs d*
        shared_next_states=next_states.reshape(bs,-1)#bs d*
        actions=self.tdv(actions).reshape(bs,-1)#bs n_a*d_a
        rewards=self.tdv(rewards).unsqueeze(-1)#bs n_a 1
        dones=self.tdv(dones).unsqueeze(-1)#bs n_a 1
        online_actions=[]#n_a bs d_a
        online_next_actions=[]#n_a bs d_a
        for i in range(self.num_agent):
            online_actions.append(self.agents[i].target_actor(states[:,i,:]).detach())
            online_next_actions.append(self.agents[i].target_actor(next_states[:,i,:]).detach())
        online_next_actions=torch.cat(online_next_actions,-1)
        losses=[]
        for i in range(self.num_agent):
            # if i==0:
            #     continue
            actor_loss,critic_loss=self.agents[i].update(states[:,i,:],shared_states,actions,online_actions,rewards[:,i,:],shared_next_states,online_next_actions,dones[:,i,:])
            writer=self.writers[i]
            writer.add_scalar('actor_loss',actor_loss,i_eps)
            writer.add_scalar('critic_loss',critic_loss,i_eps)
            writer.add_scalar('explore_rate',self.agents[i].explore_rate,i_eps)

def train_off_policy_agent(env, agent, replay,num_episodes,update_iter):
    return_list = []
    adversary_return_list = []
    i_eps=0
    loss=0.0
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                state = env.reset()
                done = False
                while True:
                    action = agent.take_action(state)
                    next_state, reward, done,truct, _ = env.step(action)
                    replay.add(state,action,reward,next_state,truct)
                    state = next_state
                    episode_return += np.array(reward).mean(axis=0)
                    if done[0][0] or truct[0][0]:
                        break
                if len(replay)>=replay.min_size:
                    for _ in range(update_iter):
                        states,actions,rewards,next_states,dones=replay.sample()
                        agent.update(states,actions,rewards,next_states,dones,i_eps)
                return_list.append(episode_return[1:].sum())
                adversary_return_list.append(episode_return[0].sum())
                i_eps+=1
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                                      'return': '%.3f' % np.mean(return_list[-10:]),'adversary_return': '%.3f' % np.mean(adversary_return_list[-10:])})
                pbar.update(1)
        agent.save(i)
    return return_list
min_size=2000
max_size=1000000
batch_size=1024

state_dim = 10
action_dim = 2
hidden_dim=128
num_agent=3
num_env=20

actor_lr = 5e-4
critic_lr=5e-4
gamma = 0.99
explore_rate=1.3
explore_rate_decay=0.99995
min_explore_rate=0.01
update_gap=100
device = torch.device("cuda:2") 

num_episodes = 2000000//num_env
update_iter=1

# agent.load(0)
if __name__ =="__main__":
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    np.random.seed(1)
    test=True
    if not test:
        os.environ['SDL_VIDEODRIVER'] = 'dummy'
        replay=Replay(min_size,max_size,batch_size)
        env=ParallelEnv(num_env)
        agent = MADDPG(state_dim,hidden_dim,action_dim,num_agent,num_env,actor_lr,critic_lr,gamma,explore_rate,explore_rate_decay,min_explore_rate,update_gap,device)

        return_list = train_off_policy_agent(env, agent, replay,num_episodes,update_iter)
    else:
        agent = MADDPG(state_dim,hidden_dim,action_dim,num_agent,num_env,actor_lr,critic_lr,gamma,explore_rate,explore_rate_decay,min_explore_rate,update_gap,device)
        agent.load(7)
        env=EnvWrapper(True)
        state=env.reset()[0]
        while True:
            action = agent.take_action([state])
            next_state, reward, done,truct, _ = env.step(action[0])
            state = next_state
            if done[0] or truct[0]:
                break
        
