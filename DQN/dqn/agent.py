class DQNagent:
    def _init_(
        self,
        state_space,
        agent_space,
        eps = 1,
        eps_min = 0.01,
        eps_decay = 5e-4,
        lr = 1e-3,
        gamma = 0.99,
        buffer_num = 10000
    )

        self.q_network = Net(obervation_dim, action_dim)
        self.action_network = Net(obervation_dim, action_dim)
        self.mse = nn.MSELoss()

        self.eps = eps
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        self.lr = lr
        self.gamma = gamma
        self.buffer = Replaybuffer(buffer_num)
        self.q_network = Net()

    def get_action(self, observation):
        q_value = self.q_network(observation)
        self.Qmax = np.argmax(q_value)
        self.last_Qmax =  
        x = np.random.random()
        if x >= eps:
            return self.Qmax
        else:
            return np.random.choice(len(q_value))
    
    def act(self, obs, reward, done):
        action = self.get_action(obs)
        if self.last_obs and self.last_action and self.last_reward and self.last_done:
            transition = (self.last_obs, self.last_action, self.last_reward, obs, self.last_done)
        # hold some information
        self.last_obs = obs
        self.last_action = action
        self.last_reward = reward
        self.last_done = done
        # insert transition
        self.buffer.insert(transition)

        return action
    
    def episode_start(self):
        self.last_obs = self.last_action = self.last_reward = self.last_done = None
    
    def learn(self):
        # sample some transitions from replay buffer
        sample = self.buffer.sample(100)
        
        # calculate TDError
        q_now = self.q_network(sample[0])[sample[1]]
        q_next = torch.max(self.q_network(sample[3]))

        # calculate loss
        loss = self.mse(self.gamma* q_next + sample[2], q_now)

        # backpropagate loss
        loss.backward()

        return loss


            


class Net(nn.Module):

    def __init__(self,obeservation_dim, action_dim):
        super(Net, self).__init__()
        self.activation = nn.ReLU()
        self.first_layer = nn.Layer(observation_dim,256)
        self.second_layer = nn.Layer(256,256)
        self.final_layer = nn.Layer(256,action_dim)

    def forward(self, observation):
        first_output = self.activation(self.first_layer(observation))
        second_output = self.activation(self.second_layer(first_output))
        final_output = self.activation(self.final_layer(second_output))
        return final_output

class Replaybuffer:
    def _init_(self, max_sixe)
        self.max_size = max.size
        self.memory = []
        self.inedx = 0

    def insert(self, transition)# action, observation, reward, next_observation, done 
        if len(memory) < max_size:
            self.memory.append(transition)
        else:
            self.memory[self.index] = transition

        self.index = (self.index + 1) % self.max_size
        
        return self.memory[self.index]

    def sample(self, size)
        indice = np.random.choice(range(len(memory)), size)
        transition = zip(*self.memory[i] for i in indice )
        return transition


