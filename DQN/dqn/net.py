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