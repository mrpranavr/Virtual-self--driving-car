# importing the libraries

import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

# Creating the architecture of the Neural Network

class Network(nn.Module):

    # Function to initialize or build the neural network
    def __init__(self, input_size, nb_action):
        super(Network, self).__init__()
        self.input_size = input_size
        self.nb_action = nb_action
        self.fc1 = nn.Linear(input_size, 30) # first hidden layer and connection(30 neurons)
        self.fc2 = nn.Linear(30, nb_action) # Second connection from hidden layer to output layer

    # This function is used to activate the neural network. Returns the Q-values
    def forward(self, state):
        x = F.relu(self.fc1(state)) # x is first hidden layer. relu is rectifier function
        q_values = self.fc2(x)
        return q_values
    
# Implementing Experience Replay

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    # Push function adds n samples into memory list and makes sure to have only n samples
    def push(self, event):
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    # Function to take samples from the memory
    def sample(self, batch_size):
        # if list = ((1,2,3),(4,5,6)), then zip(*list) = ((1,4),(2,3),(4,5))
        samples = zip(*random.sample(self.memory, batch_size))
        return map(lambda x: Variable(torch.cat(x, 0)), samples)
        # converting samples into torch variables - I GUESS

# Implementing the Deep Q-Learning algorithm

class Dqn():
    # input_size, nb_action because we create object of Network class through this class
    def __init__(self, input_size, nb_action, gamma): # Gamma is the delay coefficient
        self.gamma = gamma
        self.reward_window = []
        self.model = Network(input_size, nb_action) # Creating neural network
        self.memory = ReplayMemory(100000) # Assigning the capacity of memory
        self.optimizer = optim.Adam(self.model.parameters(), lr= 0.001) # Creating optimizer
        self.last_state = torch.Tensor(input_size).unsqueeze(0)
        # We create tensor and fake dimension corresponding to batch.
        # torch.Tensor creates class and inside input size of tensor.
        # .unsqueeze(0) creates fake dimension having first dimension ie STATE.
        self.last_action = 0
        self.last_reward = 0

    # Function to select right action at right time
    # State here is a torch tensor like the self.last_state we mentioned above
    def select_action(self, state): # State here is input state. Because output of neural network depends on input
        probs = F.softmax(self.model(Variable(state, volatile = True))*250) # Temperature parameter
        # Higher the temp -para more sure the neural network to choose next value
        action = probs.multinomial()
        return action.data[0,0] # our needed result ie, 0,1,2 is at index [0,0]

    # Learn function
    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1) # (state, action, reward) -> (0,1,2) dimensions
        next_outputs = self.model(batch_next_state).detach().max(1)[0]
        target = self.gamma * next_outputs + batch_reward 
        td_loss = F.smooth_l1_loss(outputs, target)
        self.optimizer.zero_grad() # Zero_grad will re-initialize the optimizer for each loop
        td_loss.backward(retain_variables = True) # Improve training/ Free the memory
        self.optimizer.step() # this will update the weights

    # Update function to update all elements of transition
    def update(self, reward, new_signal):
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
        self.memory.push((self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward])))
        # The new transition will contain previous state St , new state St+1 , last action played At and last reward Rt
        action = self.select_action(new_state)
        if len(self.memory.memory) > 100:
            batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(100) # this return a map object. thus splits it into each variable
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)
        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        self.reward_window.append(reward)
        if len(self.reward_window) > 1000:
            del self.reward_window[0]
        return action

    # Score function to compute score using reward_window
    def score(self):
        return sum(self.reward_window)/(len(self.reward_window) + 1.)

    # Save function to save the model
    def save(self):
        torch.save({'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    }, 'last_brain.pth')

    # Load function to load the saved model
    def load(self):
        if os.path.isfile('last_brain.pth'):
            print("=> loading checkpoint...")
            checkpoint = torch.load('last_brain.pth')
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print('done !')
        else:
            print('no checkpoint found...')

            







