
from utils import one_hot_to_index, one_hot_encode, extract_patches
import torch.nn.functional as F
import torch.nn as nn
import torch
from collections import deque
import random
import numpy as np

class GlobalConv(nn.Module):
    def __init__(self):
        super(GlobalConv, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(128 * 50 * 50, 256)  

    def forward(self, state):
        x = F.elu(self.conv1(state))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = x.view(x.size(0), -1) 
        x = self.fc(x) 
        return x

class ShapeRecognizer(torch.nn.Module):
    def __init__(self, num_shapes=5):
        super(ShapeRecognizer, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(64 * 10 * 10, num_shapes)  

    def get_patch_shapes(self, batch):
        patches = extract_patches(batch)  
        batch_size, num_patches, channels, height, width = patches.size()
        patches = patches.view(-1, channels, height, width) 

        x = self.logits(patches)
        x = torch.sigmoid(x)
        predictions = x.view(batch_size, num_patches, -1)  
        return predictions

    
    def logits(self, patch):
        patch = patch.view(-1, 3, 10, 10)
        x = F.elu(self.conv1(patch))
        x = F.elu(self.conv2(x))
        x = x.view(x.size(0), -1)  
        return self.fc(x) 
    
    def forward(self, patch, shape_label):
        x = self.logits(patch)
        x = torch.sigmoid(x)
        return torch.sum(x * shape_label, dim=1).clip(0, 1)

class MLP(nn.Module):
    def __init__(self, num_classes=5, hidden_sizes=[16]):
        super().__init__()
        
        layers = []
        
        # Input layer
        layers.append(nn.Linear(num_classes, hidden_sizes[0]))
        layers.append(nn.ReLU())
        
        # Hidden layers (if specified)
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            layers.append(nn.ReLU())
        
        # Output layer
        layers.append(nn.Linear(hidden_sizes[-1], 1)) 
        
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class Duel(nn.Module):
    def __init__(self, flat_size):
        super(Duel, self).__init__()

        self.streamV = nn.Sequential(
            nn.Linear(flat_size, 128),
            nn.ELU(),
            nn.Linear(128, 1)
        )

        self.streamA = nn.Sequential(
            nn.Linear(flat_size, 128),
            nn.ELU(),
            nn.Linear(128, 4)
        )

    def forward(self, x):
        value = self.streamV(x)
        advantage = self.streamA(x)

        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values

class SymDQN(nn.Module):
    def __init__(self, num_patches=25, max_shapes=5, num_actions=4, use_shape_recognizer=False, use_reward=False, use_ltn=False):
        super(SymDQN, self).__init__()
        self.global_conv = GlobalConv()
        self.flat_size = 256 

        self.use_shape_recognizer = use_shape_recognizer
        self.use_reward = use_reward

        if use_ltn or use_shape_recognizer or use_reward:
            self.shape_recognizer = ShapeRecognizer(max_shapes)

        if use_ltn or use_reward:
            self.reward_predictor = MLP(num_classes=max_shapes)

        if use_shape_recognizer:
            self.flat_size += num_patches * max_shapes

        if use_reward:
            self.flat_size += num_patches

        self.duel = Duel(self.flat_size)
        self.last_reward_loss = 1

    def forward(self, state):
        features = self.global_conv(state) 
        if self.use_shape_recognizer:
            patch_shapes = self.shape_recognizer.get_patch_shapes(state)  
            patch_shapes_flat = patch_shapes.transpose(-2, -1).contiguous().view(patch_shapes.size(0), -1)  
            
            features = torch.cat([features, patch_shapes_flat], dim=1) 

        if self.use_reward:
            patch_shapes = self.shape_recognizer.get_patch_shapes(state) 
            rewards = self.reward_predictor(patch_shapes)
            rewards_flat = rewards.view(rewards.size(0), -1) 
            features = torch.cat([features, rewards_flat], dim=1) 

        q_values = self.duel(features)
            
        return q_values

    def select_action(self, state):
        with torch.no_grad():
            Q = self.forward(state)
            action_index = torch.argmax(Q, dim=1)
        return action_index.item()

    def select_guided_action(self, state, actionRewards):

        indexes = [0, 1, 2, 3]
        
        if self.last_reward_loss < 0.01:
            actionRewards = torch.tensor(actionRewards)
            threshold = torch.max(actionRewards) - 0.3
            indexes = (actionRewards >= threshold).nonzero(as_tuple=True)[0].tolist()


        with torch.no_grad():
            best_q = float('-inf')
            best_a = indexes[0]
            Q = self.forward(state)[0]

            for i in indexes:
                if Q[i] > best_q:
                    best_q = Q[i]
                    best_a = i

        return best_a

class Memory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, next_states, actions, rewards, dones = zip(*batch)
        return (np.array(states, dtype=np.float32),
                np.array(next_states, dtype=np.float32),
                np.array(actions, dtype=np.int64),
                np.array(rewards, dtype=np.float32),
                np.array(dones, dtype=np.float32))

    def __len__(self):
        return len(self.buffer)
