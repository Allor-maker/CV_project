import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from torch.distributions import Categorical
import gymnasium
#from tqdm import tqdm

BATCH_SIZE = 32
LEARNING_RATE = 1e-3
NUM_CLASSES = 10
IMG_SIZE = 32  

def get_dataloader(dataset, indices):
    subset = Subset(dataset, indices)
    return DataLoader(subset, batch_size=BATCH_SIZE, shuffle=True)

class Flatten(nn.Module):
    def forward(self, batch):
        return batch.view(batch.size(0), -1)
    
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.cv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.cv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = Flatten()
        self.fc1 = nn.Linear(64 * IMG_SIZE * IMG_SIZE, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = F.relu(self.cv1(x))
        x = F.relu(self.cv2(x))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class DataPreloading():
    def __init__(self):
        self.train_data = datasets.CIFAR10("./root", train=True, download=True, transform=transforms.ToTensor())
        self.test_data = datasets.CIFAR10("./root", train=False, download=True, transform=transforms.ToTensor())
    
    def get_train_data(self):
        return self.train_data
    
class DataSelectionEnv(gymnasium.Env):
    def __init__(self):
        super(DataSelectionEnv, self).__init__()
        self.train_data = DataPreloading().get_train_data()
        self.num_class = NUM_CLASSES
        self.model = SimpleCNN()
        self.criterion = nn.CrossEntropyLoss()
        self.total_num_img = 5000
        self.indexes = self.class_select()
        self.optim = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        
        # Пространство действий: вероятности выбора изображений для каждого класса
        self.action_space = gymnasium.spaces.Box(low=0, high=1, shape=(NUM_CLASSES,), dtype=np.float32)
        self.observation_space = gymnasium.spaces.Box(low=0, high=1, shape=(NUM_CLASSES,), dtype=np.float32)
    
    def class_select(self):
        ls = {i: [] for i in range(self.num_class)}
        for x, (_, label) in enumerate(self.train_data):
            ls[label].append(x)
        return ls
    
    def sample(self, action):
        action = F.softmax(torch.tensor(action), dim=-1).numpy()
        index = []
        for i in range(self.num_class):
            num_img = int(action[i] * self.total_num_img)
            indexes = np.random.choice(self.indexes[i], num_img, replace=True)
            index.extend(indexes)
        return Subset(self.train_data, index)
    
    def step(self, action):
        indexes = self.sample(action)
        train_dataloader = get_dataloader(self.train_data, indexes)
        prev_acc = self.evaluate()
        self.train_model(train_dataloader)
        new_acc = self.evaluate()
        reward = new_acc - prev_acc
        return indexes, reward
    
    def train_model(self, dataloader):
        
        for images, labels in dataloader:
            self.optim.zero_grad()
            output = self.model(images)
            loss = self.criterion(output, labels)
            loss.backward()
            self.optim.step()
    
    def evaluate(self):
        return random.uniform(0, 1)  # Заглушка

class Actor(nn.Module):
    def __init__(self, input_size, num_actions):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, num_actions)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=-1)
        return x
    
class Critic(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 1)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

def actor_critic(env, actor, critic, episodes, max_steps=1000, gamma=0.99, lr_actor=1e-3, lr_critic=1e-3):
    optimizer_actor = optim.AdamW(actor.parameters(), lr=lr_actor)
    optimizer_critic = optim.AdamW(critic.parameters(), lr=lr_critic)
    
    for episode in range(episodes):
        state = np.ones(NUM_CLASSES) / NUM_CLASSES  # Начальное распределение
        step = 0
        while step < max_steps:
            step += 1
            state_tensor = torch.FloatTensor(state)
            action_probabilities = actor(state_tensor)
            dist = Categorical(action_probabilities)
            action = dist.sample().numpy()
            
            next_state, reward = env.step(action)
            next_state_tensor = torch.FloatTensor(next_state)
            value = critic(state_tensor)
            next_value = critic(next_state_tensor)
            
            advantage = reward + (gamma * next_value.item()) - value.item()
            loss_critic = advantage ** 2
            loss_actor = -dist.log_prob(torch.tensor(action)) * advantage
            
            optimizer_critic.zero_grad()
            loss_critic.backward()
            optimizer_critic.step()
            
            if step % 10 == 0:
                optimizer_actor.zero_grad()
                loss_actor.backward()
                optimizer_actor.step()
    return
