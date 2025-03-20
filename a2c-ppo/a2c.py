import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
from tqdm.notebook import tqdm
from torchvision import datasets,models,transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import gymnasium
import gym
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Subset
from torch.distributions import Categorical

BATCH_SIZE = 32
LEARNING_RATE = 1e-3
NUM_CLASSES = 10
IMG_SIZE = 32  

def get_dataloader(dataset, indices):
    subset = Subset(dataset, indices)
    return DataLoader(subset, batch_size=BATCH_SIZE, shuffle=True,transforms=transforms.ToTensor())


def train(net,n_epoch = 2): 
    loss_fn=nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(net.parameters(),lr=LEARNING_RATE)
    for epoch in tqdm(range(n_epoch)):

        running_loss = 0.0
        batchiter = iter(train_loader)
        for i,batch in enumerate(tqdm(batchiter)):
            x_batch,y_batch = batch

            optimizer.zero_grad()
            y_pred = net(x_batch)

            loss = loss_fn(y_pred,y_batch)

            loss.backward()
            optimizer.step()
            running_loss+=loss.item()
            if i % 500 == 499:
              print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 500))
              running_loss = 0.0
    print("обучение закончено")
    return net


class Flatten(nn.Module):
    def forward(self,batch):
        return batch.view(batch.size(0),-1)
    
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
        x = F.softmax(self.fc2(x))
        return x


class DataPreloading():
    def __init__(self):
        #поменять на нужные данные
        self.train_data = datasets.CIFAR10("./root", train=True,download=True,transform=transforms.ToTensor())
        self.test_data = datasets.CIFAR10("./root",train=False,download=True,transform=transforms.ToTensor())

    def get_train_data(self):
        return self.train_data
    
class DataSelectionEnv(gym.Env):
    def __init__(self):
        super(DataSelectionEnv, self).__init__()
        
        #двумерный массив с индексами изображений по классам
        self.indexes = self.class_select()
        self.num_class = 10
        self.model = SimpleCNN()
        self.entropy = nn.CrossEntropyLoss()
        total_num_img = 5000

    def sample(self,action):
        index = []
        for i in range(self.num_class):
            num_img = int(action[i]*self.total_num_img)#на основе распределения вероятностей из action рассчитали количество изобьражений в каждом классе
            indexes = np.random.choice(self.class_select[i],num_img,replace=True)#рандомно взяли индексы из списка индексов по классам в нужном количестве 
            index.extend(indexes)
        return Subset(self.train_data,index)#взяли подмножество из train_data по индексам изображений
    
    def class_select(self):
        ls = {i: [] for i in self.num_class}
        for x, (_,label) in enumerate(DataPreloading.get_train_data()):
            ls[label].append(x)
        return ls
    
    def step(self,action):
        indexes = self.sample(action)
        train_dataloader = get_dataloader(DataPreloading.get_train_data(),indexes)
        reward = self.check(train_dataloader) #определение награды
        return indexes, reward, _ , _ #дописать стадию done
    
    def check(self,dataloader):
        for images,labels in dataloader:
            output = self.model(images)
            #дописать определение награды

        
class Actor(nn.Module):
    def __init__(self, input_size, num_actions):
        super().__init__()
        self.fc1 = nn.Linear(input_size,64)
        self.fc2 = nn.Linear(64,num_actions)
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x),dim=-1)
        return x
    
class Critic(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size,64)
        self.fc2 = nn.Linear(64,1)
    def forward(self,x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)




def actor_critic(env,actor, critic, episodes, max_steps=1000, gamma=0.99, lr_actor=1e-3, lr_critic=1e-3):
    optimizer_actor = optim.AdamW(actor.parameters(), lr=lr_actor)
    optimizer_critic = optim.AdamW(critic.parameters(), lr=lr_critic)

    for episode in range(episodes):
        state = env.state
        done = False

        while not done:
            state_tensor = torch.FloatTensor(state)
            action_probabilities = actor(state_tensor)
            dist = Categorical(action_probabilities)  #создаем дискретное распределение
            action = dist.sample().item()  #выбираем случайным образом на основе вероятностей (PPO) действие

            next_state, reward, done = env.step(action)
            next_state_tensor = torch.FloatTensor(next_state)
            value = critic(state_tensor)
            next_value = critic(next_state_tensor)
            
            Adv = reward + (gamma * next_value) - value 
            loss_critic = Adv ** 2
            loss_actor = 1  #подставить функциб от Adv

            optimizer_actor.zero_grad()
            loss_actor.backward()
            optimizer_actor.step()

            optimizer_critic.zero_grad()
            loss_critic.backward()
            optimizer_critic.step()

    return 