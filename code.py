import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def get_cifar10():
    train_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )
    train_dataset = torchvision.datasets.CIFAR10(
        "./data/CIFAR10", train=True, transform=train_transform, download=True
    )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )
    test_dataset = torchvision.datasets.CIFAR10(
        "./data/CIFAR10", train=False, transform=test_transform, download=True
    )

    return train_dataset, test_dataset

def train_and_test(net, loader, optimiser):
    net.train()

    running_loss = []

    for inputs, labels in tqdm(loader):
        inputs = inputs.cuda()
        labels = labels.cuda()

        optimiser.zero_grad()

        outputs = net(inputs)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        optimiser.step()

        running_loss.append(loss.item())

    avg_loss = sum(running_loss) / len(running_loss)
    print("Training average loss: {}".format(avg_loss))

    net.eval()

    running_loss = 0.0
    running_correct = 0

    for inputs, labels in loader:
        with torch.no_grad():
            inputs = inputs.cuda()
            labels = labels.cuda()

            outputs = net(inputs)
            running_loss += F.cross_entropy(outputs, labels, reduction = "sum")

            outputs = outputs.max(1)[1]
            running_correct += outputs.eq(labels.view_as(outputs)).sum().item()

    avg_loss = running_loss / len(loader.dataset)
    percent_correct = running_correct * 100.0 / len(loader.dataset)
    print("Testing loss: {}, Accuracy: {}/10000 ({}%)".format(avg_loss, running_correct, percent_correct))

train_dataset, test_dataset = get_cifar10()

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=4, shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=4, shuffle=False
)

net = Net()
net = net.cuda()
optimiser = optim.SGD(net.parameters(), lr = 0.001, momentum = 0.9)

for epoch in range(2):
    print("Epoch: {}".format(epoch))
    train_and_test(net, train_loader, optimiser)

torch.save(net.state_dict(), "cifar_net.pth")
