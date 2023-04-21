import torch
from MNIST_dataloader import train_loader, test_loader
class ConvNet(torch.nn.Module):
  def __init__(self):
    super(ConvNet, self).__init__()
    self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)
    self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)
    self.mp = torch.nn.MaxPool2d(2)
    self.fc = torch.nn.Linear(320, 10)
  def forward(self, x):
    in_size = x.size(0)
    x = torch.nn.functional.relu(self.mp(self.conv1(x)))
    x = torch.nn.functional.relu(self.mp(self.conv2(x)))
    x = x.view(in_size, -1)
    x = self.fc(x)
    return torch.nn.functional.log_softmax(x)

model = ConvNet()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        #data, target = torch.Tensor(data), torch.Tensor(target)
        optimizer.zero_grad()
        output = model(data)
        loss = torch.nn.functional.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = torch.Tensor(data), torch.Tensor(target)
        output = model(data)
        # sum up batch loss
        test_loss += torch.nn.functional.nll_loss(output, target, size_average=False).data
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


for epoch in range(1, 10):
    train(epoch)
    test()