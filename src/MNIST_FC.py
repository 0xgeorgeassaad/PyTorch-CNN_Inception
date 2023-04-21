import torch
import torchvision


train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./data', train=True, download=True,
                               transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Normalize((0.1307,),(0.3081,))])
                               ),
    batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./data', train=False, download=True,
                               transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Normalize((0.1307,),(0.3081,))])
                               ),
    batch_size=64, shuffle=True)

class Net(torch.nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.l1 = torch.nn.Linear(784, 520)
    self.l2 = torch.nn.Linear(520, 320)
    self.l3 = torch.nn.Linear(320, 120)
    self.l4 = torch.nn.Linear(120, 10)
  def forward(self, x):
    x = x.view(-1, 784)
    x = torch.nn.functional.relu(self.l1(x))
    x = torch.nn.functional.relu(self.l2(x))
    x = torch.nn.functional.relu(self.l3(x))
    return self.l4(x)

model = Net()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

def train(epoch):
  model.train()
  for batch_idx, (data, target) in enumerate(train_loader):
    data, target = torch.Tensor(data), torch.Tensor(target)
    output = model(data)
    loss = criterion(output, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if batch_idx % 10 == 0:
      print(f'Train Epoch {epoch} Loss: {loss.data.item()}')
def test():
  model.eval()
  test_loss = 0
  correct = 0
  for data, target in test_loader:
    data, target = torch.Tensor(data), torch.Tensor(target)
    output = model(data)
    test_loss = criterion(output, target).data.item()
    pred = torch.max(output.data, 1)[1]
    correct += pred.eq(target.data.view_as(pred)).cpu().sum()
  test_loss /= len(test_loader.dataset)
  print(f'Test Loss: {test_loss}')
  print(f'Acc: {correct / len(test_loader.dataset)}')

for epoch in range(1,10):
  train(epoch)
  test()