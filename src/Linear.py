# Using 2.0.0+cu118
import torch
print(torch.__version__)

x_data = torch.tensor([[1.0], [2.0], [3.0]], requires_grad=False)
y_data = torch.tensor([[2.0], [4.0], [6.0]], requires_grad=False)

class Model(torch.nn.Module):
  def __init__(self):
    super(Model, self).__init__()
    self.linear = torch.nn.Linear(1, 1)
  def forward(self, x):
    y_pred = self.linear(x)
    return y_pred
model = Model()
criterion = torch.nn.MSELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


for epoch in range(500):
  y_pred = model(x_data)
  loss = criterion(y_pred, y_data)
  print(epoch, loss.data.item())
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

# print parameter learned values
for p in model.parameters():
  print(p)

# prediction
model.forward(torch.Tensor([[4.0]])).data[0]