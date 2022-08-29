from torch.optim.swa_utils import AveragedModel, SWALR
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils import bn_update


## 前期定义
loader, optimizer, model, loss_fn = ...
swa_model = AveragedModel(model)
scheduler = CosineAnnealingLR(optimizer, T_max=100)
swa_start = 5

## swa_lr 是最小学习率，从swa_start开始，
## anneal_strategy: linear or cos
## anneal_epochs: 周期数
swa_scheduler = SWALR(optimizer, swa_lr=0.05,anneal_strategy="linear", anneal_epochs=5)  

for epoch in range(100):
      for input, target in loader:
          optimizer.zero_grad()
          loss_fn(model(input), target).backward()
          optimizer.step()
      if epoch > swa_start:
          swa_model.update_parameters(model)
          swa_scheduler.step()
      else:
          scheduler.step()

# Update bn statistics for the swa_model at the end
update_bn(loader, swa_model)
# Use swa_model to make predictions on test data