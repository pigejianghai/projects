import torch
from torchmetrics.classification import BinaryROC, MulticlassROC, MultilabelROC

# p = lambda: torch.rand(20)
# t = lambda: torch.randint(2, (20,))

# metric = BinaryROC()
# metric.update(p(), t())
# metric.compute()
# fig, ax = metric.plot()

p = lambda: torch.randn(200, 5)
t = lambda: torch.randint(5, (200,))

metric = MulticlassROC(5)
metric.update(p(), t())
fig, ax = metric.plot()

# p = lambda: torch.rand(20, 2)
# t = lambda: torch.randint(2, (20, 2))

# metric = MultilabelROC(2)
# metric.update(p(), t())
# print(p, t)
print(type(metric))
# print(fig, ax)