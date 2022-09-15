import torch
import copy
import syft as sy
from torch import nn
from torch import optim
hook = sy.TorchHook(torch)


# 工作机作为客户端，用于训练模型，安全工作机作为服务器，用于数据的整合及交流
Li = sy.VirtualWorker(hook, id='Li')
Zhang = sy.VirtualWorker(hook, id='Zhang')
secure_worker = sy.VirtualWorker(hook, id='secure_worker')
data = torch.tensor([[0, 1], [0, 1], [1, 0], [1, 1.]], requires_grad=True)
target = torch.tensor([[0], [0], [1], [1.]], requires_grad=True)

data_Li = data[0:2]
target_Li = target[0:2]
data_Zhang = data[2:]
target_Zhang = target[2:]
Li_data = data_Li.send(Li)
Zhang_data = data_Zhang.send(Zhang)
Li_target = target_Li.send(Li)
Zhang_target = target_Zhang.send(Zhang)


model = nn.Linear(2, 1)


# 定义迭代次数
iterations = 20
worker_iters = 5

for a_iter in range(iterations):
    Li_model = model.copy().send(Li)
    Zhang_model = model.copy().send(Zhang)
    # 定义优化器
    Li_opt = optim.SGD(params=Li_model.parameters(), lr=0.1)
    Zhang_opt = optim.SGD(params=Zhang_model.parameters(), lr=0.1)
    # 并行训练
    for wi in range(worker_iters):
        # 训练Li的模型
        Li_opt.zero_grad()
        Li_pred = Li_model(Li_data)
        Li_loss = ((Li_pred - Li_target) ** 2).sum()
        Li_loss.backward()
        Li_opt.step()
        Li_loss = Li_loss.get().data
        # 训练Zhang的模型
        Zhang_opt.zero_grad()
        Zhang_pred = Zhang_model(Zhang_data)
        Zhang_loss = ((Zhang_pred - Zhang_target) ** 2).sum()
        Zhang_loss.backward()
        Zhang_opt.step()
        Zhang_loss = Zhang_loss.get().data
    # 将更新的模型发送至安全工作机
    Zhang_model.move(secure_worker)
    Li_model.move(secure_worker)
    # 模型平均
    with torch.no_grad():
        model.weight.set_(#此时Zhang和Li的model已经在安全工作机上了
            ((Zhang_model.weight.data + Li_model.weight.data) / 2).get())
        model.bias.set_(
            ((Zhang_model.bias.data + Li_model.bias.data) / 2).get())
    # 打印当前结果
    print("Li:" + str(Li_loss) + "Zhang:" + str(Zhang_loss))

# 模型评估
preds = model(data)
loss = ((preds - target) ** 2).sum()
print(preds)
print(target)
print(loss.data)
