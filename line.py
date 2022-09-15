# 有四个点，(0,0),(0,1),(1,0),(1,), 分为两类，如何用一条线把圆和三角形分开？
# 下面把问题抽象为机器学习分类模型，分别用集中训练和联邦学习进行实现。


import torch
from torch import nn
from torch import optim
import pdb
# 数据集
# 上面四个点的坐标为训练集data，用0代表圆，1代表三角形，标签为target，用张量表示如下:
data = torch.tensor([[0,0],[0,1],[1,0],[1,1.]], requires_grad=True)
target = torch.tensor([[0],[0],[1],[1.]], requires_grad=True)

"""
# 集中式训练

# 初始化模型
model = nn.Linear(2,1)

# 模型训练
def train():
    # SGD优化器
    opt = optim.SGD(params=model.parameters(),lr=0.1)

    for iter in range(20):
        # 1) 梯度清零
        opt.zero_grad()

        # 2) 预测
        pred = model(data)

        # 3) 计算损失
        loss = ((pred - target)**2).sum()

        # 4) 反向传播
        loss.backward()

        # 5) 梯度更新
        opt.step()

        # 6) 打印loss
        print(loss.data)


train()
# tensor(5.9401)
# tensor(1.5482)
# tensor(0.9018)
# tensor(0.6321)
# tensor(0.4559)
# tensor(0.3317)
# tensor(0.2429)
# tensor(0.1788)
# tensor(0.1324)
# tensor(0.0984)
# tensor(0.0734)
# tensor(0.0550)
# tensor(0.0413)
# tensor(0.0311)
# tensor(0.0235)
# tensor(0.0178)
# tensor(0.0135)
# tensor(0.0102)
# tensor(0.0078)
# tensor(0.0059)

# 打印模型参数
for param_tensor in model.state_dict():
    
    print(param_tensor,'\t',model.state_dict()[param_tensor])
# weight   tensor([[ 0.9474, -0.0380]])
# bias     tensor([0.0537])
"""

# 联邦训练

# 初始化两个虚拟节点，Alice和Bob
import syft as sy
hook = sy.TorchHook(torch)
bob = sy.VirtualWorker(hook, id="bob")
alice = sy.VirtualWorker(hook, id="alice")

# 把数据和标签划分，Alice有两条数据及对应标签，Bob有两条数据及对应标签
data_bob = data[0:2]
target_bob = target[0:2]

data_alice = data[2:]
target_alice = target[2:]

data_bob = data_bob.send(bob)
data_alice = data_alice.send(alice)
target_bob = target_bob.send(bob)
target_alice = target_alice.send(alice)

datasets = [(data_bob,target_bob),(data_alice,target_alice)]
# print(datasets)


# 在Alice和Bob上分别训练模型
model = nn.Linear(2, 1)

from syft.federated.floptimizer import Optims

workers = ['bob', 'alice']
optims = Optims(workers, optim=optim.Adam(params=model.parameters(), lr=0.1))

def train():

    for iter in range(20):

        # 迭代Alice和Bob上的数据
        for data, target in datasets:
            # pdb.set_trace()
            # 把模型发送到data所在节点
            model.send(data.location)
            print(data.location,data.location.id)

            # 调用优化器
            opt = optims.get_optim(data.location.id)

            # 梯度清零
            opt.zero_grad()
            # 根据pytorch中的backward()函数的计算，当网络参量进行反馈时，梯度是被积累的而不是被替换掉
            # 但是在每一个batch时毫无疑问并不需要将两个batch的梯度混合起来累积，因此这里就需要每个batch设置一遍zero_grad了。


            # 预测
            pred = model(data)

            # 计算损失
            loss = ((pred - target) ** 2).sum()

            # 反向传播
            loss.backward()

            # 梯度更新
            opt.step()

            # 获取模型
            model.get()

            # 打印损失
            print(loss.get().data)

train()

# tensor(0.4073)
# tensor(1.7476)
# tensor(0.0373)
# tensor(0.6147)
# tensor(0.0467)
# tensor(0.1415)
# tensor(0.2494)
# tensor(0.0358)
# tensor(0.4112)
# tensor(0.0508)
# tensor(0.4553)
# tensor(0.0705)
# tensor(0.3995)
# tensor(0.0603)
# tensor(0.2889)
# tensor(0.0305)
# tensor(0.1692)
# tensor(0.0055)
# tensor(0.0739)
# tensor(0.0031)
# tensor(0.0190)
# tensor(0.0242)
# tensor(0.0023)
# tensor(0.0549)
# tensor(0.0098)
# tensor(0.0761)
# tensor(0.0241)
# tensor(0.0767)
# tensor(0.0329)
# tensor(0.0581)
# tensor(0.0320)
# tensor(0.0321)
# tensor(0.0239)
# tensor(0.0130)
# tensor(0.0140)
# tensor(0.0100)
# tensor(0.0064)
# tensor(0.0231)
# tensor(0.0024)
# tensor(0.0438)

# 打印模型参数
for param_tensor in model.state_dict():
    # pdb.set_trace()
    print(param_tensor,'\t',model.state_dict()[param_tensor])
# weight   tensor([[ 1.1871, -0.0440]])
# bias     tensor([0.0198])

