import torch
from torch import optim, nn
import syft as sy
import copy

hook = sy.TorchHook(torch)

# 创建一对工作机
bob = sy.VirtualWorker(hook, id='bob')
alice = sy.VirtualWorker(hook, id='alice')
# 中央服务器
secure_worker = sy.VirtualWorker(hook, id='secure_worker')

# 数据集
data = torch.tensor([[0,0],[0,1],[1,0],[1,1.]], requires_grad=True)
target = torch.tensor([[0],[0],[1],[1.]], requires_grad=True)

# 通过以下方式获取每个工作机的训练数据的指针
# 向bob和alice发送一些训练数据
bob_data = data[0:2].send(bob)
bob_target = target[0:2].send(bob)
alice_data = data[2:].send(alice)
alice_target = target[2:].send(alice)

# 建立模型
model = nn.Linear(2, 1)

# 设置epoch和iter数目
epochs = 10
worker_iters = 5
for epoch in range(epochs):
    # 发送模型给工作机
    # bob_model = model.copy().send(bob)
    alice_model = model.copy().send(alice)
    # # 每个epoch本地模型复制全局模型实现了模型的下发！！！
    bob_model = model.copy()
    print("初始模型：",bob_model.state_dict())
    bob_model = bob_model.send(bob)



    # 给每个工作机设置优化器
    bob_opt = optim.SGD(params=bob_model.parameters(), lr=0.1)
    alice_opt = optim.SGD(params=alice_model.parameters(), lr=0.1)


    # 并行进行训练两个工作机的模型
    for worker_iter in range(worker_iters):
        # 训练bob的模型

        bob_opt.zero_grad()
        bob_pred = bob_model(bob_data)
        bob_loss = ((bob_pred - bob_target) ** 2).sum()
        bob_loss.backward()

        bob_opt.step()
        bob_loss = bob_loss.get().data

        # 训练alice的模型

        alice_opt.zero_grad()
        alice_pred = alice_model(alice_data)
        alice_loss = ((alice_pred - alice_target) ** 2).sum()
        alice_loss.backward()

        alice_opt.step()
        alice_loss = alice_loss.get().data

    # 将训练好的模型都发送到中央服务器去
    bob_model.move(secure_worker)
    alice_model.move(secure_worker)
    # print(bob_model, bob_model.state_dict())
    # print(alice_model.state_dict())

    # 进行模型平均
    with torch.no_grad():
        model.weight.set_(((alice_model.weight.data + bob_model.weight.data) / 2).get())
        model.bias.set_(((alice_model.bias.data + bob_model.bias.data) / 2).get())
        print("平均后的模型参数",model.state_dict())

    print("bob loss: {}".format(bob_loss))
    print("alice loss: {}".format(alice_loss))

for param_tensor in model.state_dict():
    # pdb.set_trace()
    print(param_tensor,'\t',model.state_dict()[param_tensor])

# 初始模型： OrderedDict([('weight', tensor([[ 0.5116, -0.4069]])), ('bias', tensor([0.0446]))])
# 平均后的模型参数 OrderedDict([('weight', tensor([[ 0.6442, -0.1838]])), ('bias', tensor([0.2183]))])
# bob loss: 0.02784396894276142
# alice loss: 0.014486400410532951
# 初始模型： OrderedDict([('weight', tensor([[ 0.6442, -0.1838]])), ('bias', tensor([0.2183]))])
# 平均后的模型参数 OrderedDict([('weight', tensor([[ 0.6897, -0.1166]])), ('bias', tensor([0.2035]))])
# bob loss: 0.014929068274796009
# alice loss: 0.0038348352536559105
# 初始模型： OrderedDict([('weight', tensor([[ 0.6897, -0.1166]])), ('bias', tensor([0.2035]))])
# 平均后的模型参数 OrderedDict([('weight', tensor([[ 0.7230, -0.0819]])), ('bias', tensor([0.1727]))])
# bob loss: 0.008684148080646992
# alice loss: 0.0013633652124553919
# 初始模型： OrderedDict([('weight', tensor([[ 0.7230, -0.0819]])), ('bias', tensor([0.1727]))])
# 平均后的模型参数 OrderedDict([('weight', tensor([[ 0.7531, -0.0595]])), ('bias', tensor([0.1459]))])
# bob loss: 0.0052796052768826485
# alice loss: 0.0005024104611948133
# 初始模型： OrderedDict([('weight', tensor([[ 0.7531, -0.0595]])), ('bias', tensor([0.1459]))])
# 平均后的模型参数 OrderedDict([('weight', tensor([[ 0.7808, -0.0443]])), ('bias', tensor([0.1240]))])
# bob loss: 0.003339065471664071
# alice loss: 0.00017670000670477748
# 初始模型： OrderedDict([('weight', tensor([[ 0.7808, -0.0443]])), ('bias', tensor([0.1240]))])
# 平均后的模型参数 OrderedDict([('weight', tensor([[ 0.8059, -0.0337]])), ('bias', tensor([0.1062]))])
# bob loss: 0.0021886699832975864
# alice loss: 5.620685260510072e-05
# 初始模型： OrderedDict([('weight', tensor([[ 0.8059, -0.0337]])), ('bias', tensor([0.1062]))])
# 平均后的模型参数 OrderedDict([('weight', tensor([[ 0.8286, -0.0261]])), ('bias', tensor([0.0914]))])
# bob loss: 0.0014803948579356074
# alice loss: 1.4761944839847274e-05
# 初始模型： OrderedDict([('weight', tensor([[ 0.8286, -0.0261]])), ('bias', tensor([0.0914]))])
# 平均后的模型参数 OrderedDict([('weight', tensor([[ 0.8489, -0.0207]])), ('bias', tensor([0.0791]))])
# bob loss: 0.0010284699965268373
# alice loss: 2.4733315058256267e-06
# 初始模型： OrderedDict([('weight', tensor([[ 0.8489, -0.0207]])), ('bias', tensor([0.0791]))])
# 平均后的模型参数 OrderedDict([('weight', tensor([[ 0.8669, -0.0167]])), ('bias', tensor([0.0686]))])
# bob loss: 0.000730528321582824
# alice loss: 3.90692775908974e-08
# 初始模型： OrderedDict([('weight', tensor([[ 0.8669, -0.0167]])), ('bias', tensor([0.0686]))])
# 平均后的模型参数 OrderedDict([('weight', tensor([[ 0.8829, -0.0137]])), ('bias', tensor([0.0597]))])
# bob loss: 0.0005283153150230646
# alice loss: 3.646576942628599e-07
# weight 	 tensor([[ 0.8829, -0.0137]])
# bias 	 tensor([0.0597])
