from imaplib import Time2Internaldate
import torch
import time

from time import process_time

# 用于构建NN
import torch.nn as nn

# 需要用到这个库里面的激活函数
import torch.nn.functional as F

# 用于构建优化器
import torch.optim as optim

# 用于初始化数据
from torchvision import datasets, transforms

# 用于分布式训练
import syft as sy

hook = sy.TorchHook(torch)
Bob = sy.VirtualWorker(hook, id='Bob')
Alice = sy.VirtualWorker(hook, id='Alice')


class Arguments():
    def __init__(self):
        self.batch_size = 1
        self.test_batch_size = 100
        self.epochs = 10
        self.lr = 0.01
        self.momentum = 0.5
        self.seed = 1
        self.log_interval = 1
        self.save_model = True


# 实例化参数类
args = Arguments()

# 固定化随机数种子，使得每次训练的随机数都是固定的
torch.manual_seed(args.seed)
# 定义联邦训练数据集，定义转换器为 x=(x-mean)/标准差
fed_dataset_Bob = datasets.MNIST('./mnist_data', download=False, train=True,
                                 transform=transforms.Compose(
                                     [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))

# 定义数据加载器，shuffle是采用随机的方式抽取数据
fed_loader_Bob = torch.utils.data.DataLoader(fed_dataset_Bob, batch_size=args.batch_size, shuffle=True)

# 定义联邦训练数据集，定义转换器为 x=(x-mean)/标准差
fed_dataset_Alice = datasets.MNIST('./mnist_data', download=False, train=True,
                                   transform=transforms.Compose(
                                       [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))

# 定义数据加载器，shuffle是采用随机的方式抽取数据
fed_loader_Alice = torch.utils.data.DataLoader(fed_dataset_Alice, batch_size=args.batch_size, shuffle=True)

# 定义测试集
test_dataset = datasets.MNIST('data', download=True, train=False,
                              transform=transforms.Compose(
                                  [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))

# 定义测试集加载器
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=True)


# 构建神经网络模型
class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        # 输入维度为1，输出维度为20，卷积核大小为：5*5，步幅为1
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        # 最后映射到10维上
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        # print(x.shape)
        x = F.relu(self.conv1(x))  # 28*28*1 -> 24*24*20
        # print(x.shape)
        # 卷机核：2*2 步幅：2
        x = F.max_pool2d(x, 2, 2)  # 24*24*20 -> 12*12*20
        # print(x.shape)
        x = F.relu(self.conv2(x))  # 12*12*20 -> 8*8*30
        # print(x.shape)
        x = F.max_pool2d(x, 2, 2)  # 8*8*30 -> 4*4*50
        # print(x.shape)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # 使用logistic函数作为softmax进行激活吗就
        return F.log_softmax(x, dim=1)


def fedavg_updata_weight(model: Net, Alice_model: Net, Bob_model: Net, num: int):
    """
    训练中需要修改的参数如下，对以下参数进行avg
    conv1.weight
    conv1.bias
    conv2.weight
    conv2.bias
    fc1.weight
    fc1.bias
    fc2.weight
    fc2.bias
    """
    model.conv1.weight.set_((Bob_model.conv1.weight.data + Alice_model.conv1.weight.data) / num)
    model.conv1.bias.set_((Bob_model.conv1.bias.data + Alice_model.conv1.bias.data) / num)
    model.conv2.weight.set_((Bob_model.conv2.weight.data + Alice_model.conv2.weight.data) / num)
    model.conv2.bias.set_((Bob_model.conv2.bias.data + Alice_model.conv2.bias.data) / num)
    model.fc1.weight.set_((Bob_model.fc1.weight.data + Alice_model.fc1.weight.data) / num)
    model.fc1.bias.set_((Bob_model.fc1.bias.data + Alice_model.fc1.bias.data) / num)
    model.fc2.weight.set_((Bob_model.fc2.weight.data + Alice_model.fc2.weight.data) / num)
    model.fc2.bias.set_((Bob_model.fc2.bias.data + Alice_model.fc2.bias.data) / num)
    print("更新一次")


def train(model: Net, fed_loader: torch.utils.data.DataLoader):
    Bob_model = Net()
    Alice_model = Net()
    # 定义Bob的优化器
    Bob_opt = optim.SGD(Bob_model.parameters(), lr=args.lr)
    # 定义Alice的优化器
    Alice_opt = optim.SGD(Alice_model.parameters(), lr=args.lr)

    model.train()
    Bob_model.train()
    Alice_model.train()

    Bob_model.send(Bob)
    Alice_model.send(Alice)
    for epoch in range(1, args.epochs + 1):
        # 传递模型
        Alice_loss = 0
        Bob_loss = 0

        # 模拟Bob训练数据
        for epoch_ind, (data, target) in enumerate(fed_loader):
            data = data.send(Bob)
            target = target.send(Bob)

            Bob_opt.zero_grad()
            pred = Bob_model(data)
            Bob_loss = F.nll_loss(pred, target)
            Bob_loss.backward()
            Bob_opt.step()

            if (epoch_ind % 50 == 0):
                print("There is epoch:{} epoch_ind:{} in Bob loss:{:.6f}".format(epoch, epoch_ind,
                                                                                 Bob_loss.get().data.item()))

        # 模拟Alice训练模型
        for epoch_ind, (data, target) in enumerate(fed_loader):
            data = data.send(Alice)
            target = target.send(Alice)

            Alice_opt.zero_grad()
            pred = Alice_model(data)
            Alice_loss = F.nll_loss(pred, target)
            Alice_loss.backward()
            Alice_opt.step()
            if (epoch_ind % 50 == 0):
                print("There is epoch:{} epoch_ind:{} in Alice loss:{:.6f}".format(epoch, epoch_ind,
                                                                                   Alice_loss.get().data.item()))

        with torch.no_grad():
            Bob_model.get()
            Alice_model.get()
            # 更新权重
            fedavg_updata_weight(model, Alice_model, Bob_model, 2)
            Bob_model = model.copy().send(Bob)

        if epoch % args.log_interval == 0:
            # 获得loss
            # 模型的loss
            # pred = model(fed_loader)
            # Loss = F.nll_loss(pred,target)
            print("Bob in train:")
            test(Bob_model, test_loader)
            print("Alice in train:")
            test(Alice_model, test_loader)
            print("model in train:")
            test(model, test_loader)


# 定义测试函数
def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(test_loader.dataset)
        print('\nTest set : Average loss : {:.4f}, Accuracy: {}/{} ( {:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))


if __name__ == '__main__':
    model = Net()
    # start=time.clock()
    start = process_time()

    train(model, fed_loader_Bob)
    # mid=time.clock()
    mid = process_time()
    print("model in test:")
    test(model, test_loader)
    # end=time.clock()
    end = process_time()
    time1 = mid - start
    time2 = end - mid
    print("训练时间：{}h{}m{}s 测试时间为：{}h{}m{}s".format(time1 // 60 // 60, time1 // 60, time1 % 60, time2 // 60 // 60,
                                                  time2 // 60, time2 % 60))

    if (args.save_model):
        torch.save(model.state_dict(), "Net.weight")

# 更新一次
# Bob in train:
#
# Test set : Average loss : 0.0407, Accuracy: 9874/10000 ( 99%)
#
# Alice in train:
#
# Test set : Average loss : 0.0491, Accuracy: 9844/10000 ( 98%)
#
# model in train:
#
# Test set : Average loss : 2.1383, Accuracy: 2904/10000 ( 29%)



# Traceback (most recent call last):
#   File "/home/kfy/anaconda3/envs/syftpy/lib/python3.7/site-packages/syft/frameworks/torch/tensors/interpreters/native.py", line 290, in handle_func_command
#     cmd, args, kwargs, return_args_type=True
#   File "/home/kfy/anaconda3/envs/syftpy/lib/python3.7/site-packages/syft/generic/frameworks/hook/hook_args.py", line 157, in unwrap_args_from_function
#     new_args = hook_args(args)
#   File "/home/kfy/anaconda3/envs/syftpy/lib/python3.7/site-packages/syft/generic/frameworks/hook/hook_args.py", line 350, in <lambda>
#     return lambda x: f(lambdas, x)
#   File "/home/kfy/anaconda3/envs/syftpy/lib/python3.7/site-packages/syft/generic/frameworks/hook/hook_args.py", line 566, in seven_fold
#     lambdas[1](args[1], **kwargs),
#   File "/home/kfy/anaconda3/envs/syftpy/lib/python3.7/site-packages/syft/generic/frameworks/hook/hook_args.py", line 328, in <lambda>
#     else lambda i: forward_func[type(i)](i)
#   File "/home/kfy/anaconda3/envs/syftpy/lib/python3.7/site-packages/syft/frameworks/torch/hook/hook_args.py", line 33, in <lambda>
#     else (_ for _ in ()).throw(PureFrameworkTensorFoundError),
#   File "/home/kfy/anaconda3/envs/syftpy/lib/python3.7/site-packages/syft/frameworks/torch/hook/hook_args.py", line 33, in <genexpr>
#     else (_ for _ in ()).throw(PureFrameworkTensorFoundError),
# syft.exceptions.PureFrameworkTensorFoundError
#
# During handling of the above exception, another exception occurred:
#
# Traceback (most recent call last):
#   File "/home/kfy/pysyft/mnist_bing2.py", line 210, in <module>
#     train(model, fed_loader_Bob)
#   File "/home/kfy/pysyft/mnist_bing2.py", line 146, in train
#     pred = Bob_model(data)
#   File "/home/kfy/anaconda3/envs/syftpy/lib/python3.7/site-packages/torch/nn/modules/module.py", line 532, in __call__
#     result = self.forward(*input, **kwargs)
#   File "/home/kfy/pysyft/mnist_bing2.py", line 82, in forward
#     x = F.relu(self.conv1(x))  # 28*28*1 -> 24*24*20
#   File "/home/kfy/anaconda3/envs/syftpy/lib/python3.7/site-packages/torch/nn/modules/module.py", line 532, in __call__
#     result = self.forward(*input, **kwargs)
#   File "/home/kfy/anaconda3/envs/syftpy/lib/python3.7/site-packages/torch/nn/modules/conv.py", line 345, in forward
#     return self.conv2d_forward(input, self.weight)
#   File "/home/kfy/anaconda3/envs/syftpy/lib/python3.7/site-packages/torch/nn/modules/conv.py", line 342, in conv2d_forward
#     self.padding, self.dilation, self.groups)
#   File "/home/kfy/anaconda3/envs/syftpy/lib/python3.7/site-packages/syft/generic/frameworks/hook/trace.py", line 83, in trace_wrapper
#     response = func(*args, **kwargs)
#   File "/home/kfy/anaconda3/envs/syftpy/lib/python3.7/site-packages/syft/generic/frameworks/hook/hook.py", line 586, in overloaded_func
#     response = handle_func_command(command)
#   File "/home/kfy/anaconda3/envs/syftpy/lib/python3.7/site-packages/syft/frameworks/torch/tensors/interpreters/native.py", line 324, in handle_func_command
#     response = cls._get_response(cmd, args, kwargs)
#   File "/home/kfy/anaconda3/envs/syftpy/lib/python3.7/site-packages/syft/frameworks/torch/tensors/interpreters/native.py", line 346, in _get_response
#     response = command_method(*args, **kwargs)
# RuntimeError: Expected 4-dimensional input for 4-dimensional weight 20 1 5 5, but got 1-dimensional input of size [0] instead
#
# Process finished with exit code 1