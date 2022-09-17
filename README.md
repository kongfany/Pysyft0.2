# Pysyft0.2初学

`联邦学习框架0.2版本`

> PySyft是用于安全和隐私深度学习的Python库，它在主流深度学习框架（例如PyTorch和TensorFlow）中使用联邦学习，差分隐私和加密计算（例如多方计算（MPC）和同态加密（HE））将隐私数据与模型训练分离。
>
> https://github.com/OpenMined/PySyft

> 使用版本为0.2.4
>
> [syft_0.2.x](https://github.com/OpenMined/PySyft/tree/syft_0.2.x)

## 安装

[blog](https://www.cnblogs.com/mlblog27/p/14258662.html)

1. 创建conda的虚拟环境，指定python版本为3.7
2. 进入虚拟环境，安装pytorch（GPU / CPU版本）
3. 安装PySyft（0.2.4版本）
4. 重新安装PySyft的依赖

```
conda create -n syftpy python=3.7 --yes
conda activate syftpy # 进入虚拟环境

conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.1 -c pytorch# 下载cuda(10.1)版本pytorch 

pip3 install syft==0.2.4 --no-dependencies

pip install lz4~=3.0.2 msgpack~=1.0.0 phe~=1.4.0 scipy~=1.4.1 syft-proto~=0.2.5.a1 tblib~=1.6.0 websocket-client~=0.57.0 pip install websockets~=8.1.0 zstd~=1.4.4.0 Flask~=1.1.1 tornado==4.5.3 flask-socketio~=4.2.1 lz4~=3.0.2 Pillow~=6.2.2 pip install requests~=2.22.0 numpy~=1.18.1

conda install jupyter notebook==5.7.8
```

检查是否安装成功

```python
import syft as sy
print(sy.version.__version__)
```

## 简介

[参考](https://zhuanlan.zhihu.com/p/114774133)

- 首先，建立了一个用于worker间通信的标准化协议，以使联邦学习成为可能。
- 然后，开发了一个基于张量的链抽象模型，以有效覆盖运算（或编码新运算），例如在worker间发送／共享张量。
- 最后，提供了用这个新框架实现最近提出的差分隐私和多方计算协议的元素。

## 入门

[参考](https://zhuanlan.zhihu.com/p/411422407)

学习hook虚拟化节点机制，目标是实现机器学习由中心化到去中心化的转变，进而实现数据可用不可见。

```python
# 创建hook,hook是一个初始化操作:
hook = sy.TorchHook(torch)

# 虚拟一个bob机器:
# 去中心化机器学习的目标之一就是能够操作不在本地的数据，因此syft里面的hook机制就是创建虚拟节点，模拟多节点联合建模的过程。
# 此时，本地为Me，远程有Bob，Bob在哪里不用关心，也许在火星。
bob = sy.VirtualWorker(hook, id='bob')

# 创建两个Tensor:
x = torch.tensor([1,2,3,4,5])
y = torch.tensor([1,1,1,1,1])

# 把这两个Tensor发给bob:
x_ptr = x.send(bob)
y_ptr = y.send(bob)

# 看看bob有的数据对象:
print('bob._objects = ', bob._objects, 'after send')
# bob._objects =  {47264313560: tensor([1, 2, 3, 4, 5]), 11757937222: tensor([1, 1, 1, 1, 1])} after send

# 此时，x_ptr和y_ptr是两个指向bob的对象，可以理解为指针的功能，对这两个指针对象做加法:
z = x_ptr + y_ptr
print('z = ', z)
# z =  (Wrapper)>[PointerTensor | me:92333043891 -> bob:12930483167]

# 使用Tensor指针
# 上面的操作不够方便，在创建Tensor时可以直接发给bob
x = torch.tensor([1,2,3,4,5]).send(bob)
y = torch.tensor([1,1,1,1,1]).send(bob)
z = x + y
print(z)
# (Wrapper)>[PointerTensor | me:20992276262 -> bob:501242833]
print(z.get())
tensor([2, 3, 4, 5, 6])

# 梯度操作
# x、y增加梯度:
x = torch.tensor([1,2,3,4,5.], requires_grad=True).send(bob)
y = torch.tensor([1,1,1,1,1.], requires_grad=True).send(bob)

# 求和
z = (x + y).sum()
# 反向传播
z.backward()

print(z)
# (Wrapper)>[PointerTensor | me:45617407530 -> bob:9347229155]
print(z.backward())
# (Wrapper)>[PointerTensor | me:87874788594 -> bob:53673560597]
print(x.grad)
# (Wrapper)>[PointerTensor | me:50524552071 -> bob:6037595972]::grad
```

## 实战

![](/images/202206/18.png)

[参考](https://zhuanlan.zhihu.com/p/411451968)

有四个点，(0,0),(0,1),(1,0),(1,1), 分为两类，如何用一条线把圆和三角形分开？

下面把问题抽象为机器学习分类模型，分别用集中训练和联邦学习进行实现。

张量：[张量](https://blog.csdn.net/weixin_42259833/article/details/124766853)（tensor）是多维数组，目的是把向量、矩阵推向更高的维度。

nn.Linear的[基本用法](https://blog.51cto.com/u_11466419/5184188?b=totalstatistic)
nn.Linear定义一个神经网络的线性层，方法签名如下：

```python
torch.nn.Linear(in_features, # 输入的神经元个数
           out_features, # 输出神经元个数
           bias=True # 是否包含偏置
           )
```

Linear其实就是执行了一个转换函数，即：

y = x A T + b y = xA^T + b y=xAT+b

其中 A T A^T AT是模型要学习的参数，b是偏置

个人理解：`本实验`联邦学习的过程中仅仅是每个节点单独训练了一个模型，并没有进行聚合操作。但是为什么打印模型参数的的时候只有一个？

因为共用的一个模型



```python
import torch
from torch import nn
from torch import optim
import pdb
# 数据集
# 上面四个点的坐标为训练集data，用0代表圆，1代表三角形，标签为target，用张量表示如下:
data = torch.tensor([[0,0],[0,1],[1,0],[1,1.]], requires_grad=True)
target = torch.tensor([[0],[0],[1],[1.]], requires_grad=True)

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
    pdb.set_trace()
    print(param_tensor,'\t',model.state_dict()[param_tensor])
# weight   tensor([[ 0.9474, -0.0380]])
# bias     tensor([0.0537])
```



```python
import torch
from torch import nn
from torch import optim
import pdb
# 数据集
# 上面四个点的坐标为训练集data，用0代表圆，1代表三角形，标签为target，用张量表示如下:
data = torch.tensor([[0,0],[0,1],[1,0],[1,1.]], requires_grad=True)
target = torch.tensor([[0],[0],[1],[1.]], requires_grad=True)

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
print(datasets)


# 在Alice和Bob上分别训练模型
model = nn.Linear(2, 1)

from syft.federated.floptimizer import Optims

workers = ['bob', 'alice']
optims = Optims(workers, optim=optim.Adam(params=model.parameters(), lr=0.1))

def train():

    for iter in range(20):

        # 迭代Alice和Bob上的数据
        for data, target in datasets:
            
            # 把模型发送到data所在节点
            model.send(data.location)

            # 调用优化器
            opt = optims.get_optim(data.location.id)

            # 梯度清零
            opt.zero_grad()

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
    print(param_tensor,'\t',model.state_dict()[param_tensor])
# weight   tensor([[ 1.1871, -0.0440]])
# bias     tensor([0.0198])
```

------

## 简介-基本函数

[参考](https://blog.csdn.net/QinZheng7575/article/details/121918751)

```python
#初始化
import torch
import syft as sy
hook = sy.TorchHook(torch)#增加额外的功能
kong = sy.VirtualWorker(hook=hook,id='kong')
```
发送tensor
然后创建一个远程的虚拟打工人，并创建一些数据，才能分发给他

```python
data = torch.tensor([0, 1, 2, 1, 2])#创建tensor数据
data_ptr = data.send(kong)#指针指向这个数据
print(data_ptr)
# (Wrapper)>[PointerTensor | me:88077243152 -> kong:2511276968]

```

看到这个指针，从me（pysyft自动生成的）指向了kong，并且拥有一个id

现在kong拥有了我们给它发送的tensor。可以用kong._objects来查看kong拥有的东西

```python
print(kong._objects)
# {2511276968: tensor([0, 1, 2, 1, 2])}
```

返还tensor

远处的打工人kong算好了数据，应该把数据传回来，我们通过.get()从远处的打工人那里拿

```python
data = data_ptr.get()#取回数据
print(data)
# tensor([0, 1, 2, 1, 2])

print(kong._objects)#看看此时打工人手上有啥
# {}
```

通过指针张量(Pointer Tensor)做深度学习

```python
a = torch.tensor([3.14, 6.28]).send(kong)
b = torch.tensor([6.14, 3.28]).send(kong)
c = a + b
print(c)
# (Wrapper)>[PointerTensor | me:64012315468 -> kong:12403764116]
```

在机器上执行c = a + b的时候，一个指令下发给远处的kong，他创建了新的张量，然后给我们发回了一个指针 c ，使用这个API，我们就可以在原有的pytorch代码上，些许改变得到想要的结果。

```python
train = torch.tensor([2.4, 6.2], requires_grad=True).send(qin)
label = torch.tensor([2, 6.]).send(qin)

loss = (train - label).abs().sum()
loss.backward()
train = train.get()

print(train)
print(train.grad)
# tensor([2.4000, 6.2000], requires_grad=True)
# tensor([1., 1.])
```

————————

[参考](https://blog.csdn.net/qq_45931661/article/details/122524776)

me代表的是服务器的id（默认），one代表的是客户端的id，冒号后面的数字代表的是在客户端中的tensor的地址。这里可能会疑惑，为什么x_ptr的地址会改变，实际上，在执行x_ptr + y_ptr时，并不是在本地执行的加法，一个命令序列化后发送给了one，one执行了这个计算操作，创建了一个tensor，然后返回了一个指针到本地机器。
get方法收回模型，并且销毁x_ptr指针。



## 以Minist识别为例

参考[1](https://blog.csdn.net/QinZheng7575/article/details/121918751),[2](https://blog.csdn.net/weixin_45520982/article/details/116781765)

[官方](https://github.com/OpenMined/PySyft/blob/syft_0.2.x/examples/tutorials/Part%2006%20-%20Federated%20Learning%20on%20MNIST%20using%20a%20CNN.ipynb)



> 训练过程大概是：将数据分发给每个用户，然后把模型发送给远程用户。训练一段时间后，更新模型（收回模型）。。。
>
> 根据训练过程可以看出，相当于先在第一个用户上进行训练，然后再在第二个用户上训练，并没有用到模型聚合。



mnist.py

mnist数据集的导入

```python
#数据集mnist的导入
mnist_data = datasets.MNIST("D:\Python\PycharmProjects\PFLTest\mnist_data2", download=True, train=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.13066062,), (0.30810776,))
    ]))
```

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import logging
import syft as sy
import torchvision

hook = sy.TorchHook(torch)
qin = sy.VirtualWorker(hook=hook, id="qin")
zheng = sy.VirtualWorker(hook=hook, id="zheng")
#设定参数
args = {
    'use_cuda': True,
    'batch_size': 64,
    'test_batch_size': 1000,
    'lr': 0.01,
    'log_interval': 10,
    'epochs': 10,
    'save_model': False
}
use_cuda = args['use_cuda'] and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# 我们首先加载数据，然后使用。联邦方法。
# 此联邦数据集现在提供给 Federated DataLoader。测试数据集保持不变。
# 下面是训练数据，需要分发给远处打工人
federated_train_loader = sy.FederatedDataLoader(
    datasets.MNIST('./mnist_data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))
    .federate((qin, zheng)),
    batch_size=args['batch_size'], shuffle=True
)
# federate()函数已经实现了分发（省去了我们一个一个send()的麻烦）
# 下面是测试数据，在我们本地
test_loader = DataLoader(
    datasets.MNIST('./mnist_data', train=False,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args['batch_size'], shuffle=True
)

# CNN网络
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1),
            #输出26*26*32
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64,kernel_size=3, stride=1),
            #输出24*24*64
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(in_features=64*12*12, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=10),
        )
        self.dropout = nn.Dropout2d(0.25)  # 随机丢弃

    def forward(self, x):
        x = self.conv(x)#输入的时候是28*28*1,输出应该是24*24*64
        x = F.max_pool2d(x, 2)#用步长为2的池化,输出12*12*64
        x = x.view(-1, 64*12*12)#此时将其拉成一条直线进入全连接
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    # 远程迭代
    # for batch_idx, (data, target) in enumerate(federated_train_loader):
    #     if batch_idx < 5:
    #         print(batch_idx, type(data), data.location)
    #     else:
    #         break

    # 0 <class 'torch.Tensor'> < VirtualWorker id:qin  # objects:4>
    # 1 <class 'torch.Tensor'> < VirtualWorker id:qin  # objects:4>
    # 2 <class 'torch.Tensor'> < VirtualWorker id:qin  # objects:4>
    # 3 <class 'torch.Tensor'> < VirtualWorker id:qin  # objects:4>
    # 4 <class 'torch.Tensor'> < VirtualWorker id:qin  # objects:4>

    for batch_idx, (data, target) in enumerate(train_loader):  # enumrate用来编序号

        # model = model.send(data.location)  # 发送模型到远程
        model.send(data.location)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)

        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        # 以上都是发送命令给远程，下面是取回更新的模型
        model.get()
        if batch_idx % args['log_interval'] == 0:  # 打印间隔时间
            # 由于损失也是在远处产生的，因此我们需要把它取回来
            loss = loss.get()
            print('Train Epoch:{}[{}/{}({:.06f}%)]\tLoss:{:.06f}'.format(
                epoch,
                batch_idx * args['batch_size'],
                len(train_loader) * args['batch_size'],
                100. * batch_idx / len(train_loader),
                loss.item()
            ))
def test(model, device, test_loader):
    model.eval()
    '''返回model的返回值以字符串显示,使用PyTorch进行训练和测试时
    一定注意要把实例化的model指定train/eval，eval（）时，
    框架会自动把BN和DropOut固定住，不会取平均，而是用训练好的值，不然的话，
    一旦test的batch_size过小，很容易就会被BN层导致生成图片颜色失真极大！！'''
    test_loss = 0 #测试损失
    correct=0 #正确率
    with torch.no_grad():
        for data,target in test_loader:
            data,target = data.to(device),target.to(device)
            output=model(data)
            #将损失加起来
            test_loss+=F.nll_loss(output,target,reduction='sum').item()
            '''nll_loss的解释请看
            https://blog.csdn.net/qq_22210253/article/details/85229988
            和https://www.cnblogs.com/ranjiewen/p/10059490.html'''
            #进行预测最可能的分类
            pred =output.argmax(dim=1,keepdim=True)
            correct+=pred.eq(target.view_as(pred)).sum().item()#???
    test_loss/=len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss, correct, len(test_loader.dataset),
                100. * correct / len(test_loader.dataset)))


model = Net().to(device)
optimizer = optim.SGD(model.parameters(), lr=args['lr'])
logging.info("开始训练!!\n")
for epoch in range(1, args['epochs']+1):
    train(args, model, device, federated_train_loader, optimizer, epoch)
    test(model, device, test_loader)
# if (args["save_model"]):
torch.save(model.state_dict(), "mnist_cnn.pt")
```




```

Train Epoch:10[59520/60032(99.147122%)] Loss:0.008309

Test set: Average loss: 0.0435, Accuracy: 9855/10000 (99%)
```

## Minist

[参考](https://blog.csdn.net/qq_45931661/article/details/122560499)

在先实现C/S架构下的横向联邦学习模型。
大概处理过程如下：

**1.数据预处理，得到data_loader
2.建立虚拟机，分配数据集
3.初始化模型
4.将模型发送给虚拟机
5.指导虚拟机训练
6.回收模型**

手写数字识别模型（非并行训练）minist_chuan

服务器先将model发送Alice，Alice利用本地数据进行训练以后，再将模型发送给Server，Server接受到模型后，将模型发送给Bob，Bob利用本地数据进行训练，训练完成后，交给Server，Server利用本地的测试数据对model进行评估，然后将这个模型分发给Alice和Bob。
但是这样训练的缺点非常明显：Bob可以对接受模型的参数进行推理，可能能得到Alice本地数据的部分特征，从而破坏了数据的隐私性。非并行训练，训练时间长，Bob的数据是后训练的，可能占总模型的大，而不能很好的利用到双方的数据。

//等同于上个

手写数字识别模型（并行训练）

“您好作者，我是一名联邦学习的小白，读了您的文章我深受启发。我也同样遇到了聚合模型准确率不佳的问题，您最后的联邦平均过程，只是对中心模型的参数进行了更新，并未将客户端的本地模型进行参数更新，或许是因为客户端训练的模型之间差异较大，导致聚合效果不是很好。我尝试每轮联邦训练过后都更新本地模型降低差异（可能更贴近联邦学习的思想），发现可以提升训练的准确率，希望可以对您有所帮助。”

？？？？。。。。

[参考](https://blog.csdn.net/qq_45931661/article/details/122560499)

把初始模型发送到客户端，客户端用自己所拥有的数据进行训练后，再将模型返回给服务器。服务器将两个模型进行整合，得到一个对双方数据都有较好预测效果的模型。

但是合并后确实没有下发模型。

## 并行训练

[参考。4](https://blog.csdn.net/Yohuna/article/details/123789715)

```python
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
    bob_model = model.copy().send(bob)
    alice_model = model.copy().send(alice)
    # 每个epoch本地模型复制全局模型实现了模型的下发！！！

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

    # 进行模型平均
    with torch.no_grad():
        model.weight.set_(((alice_model.weight.data + bob_model.weight.data) / 2).get())
        model.bias.set_(((alice_model.bias.data + bob_model.bias.data) / 2).get())
        # print(model,model.state_dict())

    print("bob loss: {}".format(bob_loss))
    print("alice loss: {}".format(alice_loss))

for param_tensor in model.state_dict():
    # pdb.set_trace()
    print(param_tensor,'\t',model.state_dict()[param_tensor])
```

```
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
```

MNIST并行

```python
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
        self.batch_size = 64
        self.test_batch_size = 100
        self.epochs = 2
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
    model.train()
    Bob_model = model.copy()
    Alice_model = model.copy()


    Bob_model.train()
    Alice_model.train()

    # Bob_model.send(Bob)
    # Alice_model.send(Alice)
    for epoch in range(1, args.epochs + 1):
        # 传递模型
        Bob_model = model.copy()
        Alice_model = model.copy()
        Bob_model.send(Bob)
        Alice_model.send(Alice)
        # 定义Bob的优化器
        Bob_opt = optim.SGD(Bob_model.parameters(), lr=args.lr)
        # 定义Alice的优化器
        Alice_opt = optim.SGD(Alice_model.parameters(), lr=args.lr)
        # Alice_loss = 0
        # Bob_loss = 0

        # 模拟Bob训练数据
        for epoch_ind, (data, target) in enumerate(fed_loader):
            data = data.send(Bob)
            target = target.send(Bob)

            Bob_opt.zero_grad()
            pred = Bob_model(data)
            Bob_loss = F.nll_loss(pred, target)
            # Bob_loss.requires_grad_()
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
            # Alice_loss.requires_grad_()
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
            # Bob_model = model.copy().send(Bob)

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
```

```
# Bob in train:
#
# Test set : Average loss : 0.0938, Accuracy: 9730/10000 ( 97%)
#
# Alice in train:
#
# Test set : Average loss : 0.1088, Accuracy: 9657/10000 ( 97%)
#
# model in train:
#
# Test set : Average loss : 0.0965, Accuracy: 9720/10000 ( 97%)
#
# model in test:
#
# Test set : Average loss : 0.0965, Accuracy: 9720/10000 ( 97%)
#
# 训练时间：0.0h43.0m24.889736254000127s 测试时间为：0.0h0.0m27.157271203000164s
```

