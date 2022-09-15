import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import logging
import syft as sy
import torchvision
import  pdb
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
    #     pdb.set_trace()
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
        print(data.location)
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

# Train Epoch:10[2560/60032(4.264392%)]	Loss:0.087165
# <VirtualWorker id:qin #objects:13>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# Train Epoch:10[3200/60032(5.330490%)]	Loss:0.015099
# <VirtualWorker id:qin #objects:13>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# Train Epoch:10[3840/60032(6.396588%)]	Loss:0.048682
# <VirtualWorker id:qin #objects:13>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# Train Epoch:10[4480/60032(7.462687%)]	Loss:0.010200
# <VirtualWorker id:qin #objects:13>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# Train Epoch:10[5120/60032(8.528785%)]	Loss:0.003653
# <VirtualWorker id:qin #objects:13>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# Train Epoch:10[5760/60032(9.594883%)]	Loss:0.069432
# <VirtualWorker id:qin #objects:13>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# Train Epoch:10[6400/60032(10.660981%)]	Loss:0.013932
# <VirtualWorker id:qin #objects:13>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# Train Epoch:10[7040/60032(11.727079%)]	Loss:0.036530
# <VirtualWorker id:qin #objects:13>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# Train Epoch:10[7680/60032(12.793177%)]	Loss:0.013720
# <VirtualWorker id:qin #objects:13>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# Train Epoch:10[8320/60032(13.859275%)]	Loss:0.022364
# <VirtualWorker id:qin #objects:13>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# Train Epoch:10[8960/60032(14.925373%)]	Loss:0.061565
# <VirtualWorker id:qin #objects:13>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# Train Epoch:10[9600/60032(15.991471%)]	Loss:0.013321
# <VirtualWorker id:qin #objects:13>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# Train Epoch:10[10240/60032(17.057569%)]	Loss:0.025544
# <VirtualWorker id:qin #objects:13>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# Train Epoch:10[10880/60032(18.123667%)]	Loss:0.043692
# <VirtualWorker id:qin #objects:13>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# Train Epoch:10[11520/60032(19.189765%)]	Loss:0.012450
# <VirtualWorker id:qin #objects:13>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# Train Epoch:10[12160/60032(20.255864%)]	Loss:0.009575
# <VirtualWorker id:qin #objects:13>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# Train Epoch:10[12800/60032(21.321962%)]	Loss:0.047044
# <VirtualWorker id:qin #objects:13>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# Train Epoch:10[13440/60032(22.388060%)]	Loss:0.010062
# <VirtualWorker id:qin #objects:13>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# Train Epoch:10[14080/60032(23.454158%)]	Loss:0.037832
# <VirtualWorker id:qin #objects:13>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# Train Epoch:10[14720/60032(24.520256%)]	Loss:0.003940
# <VirtualWorker id:qin #objects:13>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# Train Epoch:10[15360/60032(25.586354%)]	Loss:0.008579
# <VirtualWorker id:qin #objects:13>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# Train Epoch:10[16000/60032(26.652452%)]	Loss:0.012265
# <VirtualWorker id:qin #objects:13>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# Train Epoch:10[16640/60032(27.718550%)]	Loss:0.085300
# <VirtualWorker id:qin #objects:13>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# Train Epoch:10[17280/60032(28.784648%)]	Loss:0.119344
# <VirtualWorker id:qin #objects:13>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# Train Epoch:10[17920/60032(29.850746%)]	Loss:0.055360
# <VirtualWorker id:qin #objects:13>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# Train Epoch:10[18560/60032(30.916844%)]	Loss:0.122496
# <VirtualWorker id:qin #objects:13>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# Train Epoch:10[19200/60032(31.982942%)]	Loss:0.025502
# <VirtualWorker id:qin #objects:13>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# Train Epoch:10[19840/60032(33.049041%)]	Loss:0.018936
# <VirtualWorker id:qin #objects:13>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# Train Epoch:10[20480/60032(34.115139%)]	Loss:0.004858
# <VirtualWorker id:qin #objects:13>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# Train Epoch:10[21120/60032(35.181237%)]	Loss:0.135052
# <VirtualWorker id:qin #objects:13>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# Train Epoch:10[21760/60032(36.247335%)]	Loss:0.008347
# <VirtualWorker id:qin #objects:13>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# Train Epoch:10[22400/60032(37.313433%)]	Loss:0.018987
# <VirtualWorker id:qin #objects:13>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# Train Epoch:10[23040/60032(38.379531%)]	Loss:0.054907
# <VirtualWorker id:qin #objects:13>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# Train Epoch:10[23680/60032(39.445629%)]	Loss:0.041214
# <VirtualWorker id:qin #objects:13>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# Train Epoch:10[24320/60032(40.511727%)]	Loss:0.017203
# <VirtualWorker id:qin #objects:13>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# Train Epoch:10[24960/60032(41.577825%)]	Loss:0.006267
# <VirtualWorker id:qin #objects:13>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# Train Epoch:10[25600/60032(42.643923%)]	Loss:0.045798
# <VirtualWorker id:qin #objects:13>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# Train Epoch:10[26240/60032(43.710021%)]	Loss:0.056282
# <VirtualWorker id:qin #objects:13>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# Train Epoch:10[26880/60032(44.776119%)]	Loss:0.027300
# <VirtualWorker id:qin #objects:13>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# Train Epoch:10[27520/60032(45.842217%)]	Loss:0.019397
# <VirtualWorker id:qin #objects:13>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# Train Epoch:10[28160/60032(46.908316%)]	Loss:0.005717
# <VirtualWorker id:qin #objects:13>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# Train Epoch:10[28800/60032(47.974414%)]	Loss:0.058740
# <VirtualWorker id:qin #objects:13>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# Train Epoch:10[29440/60032(49.040512%)]	Loss:0.006374
# <VirtualWorker id:qin #objects:13>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:qin #objects:14>
# <VirtualWorker id:zheng #objects:12>
# <VirtualWorker id:zheng #objects:14>
# Train Epoch:10[30080/60032(50.106610%)]	Loss:0.039391
# <VirtualWorker id:zheng #objects:13>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# Train Epoch:10[30720/60032(51.172708%)]	Loss:0.053566
# <VirtualWorker id:zheng #objects:13>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# Train Epoch:10[31360/60032(52.238806%)]	Loss:0.041850
# <VirtualWorker id:zheng #objects:13>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# Train Epoch:10[32000/60032(53.304904%)]	Loss:0.021724
# <VirtualWorker id:zheng #objects:13>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# Train Epoch:10[32640/60032(54.371002%)]	Loss:0.019307
# <VirtualWorker id:zheng #objects:13>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# Train Epoch:10[33280/60032(55.437100%)]	Loss:0.050280
# <VirtualWorker id:zheng #objects:13>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# Train Epoch:10[33920/60032(56.503198%)]	Loss:0.016044
# <VirtualWorker id:zheng #objects:13>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# Train Epoch:10[34560/60032(57.569296%)]	Loss:0.134134
# <VirtualWorker id:zheng #objects:13>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# Train Epoch:10[35200/60032(58.635394%)]	Loss:0.015146
# <VirtualWorker id:zheng #objects:13>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# Train Epoch:10[35840/60032(59.701493%)]	Loss:0.020528
# <VirtualWorker id:zheng #objects:13>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# Train Epoch:10[36480/60032(60.767591%)]	Loss:0.059684
# <VirtualWorker id:zheng #objects:13>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# Train Epoch:10[37120/60032(61.833689%)]	Loss:0.019914
# <VirtualWorker id:zheng #objects:13>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# Train Epoch:10[37760/60032(62.899787%)]	Loss:0.024003
# <VirtualWorker id:zheng #objects:13>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# Train Epoch:10[38400/60032(63.965885%)]	Loss:0.088199
# <VirtualWorker id:zheng #objects:13>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# Train Epoch:10[39040/60032(65.031983%)]	Loss:0.055994
# <VirtualWorker id:zheng #objects:13>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# Train Epoch:10[39680/60032(66.098081%)]	Loss:0.003917
# <VirtualWorker id:zheng #objects:13>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# Train Epoch:10[40320/60032(67.164179%)]	Loss:0.064413
# <VirtualWorker id:zheng #objects:13>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# Train Epoch:10[40960/60032(68.230277%)]	Loss:0.027386
# <VirtualWorker id:zheng #objects:13>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# Train Epoch:10[41600/60032(69.296375%)]	Loss:0.048595
# <VirtualWorker id:zheng #objects:13>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# Train Epoch:10[42240/60032(70.362473%)]	Loss:0.004878
# <VirtualWorker id:zheng #objects:13>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# Train Epoch:10[42880/60032(71.428571%)]	Loss:0.006874
# <VirtualWorker id:zheng #objects:13>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# Train Epoch:10[43520/60032(72.494670%)]	Loss:0.014235
# <VirtualWorker id:zheng #objects:13>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# Train Epoch:10[44160/60032(73.560768%)]	Loss:0.104129
# <VirtualWorker id:zheng #objects:13>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# Train Epoch:10[44800/60032(74.626866%)]	Loss:0.020221
# <VirtualWorker id:zheng #objects:13>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# Train Epoch:10[45440/60032(75.692964%)]	Loss:0.031282
# <VirtualWorker id:zheng #objects:13>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# Train Epoch:10[46080/60032(76.759062%)]	Loss:0.040444
# <VirtualWorker id:zheng #objects:13>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# Train Epoch:10[46720/60032(77.825160%)]	Loss:0.038107
# <VirtualWorker id:zheng #objects:13>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# Train Epoch:10[47360/60032(78.891258%)]	Loss:0.014051
# <VirtualWorker id:zheng #objects:13>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# Train Epoch:10[48000/60032(79.957356%)]	Loss:0.035436
# <VirtualWorker id:zheng #objects:13>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# Train Epoch:10[48640/60032(81.023454%)]	Loss:0.132325
# <VirtualWorker id:zheng #objects:13>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# Train Epoch:10[49280/60032(82.089552%)]	Loss:0.007462
# <VirtualWorker id:zheng #objects:13>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# Train Epoch:10[49920/60032(83.155650%)]	Loss:0.007120
# <VirtualWorker id:zheng #objects:13>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# Train Epoch:10[50560/60032(84.221748%)]	Loss:0.003217
# <VirtualWorker id:zheng #objects:13>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# Train Epoch:10[51200/60032(85.287846%)]	Loss:0.027387
# <VirtualWorker id:zheng #objects:13>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# Train Epoch:10[51840/60032(86.353945%)]	Loss:0.029524
# <VirtualWorker id:zheng #objects:13>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# Train Epoch:10[52480/60032(87.420043%)]	Loss:0.010947
# <VirtualWorker id:zheng #objects:13>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# Train Epoch:10[53120/60032(88.486141%)]	Loss:0.038999
# <VirtualWorker id:zheng #objects:13>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# Train Epoch:10[53760/60032(89.552239%)]	Loss:0.017924
# <VirtualWorker id:zheng #objects:13>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# Train Epoch:10[54400/60032(90.618337%)]	Loss:0.017574
# <VirtualWorker id:zheng #objects:13>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# Train Epoch:10[55040/60032(91.684435%)]	Loss:0.066043
# <VirtualWorker id:zheng #objects:13>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# Train Epoch:10[55680/60032(92.750533%)]	Loss:0.056843
# <VirtualWorker id:zheng #objects:13>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# Train Epoch:10[56320/60032(93.816631%)]	Loss:0.018136
# <VirtualWorker id:zheng #objects:13>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# Train Epoch:10[56960/60032(94.882729%)]	Loss:0.022094
# <VirtualWorker id:zheng #objects:13>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# Train Epoch:10[57600/60032(95.948827%)]	Loss:0.066633
# <VirtualWorker id:zheng #objects:13>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# Train Epoch:10[58240/60032(97.014925%)]	Loss:0.009795
# <VirtualWorker id:zheng #objects:13>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# Train Epoch:10[58880/60032(98.081023%)]	Loss:0.006274
# <VirtualWorker id:zheng #objects:13>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# Train Epoch:10[59520/60032(99.147122%)]	Loss:0.003738
# <VirtualWorker id:zheng #objects:13>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
# <VirtualWorker id:zheng #objects:14>
#
# Test set: Average loss: 0.0551, Accuracy: 9822/10000 (98%)