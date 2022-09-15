import torch

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
Bob = sy.VirtualWorker(hook,id='Bob')
Alice = sy.VirtualWorker(hook,id='Alice')
class Arguments():
    def __init__(self):
        self.batch_size = 1
        self.test_batch_size = 100
        self.epochs = 3
        self.lr = 0.01
        self.momentum = 0.5
        self.no_cuda = False
        self.seed = 1
        self.log_interval = 30
        self.save_model = False

#实例化参数类
args = Arguments()
#判断是否使用GPu
use_cuda = not args.no_cuda and torch.cuda.is_available()
#固定化随机数种子，使得每次训练的随机数都是固定的
torch.manual_seed(args.seed)
device = torch.device('cuda' if use_cuda else 'cpu')

#定义联邦训练数据集，定义转换器为 x=(x-mean)/标准差
fed_dataset = datasets.MNIST('./mnist_data',download=True,train=True,
                            transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,),(0.3081,))]))

#定义数据加载器，shuffle是采用随机的方式抽取数据,顺便也把数据集定义在了客户端上
fed_loader = sy.FederatedDataLoader(federated_dataset=fed_dataset.federate((Alice,Bob)),batch_size=args.batch_size,shuffle=True)

#定义测试集
test_dataset = datasets.MNIST('./mnist_data',download=True,train=False,
                            transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,),(0.3081,))]))

#定义测试集加载器
test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=args.test_batch_size,shuffle=True)

#构建神经网络模型
class Net(nn.Module):
    def __init__(self) -> None:
        super(Net,self).__init__()
        #输入维度为1，输出维度为20，卷积核大小为：5*5，步幅为1
        self.conv1 = nn.Conv2d(1,20,5,1)
        self.conv2 = nn.Conv2d(20,50,5,1)
        self.fc1 = nn.Linear(4*4*50,500)
        #最后映射到10维上
        self.fc2 = nn.Linear(500,10)

    def forward(self,x):
        #print(x.shape)
        x = F.relu(self.conv1(x))#28*28*1 -> 24*24*20
        #print(x.shape)
        #卷机核：2*2 步幅：2
        x = F.max_pool2d(x,2,2)#24*24*20 -> 12*12*20
        #print(x.shape)
        x = F.relu(self.conv2(x))#12*12*20 -> 8*8*30
        #print(x.shape)
        x = F.max_pool2d(x,2,2)#8*8*30 -> 4*4*50
        #print(x.shape)
        x = x.view(-1,4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        #使用logistic函数作为softmax进行激活吗就
        return F.log_softmax(x, dim = 1)


# def train(model: Net, fed_loader: sy.FederatedDataLoader, opt: optim.SGD, epoch):
def train(args, model, device, fed_loader, opt, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(fed_loader):
        # 传递模型
        model.send(data.location)
        data, target = data.to(device), target.to(device)
        opt.zero_grad()
        pred = model(data)
        loss = F.nll_loss(pred, target)
        loss.backward()
        opt.step()

        model.get()
        if batch_idx % args.log_interval == 0:
            # 获得loss
            loss = loss.get()
            print('Train Epoch : {} [ {} / {} ({:.0f}%)] \tLoss: {:.6f}'.format(
                epoch, batch_idx * args.batch_size, len(fed_loader) *
                       args.batch_size,
                       100. * batch_idx / len(fed_loader), loss.item()))


# 定义测试函数
# def test(model, test_loader):
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(test_loader.dataset)
        print('\nTest set : Average loss : {:.4f}, Accuracy: {}/{} ( {:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
if __name__ == '__main__':
    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr = args.lr)

    for epoch in range(1, args.epochs +1):
        # train(model, fed_loader, optimizer, epoch)
        #
        # test(model, test_loader)
        train(args, model, device, fed_loader, optimizer, epoch)
        test(model, device, test_loader)

