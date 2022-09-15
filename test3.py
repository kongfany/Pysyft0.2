import torch
import syft as sy

from torch import nn
from torch import optim

hook = sy.TorchHook(torch)
Bob = sy.VirtualWorker(hook,id='Bob')
Alice = sy.VirtualWorker(hook,id='Alice')
data = torch.tensor([[0,0],[0,1], [1,0], [1,1.]],requires_grad=True)
targe = torch.tensor([[0],[0],[1], [1.]],requires_grad=True)

#全连接模型，输入是二维，输出是一维，可以理解为读两个数，输出一个数
model = nn.Linear(2,1)

data_Bob_ptr = data[:2].send(Bob)
targe_Bob_ptr = targe[:2].send(Bob)
data_Alice_ptr = data[2:].send(Alice)
targe_Alice_ptr = data[2:].send(Alice)

def train():
	#分发模型
    Bob_model = model.copy().send('Bob')
    Alice_model = model.copy().send('Alice')

    #分布式训练优化器设置，采用随机梯度下降
    Bob_opt = optim.SGD(params=Bob_model.parameters(),lr=0.1)
    Alice_opt = optim.SGD(params=Alice_model.parameters(),lr=0.1)

    for epoch in range(50):
        #梯度清0
        Bob_opt.zero_grad()
        Alice_opt.zero_grad()

        #在Bob上进行训练
        Bob_pred = Bob_model(data_Bob_ptr)
        Bob_loss = ((Bob_pred - targe_Bob_ptr)**2).sum()
        Bob_loss.backward()
        Bob_opt.step()

        #在Alice上面训练
        Alice_pred = Alice_model(data_Alice_ptr)
        Alice_loss = ((Alice_pred - targe_Alice_ptr)**2).sum()
        Alice_loss.backward()
        Alice_opt.step()

        # print(Bob_loss.get().data.item())
        print("第{}次epoch，avg_loss为：{}".format(epoch,
                                                (Bob_loss.get().data.item()+Alice_loss.get().data.item())/2))

    with torch.no_grad():
        #回收模型
        Bob_model.get()
        Alice_model.get()
        #进行模型平均后合并
        model.weight.set_((Bob_model.weight.data+Alice_model.weight.data))/2
        model.bias.set_((Bob_model.bias.data+Alice_model.bias.data))/2

	#对Bob的模型进行评估
    pred = Bob_model(torch.tensor([[0,1],[0,1], [1,0], [1,1.]]))
    print(pred.data)

	#对Alice的模型进行评估
    pred = Alice_model(torch.tensor([[0,1],[0,1], [1,0], [1,1.]]))
    print(pred.data)

train()

pred = model(torch.tensor([[0,1],[0,1], [1,0], [1,1.]]))
print(pred.data)
# 第0次epoch，avg_loss为：1.6678809523582458
# 第1次epoch，avg_loss为：0.8618229627609253
# 第2次epoch，avg_loss为：0.5861539877951145
# 第3次epoch，avg_loss为：0.4605204677209258
# 第4次epoch，avg_loss为：0.38911032187752426
# 第5次epoch，avg_loss为：0.34372521919431165
# 第6次epoch，avg_loss为：0.3135765829356387
# 第7次epoch，avg_loss为：0.29323137627216056
# 第8次epoch，avg_loss为：0.2794268667785218
# 第9次epoch，avg_loss为：0.2700423787609907
# 第10次epoch，avg_loss为：0.2636577452067286
# 第11次epoch，avg_loss为：0.259312195063103
# 第12次epoch，avg_loss为：0.2563537610549247
# 第13次epoch，avg_loss为：0.2543388404737925
# 第14次epoch，avg_loss为：0.2529660071377293
# 第15次epoch，avg_loss为：0.2520302068005549
# 第16次epoch，avg_loss为：0.2513918557924626
# 第17次epoch，avg_loss为：0.25095617672923254
# 第18次epoch，avg_loss为：0.25065843930497067
# 第19次epoch，avg_loss为：0.25045477571256924
# 第20次epoch，avg_loss为：0.2503152619137836
# 第21次epoch，avg_loss为：0.25021953290161036
# 第22次epoch，avg_loss为：0.2501536203453725
# 第23次epoch，avg_loss为：0.250108221282062
# 第24次epoch，avg_loss为：0.2500767883811932
# 第25次epoch，avg_loss为：0.25005490207149705
# 第26次epoch，avg_loss为：0.2500396727464249
# 第27次epoch，avg_loss为：0.2500289048430204
# 第28次epoch，avg_loss为：0.2500213637831621
# 第29次epoch，avg_loss为：0.25001594044351805
# 第30次epoch，avg_loss为：0.25001209727042806
# 第31次epoch，avg_loss为：0.25000927145129026
# 第32次epoch，avg_loss为：0.2500072020711741
# 第33次epoch，avg_loss为：0.25000563152161703
# 第34次epoch，avg_loss为：0.2500044838266149
# 第35次epoch，avg_loss为：0.2500035960106288
# 第36次epoch，avg_loss为：0.25000295616109725
# 第37次epoch，avg_loss为：0.25000237530184677
# 第38次epoch，avg_loss为：0.25000196397377294
# 第39次epoch，avg_loss为：0.2500015955727122
# 第40次epoch，avg_loss为：0.25000135319749006
# 第41次epoch，avg_loss为：0.25000114206000035
# 第42次epoch，avg_loss为：0.2500009277674735
# 第43次epoch，avg_loss为：0.2500008256134265
# 第44次epoch，avg_loss为：0.2500006534434078
# 第45次epoch，avg_loss为：0.25000055741952565
# 第46次epoch，avg_loss为：0.2500004755065106
# 第47次epoch，avg_loss为：0.2500004056305727
# 第48次epoch，avg_loss为：0.25000034602294363
# 第49次epoch，avg_loss为：0.2500002951747149
# tensor([[3.7308e-04],
#         [3.7308e-04],
#         [4.6202e-01],
#         [4.6300e-01]])
# tensor([[0.8305],
#         [0.8305],
#         [0.5000],
#         [1.0000]])
# tensor([[0.8309],
#         [0.8309],
#         [0.9621],
#         [1.4630]])
#
# Process finished with exit code 0