#coding=gbk
#coding:utf-8
import syft as sy
import torch
import sys
from torch.nn import Parameter
import torch.nn as nn
import torch.nn.functional as F
import pdb
# ����hook,hook��һ����ʼ������:
hook = sy.TorchHook(torch)
# print(hook)
# <syft.frameworks.torch.hook.hook.TorchHook object at 0x7fbff17a2590>

# print(torch.tensor([1,2,3,4,5]))
# # tensor([1, 2, 3, 4, 5])
# x = torch.tensor([1,2,3,4,5])
# print('x = ', x)
# # x =  tensor([1, 2, 3, 4, 5])
# y = x+x
# print('y = ', y)
# # y =  tensor([ 2,  4,  6,  8, 10])


# ����һ��bob����:
# ȥ���Ļ�����ѧϰ��Ŀ��֮һ�����ܹ��������ڱ��ص����ݣ����syft�����hook���ƾ��Ǵ�������ڵ㣬ģ���ڵ����Ͻ�ģ�Ĺ��̡�
# ��ʱ������ΪMe��Զ����Bob��Bob�����ﲻ�ù��ģ�Ҳ���ڻ��ǡ�
# bob = sy.VirtualWorker(hook, id='bob')
"""
print('bob = ', bob)
# bob =  <VirtualWorker id:bob #objects:0>

# ��������Tensor:
x = torch.tensor([1,2,3,4,5])
y = torch.tensor([1,1,1,1,1])

print('bob._objects = ', bob._objects)
# bob._objects =  {}

# ��������Tensor����bob:
x_ptr = x.send(bob)
y_ptr = y.send(bob)


# ����bob�е����ݶ���:
print('bob._objects = ', bob._objects, 'after send')
# bob._objects =  {47264313560: tensor([1, 2, 3, 4, 5]), 11757937222: tensor([1, 1, 1, 1, 1])} after send
print('x_ptr = ', x_ptr)
# x_ptr =  (Wrapper)>[PointerTensor | me:22264687418 -> bob:47264313560]
print('y_ptr = ', y_ptr)
# y_ptr =  (Wrapper)>[PointerTensor | me:84790901380 -> bob:11757937222]

print('x_ptr.location = ', x_ptr.location)
# x_ptr.location =  <VirtualWorker id:bob #objects:2>
print('x_ptr.owner = ', x_ptr.owner)
# x_ptr.owner =  <VirtualWorker id:me #objects:0>

# pdb.set_trace()

print('y_ptr.location = ', y_ptr.location)
print('y_ptr.owner = ', y_ptr.owner)
# y_ptr.location =  <VirtualWorker id:bob #objects:2>
# y_ptr.owner =  <VirtualWorker id:me #objects:0>

# ��ʱ��x_ptr��y_ptr������ָ��bob�Ķ��󣬿������Ϊָ��Ĺ��ܣ���������ָ��������ӷ�:
z = x_ptr + y_ptr
print('z = ', z)
# z =  (Wrapper)>[PointerTensor | me:92333043891 -> bob:12930483167]

# �ٿ�bob�ϵĶ���,���Կ�������bob��������һ������
print('bob._objects = ', bob._objects, 'after add')
# bob._objects =  {47264313560: tensor([1, 2, 3, 4, 5]), 11757937222: tensor([1, 1, 1, 1, 1]), 12930483167: tensor([2, 3, 4, 5, 6])} after add


# ʹ��Tensorָ��
# ����Ĳ����������㣬�ڴ���Tensorʱ����ֱ�ӷ���bob
x = torch.tensor([1,2,3,4,5]).send(bob)
y = torch.tensor([1,1,1,1,1]).send(bob)
z = x + y
print(z)
# (Wrapper)>[PointerTensor | me:20992276262 -> bob:501242833]
print(z.get())
tensor([2, 3, 4, 5, 6])

# �ݶȲ���
# x��y�����ݶ�:
x = torch.tensor([1,2,3,4,5.], requires_grad=True).send(bob)
y = torch.tensor([1,1,1,1,1.], requires_grad=True).send(bob)

# ���
z = (x + y).sum()
# ���򴫲�
z.backward()

print(z)
# (Wrapper)>[PointerTensor | me:45617407530 -> bob:9347229155]
print(z.backward())
# (Wrapper)>[PointerTensor | me:87874788594 -> bob:53673560597]
print(x.grad)
# (Wrapper)>[PointerTensor | me:50524552071 -> bob:6037595972]::grad

"""

kong = sy.VirtualWorker(hook=hook,id='kong')

data = torch.tensor([0, 1, 2, 1, 2])#����tensor����
data_ptr = data.send(kong)#ָ��ָ���������
print(data_ptr)
print(kong._objects)
data = data_ptr.get()#ȡ������
print(data)

print(kong._objects)#������ʱ����qin������ɶ
a = torch.tensor([3.14, 6.28]).send(kong)
b = torch.tensor([6.14, 3.28]).send(kong)
c = a + b
print(c)
print(kong._objects)
train = torch.tensor([2.4, 6.2], requires_grad=True).send(kong)
label = torch.tensor([2, 6.]).send(kong)

loss = (train - label).abs().sum()
loss.backward()
train = train.get()

print(train)
print(train.grad)
# (Wrapper)>[PointerTensor | me:44514703080 -> kong:83976218312]
# {83976218312: tensor([0, 1, 2, 1, 2])}
# tensor([0, 1, 2, 1, 2])
# {}
# (Wrapper)>[PointerTensor | me:65369615302 -> kong:22339908538]
# {92821642172: tensor([3.1400, 6.2800]), 55405325189: tensor([6.1400, 3.2800]), 22339908538: tensor([9.2800, 9.5600])}
# tensor([2.4000, 6.2000], requires_grad=True)
# tensor([1., 1.])