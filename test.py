#coding=gbk
#coding:utf-8
import syft as sy
import torch
import sys
from torch.nn import Parameter
import torch.nn as nn
import torch.nn.functional as F
import pdb
# 创建hook,hook是一个初始化操作:
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


# 虚拟一个bob机器:
# 去中心化机器学习的目标之一就是能够操作不在本地的数据，因此syft里面的hook机制就是创建虚拟节点，模拟多节点联合建模的过程。
# 此时，本地为Me，远程有Bob，Bob在哪里不用关心，也许在火星。
# bob = sy.VirtualWorker(hook, id='bob')
"""
print('bob = ', bob)
# bob =  <VirtualWorker id:bob #objects:0>

# 创建两个Tensor:
x = torch.tensor([1,2,3,4,5])
y = torch.tensor([1,1,1,1,1])

print('bob._objects = ', bob._objects)
# bob._objects =  {}

# 把这两个Tensor发给bob:
x_ptr = x.send(bob)
y_ptr = y.send(bob)


# 看看bob有的数据对象:
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

# 此时，x_ptr和y_ptr是两个指向bob的对象，可以理解为指针的功能，对这两个指针对象做加法:
z = x_ptr + y_ptr
print('z = ', z)
# z =  (Wrapper)>[PointerTensor | me:92333043891 -> bob:12930483167]

# 再看bob上的对象,可以看到现在bob上增加了一个对象
print('bob._objects = ', bob._objects, 'after add')
# bob._objects =  {47264313560: tensor([1, 2, 3, 4, 5]), 11757937222: tensor([1, 1, 1, 1, 1]), 12930483167: tensor([2, 3, 4, 5, 6])} after add


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

"""

kong = sy.VirtualWorker(hook=hook,id='kong')

data = torch.tensor([0, 1, 2, 1, 2])#创建tensor数据
data_ptr = data.send(kong)#指针指向这个数据
print(data_ptr)
print(kong._objects)
data = data_ptr.get()#取回数据
print(data)

print(kong._objects)#看看此时打工人qin手上有啥
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