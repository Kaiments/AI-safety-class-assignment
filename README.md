# AI-safety-class-assignment
A simple neural network
项目名称
Googlenet+cifar10
项目简介
一个简单的googlenet神经网络，并在cifar10上训练和验证
上手指南
运行此项目你只需要一个能运行python的编译器和pytorch环境，直接运行train.py即可
超参数分析
修改了参数batch_size，在一定范围内，batch_size越大，训练一个epoch的迭代次数越少，训练越快，并且训练的准确率越高。但是当batch_size过大，容易导致内存溢出，训练缓慢并且陷入局部最优
