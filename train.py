import os
import json

import torch
import torch.nn as nn
from torchvision import transforms, datasets
import torch.optim as optim
from tqdm import tqdm

from model import GoogLeNet
import torchvision

batch_size = 32
NCLASS = 10
EPOCHS = 10
learn_rate = 0.0003
ROOT = './cifar10/'
def main():
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #确定运算设备
    device = torch.device("cuda:0")  # 确定运算设备
    print("using {} device.".format(device))


    # data_transform = {
    #     "train": transforms.Compose([transforms.RandomResizedCrop(224),  # 随机裁剪为224*224
    #                                  transforms.RandomHorizontalFlip(),  # 随机翻转
    #                                  transforms.ToTensor(),  # 转化为tensor格式
    #                                  # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    #                                  transforms.Normalize([0.5], [0.5])]),
    #
    #     "val": transforms.Compose([transforms.Resize((224, 224)),
    #                                transforms.ToTensor(),
    #                                # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    #                                transforms.Normalize([0.5], [0.5])])
    #
    # }
    # train_dataset = torchvision.datasets.MNIST(
    #     root=ROOT,  # 数据集存放路径
    #     train=True,  # 确定需要train数据集还是test数据集
    #     transform=data_transform["train"],  # transform转换数据格式
    #     download=False,  # 是否需要下载
    # )
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    #
    # validate_dataset = torchvision.datasets.MNIST(
    #     root=ROOT,
    #     train=False,
    #     transform=data_transform["val"]
    # )
    # validate_loader = torch.utils.data.DataLoader(validate_dataset, batch_size=batch_size, shuffle=False,)
    data_transform = {
        "train": transforms.Compose([transforms.Grayscale(), transforms.RandomResizedCrop(224),  # 随机裁剪为224*224
                                     transforms.RandomHorizontalFlip(),  # 随机翻转
                                     transforms.ToTensor(),  # 转化为tensor格式
                                     # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                     transforms.Normalize([0.5], [0.5])]),

        "val": transforms.Compose([transforms.Grayscale(), transforms.Resize((224, 224)),
                                   transforms.ToTensor(),
                                   # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                   transforms.Normalize([0.5], [0.5])])

    }
    train_dataset = torchvision.datasets.CIFAR10(
        root=ROOT,  # 数据集存放路径
        train=True,  # 确定需要train数据集还是test数据集
        transform=data_transform["train"],  # transform转换数据格式
        download=False,  # 是否需要下载
    )
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    validate_dataset = torchvision.datasets.CIFAR10(
        root=ROOT,
        train=False,
        transform=data_transform["val"]
    )
    validate_loader = torch.utils.data.DataLoader(validate_dataset, batch_size=batch_size, shuffle=False, )

    train_num = len(train_dataset)
    val_num = len(validate_dataset)
    print("using {} images for training, {} images for validation.".format(train_num, val_num))

    net = GoogLeNet(num_classes=NCLASS)
    net.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=learn_rate)

    epochs = EPOCHS
    best_acc = 0.0
    save_path = './googleNet.pth'
    train_steps = len(train_loader)
    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            logits = net(images.to(device))
            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)

        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(validate_loader)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))  # eval model only have last output layer
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

        val_accurate = acc / val_num
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)

    print('Finished Training')


if __name__ == '__main__':
    main()
