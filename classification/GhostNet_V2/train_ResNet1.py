"""
训练器模块
"""
import os
from model import ghostnetv2
import torch
import datasets
import torch.nn as nn
from datetime import datetime
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter



# 训练器
class Trainer:
    def __init__(self, path, model, model_copy, img_save_path):
        self.path = path
        self.model = model
        self.model_copy = model_copy
        self.img_save_path = img_save_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = ghostnetv2(num_classes=7, width=1.0, dropout=0.2, args=None).to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=0.00001, weight_decay=0.0001)
        # 选择学习率调度策略
        # 方案一：StepLR
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.opt, step_size=500, gamma=0.1)

        # 方案二：ReduceLROnPlateau
        # self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.opt, mode='max', factor=0.1, patience=10, verbose=True)

        # 方案三：CosineAnnealingLR
        # self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=1000)
        self.loss_func = nn.CrossEntropyLoss()
        self.top_models = []  # 存储最高准确率的模型
        self.max_models = 3  # 保持最多3个模型文件

        trainset_list = [line.strip() for line in open(os.path.join(path, 'train.txt'), encoding='gbk')]
        valset_list = [line.strip() for line in open(os.path.join(path, 'val.txt'), encoding='gbk')]

        self.train_dataset = DataLoader(datasets.Datasets(trainset_list, path, True), batch_size=16, shuffle=True,
                                        num_workers=0)
        self.val_dataset = DataLoader(datasets.Datasets(valset_list, path, False), batch_size=16, shuffle=False,
                                      num_workers=0)

        if os.path.exists(self.model):
            self.net.load_state_dict(torch.load(self.model, map_location=self.device))
            self.net.to(self.device)
            print(f"Loaded {self.model}!")
        else:
            print("No Param!")

    # 保存权重文件的函数
    def save_top_models(self, accuracy, path):
        self.top_models.append((accuracy, path))
        self.top_models = sorted(self.top_models, key=lambda x: x[0], reverse=True)
        if len(self.top_models) > self.max_models:
            _, remove_path = self.top_models.pop(-1)
            os.remove(remove_path)

    # 训练
    def train(self, stop_value):
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        log_dir = os.path.join("./logs/ghostnetv2/", current_time)
        self.writer = SummaryWriter(log_dir=log_dir)
        for epoch in range(1, stop_value + 1):
            self.net.train()
            loss_sum = 0.
            for inputs, labels in tqdm(self.train_dataset, desc=f"epoch{epoch}(train)", unit="batch"):
                # label为载入batch为16的类别标签
                # input为载入batch为16的图片，16*18*80*80
                inputs, labels = inputs.float().to(self.device), labels.to(self.device)
                # out为模型输出的batch为16的1*1的类别logits
                out = self.net(inputs)


                # labels[:, 0]将labels [3.],[2.],[1.]转为[3，2，1]；long()将其转为int64方便进行交叉熵损失计算
                loss = self.loss_func(out, labels[:, 0].long())
                loss_sum += loss.item()
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
            avg_train_loss = loss_sum / len(self.train_dataset)
            print("模型损失：" + str(avg_train_loss))
            self.writer.add_scalar("train_loss", avg_train_loss, epoch)

            # if epoch % 20 == 0:
            # out[0]为16*7的第一个1*7数据，torch.argmax返回最大值的索引
            # 对比模型预测和标签
            #     print("网络输出视力：" + str(torch.argmax(out[0]).item()))
            #     print("真实视力：" + str(labels[0].item()))
            #     print(f"\nEpoch: {epoch}/{stop_value}, Loss: {loss}")

            # 备份
            if epoch % 1 == 0 or epoch == stop_value:
                self.net.eval()  # 模型评估
                print('\n\n验证集验证结果：')
                with torch.no_grad():
                    val_loss = 0
                    val_acc = 0
                    val_all = 0
                    for inputs, labels in tqdm(self.val_dataset, desc=f"epoch{epoch}(val)", unit="batch"):
                        inputs, labels = inputs.float().to(self.device), labels.to(self.device)
                        # 输出生成的图像
                        out = self.net(inputs)

                        loss = self.loss_func(out, labels[:, 0].long())
                        val_loss += loss.item()

                        for j in range(len(out.tolist())):
                            val_all += 1
                            out_acc = torch.argmax(out[j], dim=0)
                            if out_acc == labels[j][0]:
                                val_acc += 1
                    val_acc /= val_all
                    avg_val_loss = val_loss / len(self.val_dataset)
                    print('val Loss: {:.6f},    ±0.5 Acc: {:.6f}'.format(avg_val_loss, val_acc))
                    self.writer.add_scalar("val_loss", avg_val_loss, epoch)
                    self.writer.add_scalar("acc/val_acc", val_acc, epoch)

                    print("网络输出视力：" + str(torch.argmax(out[0]).item()))
                    print("真实视力：" + str(labels[0].item()))

                    # 保存准确率最高的三个模型
                    save_path_acc = os.path.join(r'D:\shishai\model\github\refractive_error\classification\GhostNet_V2\params',
                        f'ghostnetv2_val_acc_{val_acc:.3f}_{avg_val_loss:.3f}_epoch{epoch}.plt'
                    )
                    torch.save(self.net.state_dict(), save_path_acc)
                    self.save_top_models(val_acc, save_path_acc)

                    print("val_acc model_copy is saved !")
   # 更新学习率调度器
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                # 如果使用的是 ReduceLROnPlateau，需要传入验证集的指标
                self.lr_scheduler.step(val_acc)
            else:
                # 对于其他调度器，如 StepLR, CosineAnnealingLR 等，直接调用 step()
                self.lr_scheduler.step()

            # 打印当前学习率
            current_lr = self.opt.param_groups[0]['lr']
            print(f"当前学习率: {current_lr}")

if __name__ == '__main__':
    t = Trainer(
        path=r"D:\shishai\NIRDatasets\datasets\dataset",
        model=r'./params/ResNet18_5200_0.7633928571428571_23.444395065307617.plt',
        model_copy=r'./params/fold/v8model_{}_{}_{}.plt',
        img_save_path=r'D:\shishai\UNet\train_img')
    t.train(3000)
