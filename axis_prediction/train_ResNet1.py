from datetime import datetime
import os
# from model import ghostnet
from model_resnet import RetNet18
import torch
import datasets_gray
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.utils.data import DataLoader

# 训练器
class Trainer:
    def __init__(self, path, model, model_copy, img_save_path):
        self.path = path
        self.model = model
        self.model_copy = model_copy
        self.img_save_path = img_save_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = RetNet18().to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=0.00001, weight_decay=0.0001)
        # self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.opt, step_size=500, gamma=0.1, last_epoch=-1)
        self.loss_func = nn.SmoothL1Loss()
        self.top_models = []  # 存储最高准确率的模型
        self.max_models = 3   # 保持最多3个模型文件

        trainset_list = [line.strip() for line in open(os.path.join(path, 'train.txt'), encoding='gbk')]
        valset_list = [line.strip() for line in open(os.path.join(path, 'val.txt'), encoding='gbk')]

        self.train_dataset = DataLoader(datasets_gray.Datasets(trainset_list, path, True), batch_size=16, shuffle=True, num_workers=0)
        self.val_dataset = DataLoader(datasets_gray.Datasets(valset_list, path, False), batch_size=16, shuffle=False, num_workers=0)

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
        log_dir = os.path.join("./logs/resnet_axis/", current_time)
        self.writer = SummaryWriter(log_dir=log_dir)

        for epoch in range(1, stop_value + 1):
            loss_sum = 0.
            for inputs, labels in tqdm(self.train_dataset, desc=f"epoch{epoch}(train)", unit="batch"):
                inputs, labels = inputs.float().to(self.device), labels.to(self.device)
                out = self.net(inputs)
                loss = self.loss_func(out, labels)
                loss_sum += loss.item()
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
            print("模型损失：" + str(loss_sum / len(self.train_dataset)))
            self.writer.add_scalar("train_loss", loss_sum / len(self.train_dataset), epoch)

            # 备份
            if epoch % 1 == 0 or epoch == stop_value:
                self.net.eval()
                print('\n\n验证集验证结果：')
                with torch.no_grad():
                    val_loss = 0
                    val_acc = 0
                    val_all = 0
                    for inputs, labels in tqdm(self.val_dataset, desc=f"epoch{epoch}(val)", unit="batch"):
                        inputs, labels = inputs.float().to(self.device), labels.to(self.device)
                        out = self.net(inputs)
                        loss = self.loss_func(out, labels)
                        val_loss += loss.item()
                        for j in range(len(out)):
                            val_all += 1
                            #轴位为1维
                            ax = abs(out[j] - labels[j])
                            if ax <= 20:
                                val_acc += 1
                    val_acc /= val_all

                    print('val Loss: {:.6f},           ±0.5D Acc: {:.6f}'.format(val_loss / (len(self.val_dataset)), val_acc))
                    self.writer.add_scalar("val_loss", val_loss / len(self.val_dataset), epoch)
                    self.writer.add_scalar("acc/val_acc", val_acc, epoch)

                    # 保存准确率最高的三个模型
                    save_path_acc = r'D:\shishai\model\github\refractive_error\axis_prediction\params_resnet/resnet_axis_val_acc_{:.3f}_{:.3f}_epoch{}.plt'.format(
                        val_acc, val_loss / (len(self.val_dataset)), epoch)
                    torch.save(self.net.state_dict(), save_path_acc)
                    self.save_top_models(val_acc, save_path_acc)

                    print("val_acc model_copy is saved !")

if __name__ == '__main__':
    t = Trainer(r"D:\shishai\NIRDatasets\datasets\dataset", r'D:\shishai\model\ghostnet\params/model1_0_0.244140625_459.7992205619812.plt', r'./params/model1_{}_{}_{}.plt', img_save_path=r'D:\shishai\UNet\train_img')
    t.train(1000)

