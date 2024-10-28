import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from fusion import AFF, iAFF, DAF

class DCMModle(nn.Module):
    """Dynamic Convolutional Module used in DMNet.

    Args:
        filter_size (int): The filter size of generated convolution kernel
            used in Dynamic Convolutional Module.
        fusion (bool): Add one conv to fuse DCM output feature.
        in_channels (int): Input channels.
        channels (int): Channels after modules, before conv_seg.
        conv_cfg (dict | None): Config of conv layers.
        norm_cfg (dict | None): Config of norm layers.
        act_cfg (dict): Config of activation layers.
    """

    def __init__(self, x_in_channels, y_in_channels,  channels, filter_size=1, fusion=True):
        super(DCMModle, self).__init__()
        self.filter_size = filter_size
        self.fusion = fusion
        self.x_in_channels = x_in_channels
        self.y_in_channels = y_in_channels
        self.channels = channels

        # Global Information vector
        self.filter = nn.AdaptiveAvgPool2d(self.filter_size)
        self.reduce_conv = nn.Sequential(nn.Conv2d(self.x_in_channels, self.channels, 1),
                                         nn.BatchNorm2d(self.channels),
                                         nn.ReLU(),
                                         nn.Dropout(0.5)
                                         )
        self.filter_gen_conv = nn.Sequential(nn.Conv2d(self.y_in_channels, self.channels, 1, 1,0),
                                             nn.BatchNorm2d(self.channels),
                                             nn.ReLU(),
                                             nn.Dropout(0.5))

        self.activate = nn.Sequential(nn.BatchNorm2d(self.channels),
                                      nn.ReLU()
                                      )
        if self.fusion:
            # self.fusion_conv = nn.Conv2d(self.channels, self.channels, 1)
            self.fusion_conv = nn.Sequential(nn.Conv2d(self.channels, self.channels, 1),
                                             nn.BatchNorm2d(self.channels),
                                             nn.ReLU(),
                                             nn.Dropout(0.5))


    def forward(self, x, y):
        """Forward function."""
        # b, c1, h, w = x.shape
        b, c, h, w = y.shape
        generated_filter = self.filter_gen_conv(self.filter(y)).view(b, self.channels, self.filter_size, self.filter_size)

        x = self.reduce_conv(x)
        c = self.channels
        # [1, b * c, h, w], c = self.channels
        x = x.view(1, b * c, h, w)

        # [b * c, 1, filter_size, filter_size]
        generated_filter = generated_filter.view(b * c, 1, self.filter_size,
                                                 self.filter_size)
        pad = (self.filter_size - 1) // 2
        if (self.filter_size - 1) % 2 == 0:
            p2d = (pad, pad, pad, pad)
        else:
            p2d = (pad + 1, pad, pad + 1, pad)
        x = F.pad(input=x, pad=p2d, mode='constant', value=0)
        # [1, b * c, h, w]
        output = nn.functional.conv2d(input=x, weight=generated_filter, groups=b * c)
        # [b, c, h, w]
        output = output.view(b, c, h, w)
        output = self.activate(output)

        if self.fusion:
            output = self.fusion_conv(output)

        return output

class DCMModuleList(nn.ModuleList):
    def __init__(self, filter_sizes = [1,2,3,6], x_in_channels =2048 , y_in_channels =2048, channels = 512 ):
        super(DCMModuleList, self).__init__()
        self.fiter_sizes = filter_sizes
        self.channels = channels
        for filter_size in self.fiter_sizes:
            self.append(DCMModle(x_in_channels, y_in_channels, channels, filter_size))
    def forward(self, x, y):
        out = []
        for DCM in self:
            DCM_out = DCM(x, y)
            out.append(DCM_out)
        return out

class DMNet(nn.Module):
    def __init__(self, FM, HNC, LNC, num_classes):
        super(DMNet, self).__init__()

        self.filte_sizes = [3]
        num_size = len(self.filte_sizes)

        self.conv1_hsi = nn.Sequential(
            nn.Conv2d(
                in_channels = HNC,
                out_channels = FM*4,
                kernel_size = 3,
                stride = 1,
                padding = 1
            ),
            nn.BatchNorm2d(FM*4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv1_lidar = nn.Sequential(
            nn.Conv2d(
                in_channels = LNC,
                out_channels = FM*4,
                kernel_size = 3,
                stride = 1,
                padding = 1
            ),
            nn.BatchNorm2d(FM*4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(FM*4, FM*2, 3, 1, 1),
            nn.BatchNorm2d(FM*2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.5),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(2*FM, FM, 3, 1, 1),
            nn.BatchNorm2d(FM),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.5),
        )
        #
        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        #
        self.aff_mode1 = AFF(channels=FM, r=4)

        self.DMNet_pyramid_l2h_1 = DCMModuleList( filter_sizes = self.filte_sizes, x_in_channels = FM*4, y_in_channels = FM*4, channels = FM*2 )
        self.con_fusion_l2h_1 = nn.Sequential(
            nn.Conv2d(num_size*FM*2 + FM*4, FM*2, 3, padding=1 ),
            nn.BatchNorm2d(FM*2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.5)
        )
        self.DMNet_pyramid_h2l_1 = DCMModuleList( filter_sizes = self.filte_sizes, x_in_channels = FM*4, y_in_channels = FM*4, channels = FM*2 )
        self.con_fusion_h2l_1 = nn.Sequential(
            nn.Conv2d(num_size*FM*2 + FM*4, FM*2, 3, padding=1 ),
            nn.BatchNorm2d(FM*2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.5)
        )
        self.DMNet_pyramid_l2h_2 = DCMModuleList( filter_sizes = self.filte_sizes, x_in_channels = FM*2, y_in_channels = FM*2, channels = FM*1 )
        self.con_fusion_l2h_2 = nn.Sequential(
            nn.Conv2d(num_size*FM*1 + FM*2, FM*1, 3, padding=1 ),
            nn.BatchNorm2d(FM*1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.5)
        )
        self.DMNet_pyramid_h2l_2 = DCMModuleList( filter_sizes = self.filte_sizes, x_in_channels = FM*2, y_in_channels = FM*2, channels = FM*1 )
        self.con_fusion_h2l_2 = nn.Sequential(
            nn.Conv2d(num_size*FM*1 + FM*2, FM*1, 3, padding=1 ),
            nn.BatchNorm2d(FM*1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.5)
        )
        self.DMNet_pyramid_l2h_3 = DCMModuleList( filter_sizes = self.filte_sizes, x_in_channels = FM*1, y_in_channels = FM*1, channels = FM*1 )
        self.con_fusion_l2h_3 = nn.Sequential(
            nn.Conv2d(num_size*FM*1 + FM*1, FM*1, 3, padding=1 ),
            nn.BatchNorm2d(FM*1),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.5)
        )
        self.DMNet_pyramid_h2l_3 = DCMModuleList( filter_sizes = self.filte_sizes, x_in_channels = FM*1, y_in_channels = FM*1, channels = FM*1 )
        self.con_fusion_h2l_3 = nn.Sequential(
            nn.Conv2d(num_size*FM*1 + FM*1, FM*1, 3, padding=1 ),
            nn.BatchNorm2d(FM*1),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.5)
        )
        self.out1 = nn.Sequential(
            nn.Linear(1*FM*1*1, num_classes),
        )
        self.out2 = nn.Sequential(
            nn.Linear(1*FM*1*1, num_classes),
        )
        self.out3 = nn.Sequential(
            nn.Linear(1*FM*1*1, num_classes),
        )

    def forward(self, x, y):
        x0 = self.conv1_hsi(x)
        y0 = self.conv1_lidar(y)

        DM_out = self.DMNet_pyramid_l2h_1(x0,y0)
        DM_out.append(x0)
        x1 = torch.cat(DM_out, dim = 1)
        x1 = self.con_fusion_l2h_1(x1)
        DM_out = self.DMNet_pyramid_h2l_1(y0,x0)
        DM_out.append(y0)
        y1 = torch.cat(DM_out, dim = 1)
        y1 = self.con_fusion_h2l_1(y1)

        DM_out = self.DMNet_pyramid_l2h_2(x1,y1)
        DM_out.append(x1)
        x2 = torch.cat(DM_out, dim = 1)
        x2 = self.con_fusion_l2h_2(x2)
        DM_out = self.DMNet_pyramid_h2l_2(y1,x1)
        DM_out.append(y1)
        y2 = torch.cat(DM_out, dim = 1)
        y2 = self.con_fusion_h2l_2(y2)

        DM_out = self.DMNet_pyramid_l2h_3(x2,y2)
        DM_out.append(x2)
        x3 = torch.cat(DM_out, dim = 1)
        x3 = self.con_fusion_l2h_3(x3)
        DM_out = self.DMNet_pyramid_h2l_3(y2,x2)
        DM_out.append(y2)
        y3 = torch.cat(DM_out, dim = 1)
        y3 = self.con_fusion_h2l_3(y3)

        xy3 = self.aff_mode1(x3, y3)
        xy3 = self.relu(xy3)

        xy3 = xy3.view(xy3.size(0), -1)
        x3 = x3.view(x3.size(0), -1)
        y3 = y3.view(y3.size(0), -1)

        xy3 = self.out3(xy3)
        x3 = self.out1(x3)
        y3 = self.out2(y3)

        sem = [x3, y3, xy3]
        return sem


def train(net, data_loader, train_gt_onehot, TrainPatch, TestPatch, Label, kwargs,
          display_iter=10):

    BestAcc = 0
    device = kwargs['cuda_device']
    nclasses = kwargs['n_classes']
    epoches = kwargs['epoch']
    learning_rate = kwargs['lr']
    apha, gamma, lamda = kwargs['apha'], kwargs['gamma'], kwargs['lamda']

    net.to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    num_train = train_gt_onehot.numpy().shape[0]

    test_acc = []
    train_loss = []
    for e in range(epoches):
        avg_loss = 0.
        for batch_idx, (data_hsi, data_lidar, target_hsi) in enumerate(data_loader):
            data_hsi, data_lidar = data_hsi.cuda(), data_lidar.cuda()
            target_hsi = target_hsi.cuda()

            optimizer.zero_grad()
            [semantic_hsi, semantic_lidar, semantic_fusion] = net(data_hsi, data_lidar)

            loss_h = criterion(semantic_hsi, target_hsi)
            loss_l = criterion(semantic_lidar, target_hsi)
            loss_f = criterion(semantic_fusion, target_hsi)

            loss = apha*loss_h + gamma*loss_l + lamda*loss_f

            loss.backward()
            optimizer.step()

            avg_loss += loss.item()

            ###############   for weighted desicion fusion   ############################
            if batch_idx % 100 == 0:
                net.eval()
                TrainLabel, TestLabel =  Label[0], Label[1]
                TrainPatch1, TrainPatch2 = TrainPatch[0], TrainPatch[1]
                TestPatch1, TestPatch2 = TestPatch[0], TestPatch[1]
                TrainPatch1, TrainPatch2 = TrainPatch1.cuda(), TrainPatch2.cuda()
                ###############   for weighted desicion fusion   ############################
                [temp3, temp4, temp5]= net(TrainPatch1, TrainPatch2)

                pred_y1 = torch.max(temp3, 1)[1].squeeze()
                pred_y1 = pred_y1.cpu()
                acc1 = torch.sum(pred_y1 == TrainLabel).type(torch.FloatTensor) / TrainLabel.size(0)

                pred_y2 = torch.max(temp4, 1)[1].squeeze()
                pred_y2 = pred_y2.cpu()
                acc2 = torch.sum(pred_y2 == TrainLabel).type(torch.FloatTensor) / TrainLabel.size(0)

                pred_y3 = torch.max(temp5, 1)[1].squeeze()
                pred_y3 = pred_y3.cpu()
                acc3 = torch.sum(pred_y3 == TrainLabel).type(torch.FloatTensor) / TrainLabel.size(0)

                Classes = np.unique(TrainLabel)
                w1 = np.empty(len(Classes),dtype='float32')
                w2 = np.empty(len(Classes),dtype='float32')
                w3 = np.empty(len(Classes),dtype='float32')

                for i in range(len(Classes)):
                    cla = Classes[i]
                    right1 = 0
                    right2 = 0
                    right3 = 0
                    # right4 = 0
                    for j in range(len(TrainLabel)):
                        if TrainLabel[j] == cla and pred_y1[j] == cla:
                            right1 += 1
                        if TrainLabel[j] == cla and pred_y2[j] == cla:
                            right2 += 1
                        if TrainLabel[j] == cla and pred_y3[j] == cla:
                            right3 += 1
                    w1[i] = right1.__float__() / (right1 + right2 + right3 + 0.00001).__float__()
                    w2[i] = right2.__float__() / (right1 + right2 + right3 + 0.00001).__float__()
                    w3[i] = right3.__float__() / (right1 + right2 + right3 + 0.00001).__float__()

                w1 = torch.from_numpy(w1).cuda()
                w2 = torch.from_numpy(w2).cuda()
                w3 = torch.from_numpy(w3).cuda()

                # w1,w2,w3 = 1,1,1
                pred_y = np.empty((len(TestLabel)), dtype='float32')
                test_bs = 5000
                number = len(TestLabel) // test_bs
                for i in range(number):
                    temp = TestPatch1[i * test_bs:(i + 1) * test_bs, :, :, :]
                    temp = temp.to(device)
                    temp1 = TestPatch2[i * test_bs:(i + 1) * test_bs, :, :, :]
                    temp1 = temp1.to(device)
                    sem = net(temp, temp1)
                    temp2 = w3 * sem[2] + w2 * sem[1] + w1 * sem[0]
                    temp3 = torch.max(temp2, 1)[1].squeeze()
                    pred_y[i * test_bs:(i + 1) * test_bs] = temp3.cpu()
                    del temp, temp2, temp3

                if (i + 1) * test_bs < len(TestLabel):
                    temp = TestPatch1[(i + 1) * test_bs:len(TestLabel), :, :, :]
                    temp = temp.to(device)
                    temp1 = TestPatch2[(i + 1) * test_bs:len(TestLabel), :, :, :]
                    temp1 = temp1.to(device)
                    sem = net(temp, temp1)
                    temp2 = w3 * sem[2] + w2 * sem[1] + w1 * sem[0]
                    temp3 = torch.max(temp2, 1)[1].squeeze()
                    pred_y[(i + 1) * test_bs:len(TestLabel)] = temp3.cpu()
                    del temp, temp2, temp3
                pred_y = torch.from_numpy(pred_y).long()
                accuracy = torch.sum(pred_y == TestLabel).type(torch.FloatTensor) / TestLabel.size(0)
                test_acc.append(accuracy.item())
                print('Epoch: ', e, '| train loss: %.4f' % loss.data.cpu().numpy(),
                      '| test accuracy: %.4f' % accuracy)

                if accuracy > BestAcc:
                    torch.save(net.state_dict(), 'net_params_FusionWeight.pkl')
                    BestAcc = accuracy
                    w1B = w1
                    w2B = w2
                    w3B = w3
                weights = (w1B, w2B, w3B)
                net.train()
            ###########################################################################################

        avg_loss /= len(data_loader)
        train_loss.append(avg_loss)

        # weights = [1,1,1]
        optimizer = AdjustLearningRate(optimizer, e, learning_rate)
    return net, weights,train_loss, test_acc

def AdjustLearningRate(optimizer, e, learning_rate):
    lr = learning_rate * (0.1 ** (e // 80 ))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

def val(net, weights, TestPatch1, TestPatch2, TestLabel):
    net.load_state_dict(torch.load('net_params_FusionWeight.pkl'))
    net.eval()
    device = 0
    w1, w2, w3 = weights[0], weights[1], weights[2]
    pred_y = np.empty((len(TestLabel)), dtype='float32')
    fea = np.empty((len(TestLabel), 64), dtype='float32')
    test_bs = 64
    number = len(TestLabel) // test_bs
    for i in range(number):
        temp = TestPatch1[i * test_bs:(i + 1) * test_bs, :, :, :]
        temp = temp.cuda()
        temp1 = TestPatch2[i * test_bs:(i + 1) * test_bs, :, :, :]
        temp1 = temp1.cuda()

        sem = net(temp, temp1)
        temp2 = w3 * sem[2] + w2 * sem[1] + w1 * sem[0]
        temp3 = torch.max(temp2, 1)[1].squeeze()
        pred_y[i * test_bs:(i + 1) * test_bs] = temp3.cpu()

        del temp, temp2, temp3

    if (i + 1) * test_bs < len(TestLabel):
        temp = TestPatch1[(i + 1) * test_bs:len(TestLabel), :, :, :]
        temp = temp.cuda()
        temp1 = TestPatch2[(i + 1) * test_bs:len(TestLabel), :, :, :]
        temp1 = temp1.cuda()

        sem = net(temp, temp1)
        temp2 = w3 * sem[2] + w2 * sem[1] + w1 * sem[0]
        # temp2 = w3 * net(temp, temp1)[2] + w2 * net(temp, temp1)[1] + w1 * net(temp, temp1)[0]
        temp3 = torch.max(temp2, 1)[1].squeeze()
        pred_y[(i + 1) * test_bs:len(TestLabel)] = temp3.cpu()

        del temp, temp2, temp3
    pred_y = torch.from_numpy(pred_y).long()
    OA = torch.sum(pred_y == TestLabel).type(torch.FloatTensor) / TestLabel.size(0)
    return OA, pred_y
