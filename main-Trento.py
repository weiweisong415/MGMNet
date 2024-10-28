import torch.utils.data as dataf
import torch
import scipy.io as sio
import numpy as np
import funcs
import argparse
from sklearn.decomposition import PCA
import time
from thop import profile

parser = argparse.ArgumentParser(description='HSI classification with different methods')
parser.add_argument('--patch_size', default=11, type=int,
                    help='the patch size of network input')
parser.add_argument('--batch_size', default=64, type=int,
                    help='the batch size of network training')
parser.add_argument('--epoch', default=200, type=int,
                    help='maximum iteration (default: 50)')
parser.add_argument('--lr', default=0.01, type=float,
                    help='learning rate (default: 0.001)')
parser.add_argument('--apha', default=1, type=float,
                    help='balance for total loss')
parser.add_argument('--gamma', default=1, type=float,
                    help='balance for total loss')
parser.add_argument('--lamda', default=1, type=float,
                    help='balance for total loss')
parser.add_argument('--FM', default=96, type=int,
                    help='number of based bands')
parser.add_argument('--runs', type=int, default=1, help="Number of runs (default: 1)")


args = parser.parse_args()
CUDA_DEVICE = torch.device('cuda:{}'.format(0))
hyperparams = vars(args)

# loading data
hsi = sio.loadmat('./data/Trento/HSI.mat')['HSI']
lidar = sio.loadmat('data/Trento/LiDAR.mat')['LiDAR']
map = sio.loadmat('data/Trento/Trento_TR_50_1.mat')
train_gt = map['TRLabel']
test_gt = map['TSLabel']

N_CLASSES = 6
hsi = hsi.astype(np.float32)
lidar = lidar.astype(np.float32)

# preprocessing data, including PCA,normalization, and padding
PATCH_SIZE = hyperparams['patch_size']
padded_size = PATCH_SIZE // 2

[m, n, l] = hsi.shape
for i in range(l):
    minimal = hsi[:, :, i].min()
    maximal = hsi[:, :, i].max()
    hsi[:, :, i] = (hsi[:, :, i] - minimal)/(maximal - minimal)

NC = 20
PC = np.reshape(hsi, (m*n, l))
pca = PCA(n_components=NC, copy=True, whiten=False)
PC = pca.fit_transform(PC)
hsi = np.reshape(PC, (m, n, NC))

temp = hsi[:,:,0]
temp2 = np.pad(temp, padded_size, 'symmetric')
[m2,n2] = temp2.shape
x1 = np.empty((m2,n2,NC),dtype='float32')
for i in range(NC):
    temp = hsi[:,:,i]
    temp2 = np.pad(temp, padded_size, 'symmetric')
    x1[:,:,i] = temp2
padded_hsi = x1

if len(lidar.shape) == 2:
    l2 = 1
    minimal = lidar.min()
    maximal = lidar.max()
    lidar = (lidar - minimal) / (maximal - minimal)
    padded_lidar = np.pad(lidar, padded_size, 'symmetric')
if len(lidar.shape) == 3:
    [m, n, l2] = lidar.shape
    x2 = np.empty((m2, n2, l2), dtype='float32')
    for i in range(l2):
        minimal = lidar[:, :, i].min()
        maximal = lidar[:, :, i].max()
        lidar[:, :, i] = (lidar[:, :, i] - minimal) / (maximal - minimal)
        temp2 = np.pad(lidar[:, :, i], padded_size, 'symmetric')
        x2[:, :, i] = temp2
    padded_lidar = x2
# lidar = lidar[:,:, np.newaxis]

hyperparams.update({'n_classes': N_CLASSES,'cuda_device': CUDA_DEVICE})

results = []
OA1 = []
OA2 = []
# run the experiment several times
for run in range(args.runs):
    ##################   generating true label of training and test samples ####################################
    [ind1, ind2] = np.where(train_gt != 0)
    TrainNum = len(ind1)
    TrainPatch1 = np.empty((TrainNum, NC, PATCH_SIZE, PATCH_SIZE), dtype='float32')
    TrainPatch2 = np.empty((TrainNum, l2, PATCH_SIZE, PATCH_SIZE), dtype='float32')
    TrainLabel = np.empty(TrainNum)
    for i in range(TrainNum):
        patch1 = padded_hsi[ind1[i] :(ind1[i] + PATCH_SIZE ), ind2[i] :(ind2[i] + PATCH_SIZE), :]
        patch1 = np.reshape(patch1, (PATCH_SIZE * PATCH_SIZE, NC))
        patch1 = np.transpose(patch1)
        patch1 = np.reshape(patch1, (NC, PATCH_SIZE, PATCH_SIZE))
        TrainPatch1[i, :, :, :] = patch1
        patch2 = padded_lidar[ind1[i] :(ind1[i] + PATCH_SIZE), (ind2[i] ):(ind2[i] + PATCH_SIZE)]
        patch2 = np.reshape(patch2, (PATCH_SIZE * PATCH_SIZE, l2))
        patch2 = np.transpose(patch2)
        patch2 = np.reshape(patch2, (l2, PATCH_SIZE, PATCH_SIZE))
        TrainPatch2[i, :, :, :] = patch2
        patchlabel = train_gt[ind1[i], ind2[i]]
        TrainLabel[i] = patchlabel
    TrainLabel = TrainLabel-1
    TrainLabel = torch.from_numpy(np.asarray(TrainLabel, dtype='int64'))
    train_gt_onehot = torch.FloatTensor(TrainLabel.size(0), N_CLASSES)
    train_gt_onehot.zero_()
    train_gt_onehot.scatter_(1, TrainLabel.view(-1, 1), 1)

    [ind1, ind2] = np.where(test_gt != 0)
    TestNum = len(ind1)
    TestPatch1 = np.empty((TestNum, NC, PATCH_SIZE, PATCH_SIZE), dtype='float32')
    TestPatch2 = np.empty((TestNum, l2, PATCH_SIZE, PATCH_SIZE), dtype='float32')
    TestLabel = np.empty(TestNum)
    for i in range(TestNum):
        patch1 = padded_hsi[(ind1[i]):(ind1[i] + PATCH_SIZE), (ind2[i] ):(ind2[i] + PATCH_SIZE), :]
        patch1 = np.reshape(patch1, (PATCH_SIZE * PATCH_SIZE, NC))
        patch1 = np.transpose(patch1)
        patch1 = np.reshape(patch1, (NC, PATCH_SIZE, PATCH_SIZE))
        TestPatch1[i, :, :, :] = patch1
        patch2 = padded_lidar[(ind1[i] ):(ind1[i] + PATCH_SIZE), (ind2[i] ):(ind2[i] + PATCH_SIZE)]
        patch2 = np.reshape(patch2, (PATCH_SIZE * PATCH_SIZE, l2))
        patch2= np.transpose(patch2)
        patch2 = np.reshape(patch2, (l2, PATCH_SIZE, PATCH_SIZE))
        TestPatch2[i, :, :, :] = patch2
        patchlabel = test_gt[ind1[i], ind2[i]]
        TestLabel[i] = patchlabel
    TestLabel = TestLabel-1
    TestLabel = torch.from_numpy(np.asarray(TestLabel, dtype='int64'))
    test_gt_onehot = torch.FloatTensor(TestLabel.size(0), N_CLASSES)
    test_gt_onehot.zero_()
    test_gt_onehot.scatter_(1, TestLabel.view(-1, 1), 1)
    ##################################################################################################################

    TrainPatch1 = torch.from_numpy(TrainPatch1)
    TrainPatch2 = torch.from_numpy(TrainPatch2)
    TestPatch1 = torch.from_numpy(TestPatch1)
    TestPatch2 = torch.from_numpy(TestPatch2)
    TrainPatch = (TrainPatch1, TrainPatch2)
    TestPatch = (TestPatch1, TestPatch2)
    Label = (TrainLabel, TestLabel)

    # Generate the dataset
    dataset = dataf.TensorDataset(TrainPatch1, TrainPatch2, TrainLabel)
    train_loader = dataf.DataLoader(dataset, batch_size=hyperparams['batch_size'], shuffle=True)

    FM = hyperparams['FM']
    model = funcs.DMNet(FM, NC, l2, N_CLASSES)
    ############           compute the model complexity   ##########################################
    input_data1 = torch.randn(1,20,11,11)
    input_data2 = torch.randn(1, 1, 11, 11)
    input_data = (input_data1, input_data2)
    Flops, params = profile(model, input_data)
    print('Flops: % .4fM'%(Flops / 1000000))
    print('params: % .4fM' % (params / 1000000))
    ################################################################################################
    model.cuda()

    print(model)
    start = time.time()
    # training network
    model, weights, train_loss, test_acc = funcs.train(model, train_loader, train_gt_onehot, TrainPatch, TestPatch, Label, hyperparams)

    end = time.time()
    train_time = end - start
    ##  save training loss and test accuracy values   ####
    # with open("results/Trento/train_loss.txt", 'w') as loss:
    #     loss.write(str(train_loss))
    # with open("results/Trento/test_acc.txt", 'w') as acc:
    #     acc.write(str(test_acc))

    start = time.time()
    oa, probs= funcs.val(model, weights, TestPatch1, TestPatch2, TestLabel)
    end = time.time()
    test_time = end - start

    # sio.savemat('results/Trento/results_num50_' + str(run+1)+'.mat',
    #             {'pred_lab': probs.numpy(), 'true_lab': TestLabel.numpy()})
    # results.append(oa.numpy())

    print(oa)
    print('The Training time is: ', train_time)
    print('The Test time is: ', test_time)

print(results)


