import os
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import argparse

import torch
import torchvision
from torchvision import transforms

from data.dataset import Dataset
from model.hkudetector import resnet50
from utils.loss import yololoss

# parse args
parser = argparse.ArgumentParser()
parser.add_argument('--yolo_S', default=14, type=int, help='YOLO grid num')  # 20 for 640, 14 for 448
parser.add_argument('--yolo_B', default=2, type=int, help='YOLO box num') # large B reduces the loss significantly
parser.add_argument('--yolo_C', default=5, type=int, help='detection class num')

parser.add_argument('--num_epochs', default=5, type=int, help='number of epochs') # 10
parser.add_argument('--batch_size', default=12, type=int, help='batch size') # 12 cannot increase batch_size too much due to limited memory allocation, 14 is almost the maximum batch size, increase batch size reduce grad explosion
parser.add_argument('--learning_rate', default=1e-5, type=float, help='learning rate') # 1e-5 5e-5 is almost the fast otherwise gradient explode without schedular

parser.add_argument('--seed', default=666, type=int, help='random seed')
parser.add_argument('--dataset_root', default='HKU-DASC7606-A1/src/ass1_dataset', type=str, help='dataset root')
parser.add_argument('--output_dir', default='HKU-DASC7606-A1/checkpoints', type=str, help='output directory')

parser.add_argument('--l_coord', default=5., type=float, help='hyper parameter for localization loss')
parser.add_argument('--l_noobj', default=0.5, type=float, help='hyper parameter for no object loss')

# additional argument 
parser.add_argument('--weight_decay', default=5e-8, type=float, help='weight decay for AdamW')
parser.add_argument('--scheduler', default=True, type=bool, help='switch on scheduler')
parser.add_argument('--dropblock', default=False, type=bool, help='switch on DropBlock')
parser.add_argument('--load_pretrain', default=False, type=bool, help='load preset pytorch model')
parser.add_argument('--load_pretrain_epoch', default=True, type=bool, help='load interrupted training')
parser.add_argument('--which_checkpoint', default=5, type=int, help='load the check point saved from the end of which epoch')

args = parser.parse_args()

# fine tune
def load_pretrained(net):
    resnet = torchvision.models.resnet50(pretrained=True)
    resnet_state_dict = resnet.state_dict()
    
    net_dict = net.state_dict()
    for k in resnet_state_dict.keys():
        if k in net_dict.keys() and not k.startswith('fc'):
            net_dict[k] = resnet_state_dict[k]
    net.load_state_dict(net_dict)
    



'''
This code is used to load a pretrained ResNet50 model into another network. The first line loads a pretrained ResNet50 into the "resnet" variable. The second line stores the state dictionary of the loaded ResNet50 model into the "resnet_state_dict" variable. The next line stores the state dictionary of the other network into the "net_dict" variable. The for loop then iterates through each key of the "resnet_state_dict" and checks if the key exists in the "net_dict". If it does, and the key does not start with "fc" (the fully connected layers), it will copy the value of the key in the "resnet_state_dict" to the "net_dict". The last line then uses the updated "net_dict" to load the state dictionary of the other network.
'''

####################################################################
# Environment Setting
# We suggest using only one GPU, or you should change the codes about model saving and loading

# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('NUMBER OF CUDA DEVICES:', torch.cuda.device_count())

# Other settings
# args.load_pretrain = True
print(args)

output_dir = args.output_dir
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

np.random.seed(args.seed)
torch.manual_seed(args.seed)

####################################################################
criterion = yololoss(args, l_coord=args.l_coord, l_noobj=args.l_noobj)

hku_mmdetector = resnet50(args=args)
if args.load_pretrain:
    load_pretrained(hku_mmdetector)
hku_mmdetector = hku_mmdetector.to(device)

####################################################################
# Multiple GPUs if needed
# if torch.cuda.device_count() > 1:
#     hku_mmdetector = torch.nn.DataParallel(hku_mmdetector)



# initialize optimizer
optimizer = torch.optim.AdamW(hku_mmdetector.parameters(), betas=(0.9, 0.999), weight_decay=args.weight_decay, lr=args.learning_rate)

# scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.99 ** epoch)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3, eta_min=0)

scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=3,T_mult=2)
start_epoch = 0
###########   load previous training #########
if args.load_pretrain_epoch:
    checkpoint = torch.load(os.path.join(output_dir, 'hku_mmdetector_epoch_'+str(args.which_checkpoint)+'.pth'))
    hku_mmdetector.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    start_epoch = checkpoint['epoch']
    
hku_mmdetector.train()
################################################
    
# initialize dataset
train_dataset = Dataset(args, split='train', transform=[transforms.ToTensor()])
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

###################################################################
# TODO: Please fill the codes below to initialize the validation dataset
##################################################################

val_dataset = Dataset(args, split='train', transform=[transforms.ToTensor()])
val_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

##################################################################

print(f'NUMBER OF DATA SAMPLES: {len(train_dataset)}')
print(f'BATCH SIZE: {args.batch_size}')

train_dict = dict(iter=[], loss=[])
val_dict = dict(iter=[], loss=[])
best_val_loss = np.inf

for epoch in range(start_epoch, args.num_epochs):
    hku_mmdetector.train()
    
    
    # training
    total_loss = 0.
    print(('\n' + '%10s' * 3) % ('epoch', 'loss', 'gpu'))
    progress_bar = tqdm.tqdm(enumerate(train_loader), total=len(train_loader))
    for i, (images, target) in progress_bar:
        images = images.to(device)
        target = target.to(device)
        
        pred = hku_mmdetector(images)
      
        loss = criterion(pred, target)

        total_loss += loss.data

        ###################################################################
        # TODO: Please fill the codes here to complete the gradient backward
        ##################################################################
        optimizer.zero_grad()
        loss.backward()
        
        # torch.nn.utils.clip_grad_norm_(hku_mmdetector.parameters(), 0.5) # args.clip
        
        optimizer.step()
        
        avg_loss = total_loss / (i + 1)
        if (i + 1) % 5 == 0: # every 5 batch
            print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f, average_loss: %.4f' % (epoch +1, args.num_epochs,
                                                                                 i + 1, len(train_loader), loss.data, avg_loss))
        pass
        
        ##################################################################

        mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)
        s = ('%10s' + '%10.4g' + '%10s') % ('%g/%g' % (epoch + 1, args.num_epochs), total_loss / (i + 1), mem)
        progress_bar.set_description(s)
    
    if args.scheduler == True:
        scheduler.step()

    # validation
    validation_loss = 0.0
    hku_mmdetector.eval()
    progress_bar = tqdm.tqdm(enumerate(val_loader), total=len(val_loader))
    for i, (images, target) in progress_bar:
        images = images.to(device)
        target = target.to(device)

        prediction = hku_mmdetector(images)
        loss = criterion(prediction, target)
        validation_loss += loss.data
    validation_loss /= len(val_loader)
    print("validation loss:", validation_loss.item())

    if best_val_loss > validation_loss:
        best_val_loss = validation_loss

        save = {'state_dict': hku_mmdetector.state_dict(), 
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                "epoch": epoch}
        torch.save(save, os.path.join(output_dir, 'hku_mmdetector_best.pth'))

    save = {'state_dict': hku_mmdetector.state_dict(), 
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                "epoch": epoch}
    torch.save(save, os.path.join(output_dir, 'hku_mmdetector_epoch_'+str(epoch+1)+'.pth')) # start at the 0th epoch, empirically start at the 1st
    
    # save loss data
    train_dict['iter'].append(epoch +1)
    train_dict['loss'].append(np.array(avg_loss.cpu()))
    
    val_dict['iter'].append(epoch +1)
    val_dict['loss'].append(np.array(validation_loss.cpu()))

    torch.cuda.empty_cache()

# save loss data to files and plot graphs   
loss_curve_dir = 'HKU-DASC7606-A1/loss_curve'
if not os.path.exists(loss_curve_dir):
    os.makedirs(loss_curve_dir)
    
np.save(os.path.join(loss_curve_dir, 'B_{}_lr_{}_wd_{}_Sch_{}_db_{}.npy') \
        .format(args.yolo_B, args.learning_rate, args.weight_decay, args.scheduler, args.dropblock), train_dict)
np.save(os.path.join(loss_curve_dir, 'B_{}_lr_{}_wd_{}_Sch_{}_db_{}.npy') \
        .format(args.yolo_B, args.learning_rate, args.weight_decay, args.scheduler, args.dropblock), val_dict)

plt.plot(train_dict['iter'], train_dict['loss'], '.-', label='train')
plt.plot(val_dict['iter'], val_dict['loss'], '.-', label='val')
plt.legend()
plt.xlabel('EPOCH')
plt.ylabel('LOSS')
plt.savefig(os.path.join(loss_curve_dir, 'B_{}_lr_{}_wd_{}_Sch_{}_db_{}.png') \
            .format(args.yolo_B, args.learning_rate, args.weight_decay, args.scheduler, args.dropblock))
plt.show()
    