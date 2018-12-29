
#input skip, residual in mid, subpixel, see interm noise
# dice erros *2 - is there a different in optimising 1-2*d and 1-d


#inputs skipped  + mid skip
#noisy in val+train

batch_size = 2
batch_size_val=2
init_lr = 0.01
train_size=None    # Set to None to use all     # DataLoader: train size
if train_size==None: opt_step_size=20*130 #15*125  # scheduler=for StepLR
else:opt_step_size=30*(train_size/batch_size)   

gpu_number = "0"            # which gpu to run on( as str 0-2)
noisy_sigma = 0.3             # gaussian noise std
noisy_smooth=2.0    # gaussian_filter smooth
noisy_mean=0      # gaussian_filter mean
standardizer_type=['mode'] # options: 'he' and ('mode' or mean' or 'median')  override order

batch_norm=False # If False instance norm is taken

leaky_relu=True;relu=False;elu=False # activations in overridding order

grad_accu_times = 2
epochs=800
batch_size = 2
batch_size_val=2

opt_gamma=0.5           # scheduler gamma
dropout_toggel,dropblock_toggle=False, True   # only one true, dropblock overrides dropout
dropblock_blocks=3
dropblock_mid=7

dropout_amount=0.3

save_initial='unetmodularinst'

# opt on indv dices/weighted or reg dice
#SIZES
# ('input unet', (2, 4, 192, 192, 128))
# ('first down', (2, 4, 96, 96, 64))
# ('BlockU 4 shape:', (2, 8, 96, 96, 64))
# ('unet output', (2, 4, 192, 192, 128))


import os
from DropBlock3d import DropBlock2D, DropBlock3D,LinearScheduler
import gzip
from sh import gunzip
import numpy as np
import pandas as pd
import gzip
import nibabel as nib
import matplotlib.pyplot as plt
import io
import torch
import torch.utils#.data.Dataset
import glob
#from sklearn.preprocessing import OneHotEncoder 
import imgaug as ia
from imgaug import augmenters as iaa
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
from scipy import ndimage as nd
import torch as torch
import torch
from torch.autograd import Variable
import warnings
warnings.filterwarnings('ignore', '.*output shape of zoom.*')

from random import shuffle
 
 
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]=gpu_number

import time
from torch.optim.lr_scheduler import StepLR
import torch as torch
from skimage import exposure
from tensorboardX import SummaryWriter
from tumor_patch_function import get_dim, tumor_patch, non_tumor_patch
writer = SummaryWriter()


hgg_dir='/home/Drive/rahul/MICCAI_BraTS_2018_Data_Training/HGG/'
lgg_dir='/home/Drive/rahul/MICCAI_BraTS_2018_Data_Training/LGG/'
csv_dir='/home/Drive/rahul/MICCAI_BraTS_2018_Data_Training/brats.csv'
val_dir='/home/Drive/rahul/MICCAI_BraTS_2018_Data_Training/brats_val/'



def noisy(image): # only for inputs

    d,row,col,ch= image.shape   
    gauss = np.random.normal(noisy_mean,noisy_sigma,(d,row,col,ch))
    gauss = gauss.reshape(d,row,col,ch)
    noisy = image + gauss
    noisy = nd.gaussian_filter(noisy, sigma=(noisy_smooth,noisy_smooth, noisy_smooth, noisy_smooth), order=0)

    return noisy

def mode_std(z):
    mode=z.float().flatten().mode()[0]
    std=(z-mode)**2
    return torch.sqrt(std.sum()/len(std))
    

def standardizer(img,st=standardizer_type):

    img=torch.clamp(torch.tensor(img),0,255).float()
    if 'he' in st: img=exposure.equalize_hist(np.array(img)); img=torch.tensor(img)
    if 'mode' in st: mean=img.float().flatten().mode()[0] ; std=mode_std(img)
    if 'mean' in st : mean=img.mean(); std=img.std()
    if 'median' in st : mean=img.median(); std=img.std()
    
    img=img-mean
    img=img/std

    return(img)

def one_hot(a):
    g1=a[:,:,:]==0
    g2=a[:,:,:]==1
    g3=a[:,:,:]==2
    g4=a[:,:,:]==3
    return torch.stack((g1,g2,g3,g4),0)


 
def read_img(flair,t1,t1ce,t2,seg,mode='train'):
            flair = nib.load(flair)   # read image
            t1 = nib.load(t1)
            t1ce = nib.load(t1ce)
            t2 = nib.load(t2)
            seg = nib.load(seg)

            flair = np.array(flair.dataobj)[20:212,20:212,15:143] ; t1 = np.array(t1.dataobj)[20:212,20:212,15:143] 
            t1ce= np.array(t1ce.dataobj)[20:212,20:212,15:143] ; t2 = np.array(t2.dataobj)[20:212,20:212,15:143] 
            seg = np.array(seg.dataobj)[20:212,20:212,15:143]  
                
            # normalise
            flair=standardizer(flair) ; t1=standardizer(t1); t1ce=standardizer(t1ce); t2=standardizer(t2)
            # zoom=0.5
            # seg = nd.interpolation.zoom(seg, zoom=zoom) # creates more classes
            seg[seg==4]=3
            
            
            img=np.stack([flair,t1,t1ce,t2])

            # if mode =='train':
            img=noisy(img)
            # concat_full=np.concatenate([concat,seg],axis=2)
#             images_aug=self.seq_seg.augment_images(concat_full)
            # images_aug=concat_full

            
            seg_orig=torch.clamp(torch.from_numpy(seg),0,3)
#             img,seg_orig=tumor_patch(torch.from_numpy(img),seg_orig)  


            seg=one_hot(seg_orig) 
            
            a=np.random.choice([0,1])
            if a==0: img,seg_orig,seg=non_tumor_patch(torch.tensor(img),seg_orig,seg)
            else: img,seg_orig,seg=tumor_patch(torch.tensor(img),seg_orig,seg)
#             print('shapes: ', img.size(),seg_orig.size(),seg.size() )


            return(img,seg,seg_orig)

class BraTS_FLAIR_val(torch.utils.data.Dataset):  # Validation dataset getter
 

    def __init__(self, csv_file, root_dir, transform=None):  
        self.paths = glob.glob('/home/Drive/rahul/MICCAI_BraTS_2018_Data_Training/brats_val/**')
        self.root_dir = root_dir
        self.transform = transform
        # self.val=val

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx): 
        
        
        flair=self.paths[idx]+'/'+str(self.paths[idx])[-17:]+'_flair.nii'  # image files location
        t1=self.paths[idx]+'/'+str(self.paths[idx])[-17:]+'_t1.nii'
        t1ce=self.paths[idx]+'/'+str(self.paths[idx])[-17:]+'_t1ce.nii'
        t2=self.paths[idx]+'/'+str(self.paths[idx])[-17:]+'_t2.nii'        
        seg=self.paths[idx]+'/'+str(self.paths[idx])[-17:]+'_seg.nii'
        

        img,seg,seg_orig=read_img(flair=flair,t1=t1,t1ce=t1ce,t2=t2,seg=seg)

        # img=torch.from_numpy(img)


        sample = {'img': img, 'mask': seg.type(torch.ByteTensor),'seg_orig':seg_orig.type(torch.ByteTensor)}
       
        if self.transform:
            sample = self.transform(sample)

        return sample

class BraTS_FLAIR(torch.utils.data.Dataset):


    def __init__(self, csv_file, root_dir, transform=None,train_size=train_size):  
        # self.brats = pd.read_csv(csv_file)
        # self.brats=self.brats.iloc[1:,:]
        self.root_dir = root_dir
        self.transform = transform
        # self.val=val
        # sometimes=lambda aug: iaa.Sometimes(0.5, aug)
        # self.seq=iaa.Sequential([
        #     iaa.Sometimes(0.2,iaa.OneOf(iaa.GaussianBlur(sigma=(1, 1.3)),iaa.AverageBlur(k=((5, 5), (1, 3))))) , # gauss OR avg blue 
        #     iaa.Sometimes(0.2,iaa.Dropout((0.03, 0.08))) ])

        self.hggs=glob.glob(hgg_dir+'*')
        self.lggs=glob.glob(lgg_dir+'*')
        self.brats=self.hggs+self.lggs
        if train_size: self.brats=self.brats[:train_size]  #  chosen subset
        shuffle(self.brats)
          

    def __len__(self):                                    
        return len(self.brats)

    def __getitem__(self, idx): 


        folder=self.brats[idx]
 
        flair=self.brats[idx]+'/'+self.brats[idx][54:]+'_flair.nii.gz'  
        t1=self.brats[idx]+'/'+self.brats[idx][54:]+'_t1.nii.gz'
        t1ce=self.brats[idx]+'/'+self.brats[idx][54:]+'_t1ce.nii.gz'
        t2=self.brats[idx]+'/'+self.brats[idx][54:]+'_t2.nii.gz'
        seg=self.brats[idx]+'/'+self.brats[idx][54:]+'_seg.nii.gz'
        
        # unzip if not already unzipped 
        try: gunzip(flair)
        except: pass
        try:gunzip(t1)
        except:pass
        try:gunzip(t1ce)
        except:pass
        try:gunzip(t2)
        except:pass
        try:gunzip(seg)
        except:pass
        
        flair=self.brats[idx]+'/'+self.brats[idx][54:]+'_flair.nii'  # image files location
        t1=self.brats[idx]+'/'+self.brats[idx][54:]+'_t1.nii'
        t1ce=self.brats[idx]+'/'+self.brats[idx][54:]+'_t1ce.nii'
        t2=self.brats[idx]+'/'+self.brats[idx][54:]+'_t2.nii'
        seg=self.brats[idx]+'/'+self.brats[idx][54:]+'_seg.nii'

        img,seg,seg_orig= read_img(flair=flair,t1=t1,t1ce=t1ce,t2=t2,seg=seg)

        sample = {'img': img, 'mask': seg.type(torch.ByteTensor),'seg_orig':seg_orig.type(torch.ByteTensor)}
        
        if self.transform:
            sample = self.transform(sample)

        return sample
 
# same as simplified_mnc with non weighted regular dice loss

# with transpose down conv and up conv in beginning interpolation and up blocks



 
brats=pd.read_csv('/home/Drive/rahul/MICCAI_BraTS_2018_Data_Training/brats.csv')
hgg_dir='/home/Drive/rahul/MICCAI_BraTS_2018_Data_Training/HGG/'
lgg_dir='/home/Drive/rahul/MICCAI_BraTS_2018_Data_Training/LGG/'
csv_dir='/home/Drive/rahul/MICCAI_BraTS_2018_Data_Training/brats.csv'
val_dir='/home/Drive/rahul/MICCAI_BraTS_2018_Data_Training/brats_val/'

class my_norm(torch.nn.Module):
	def __init__(self, channels):
		super(my_norm,self).__init__()
		self.norm=torch.nn.BatchNorm3d(channels) if batch_norm==True else torch.nn.InstanceNorm3d(channels, affine=True)

	def forward(self,x): return(self.norm(x))


class my_activation(torch.nn.Module):
	def __init__(self):
		super(my_activation,self).__init__()
		if leaky_relu:
			self.activation=torch.nn.LeakyReLU(inplace=True) 
		if relu:
			self.activation=torch.nn.ReLU(inplace=True) 
		if elu:
			self.activation=torch.nn.ELU(inplace=True) 
	def forward(self,x):return(self.activation(x))



class UNet_down_block(torch.nn.Module):
    def __init__(self, input_channel, output_channel, down_size):
        super(UNet_down_block, self).__init__()
        self.conv1 = torch.nn.Conv3d(input_channel, output_channel, 3, padding=1)
        self.bn1 = my_norm(output_channel) 
        self.conv2 = torch.nn.Conv3d(output_channel, output_channel, 3, padding=1)
        self.bn2 = my_norm(output_channel)
        self.conv3 = torch.nn.Conv3d(output_channel, output_channel, 3, padding=1)
        self.bn3 = my_norm(output_channel)
        self.max_pool = torch.nn.MaxPool3d(2, 2)
        self.relu = my_activation()
        self.down_size = down_size
        if dropout_toggel==True: self.dropout=torch.nn.Dropout3d(dropout_amount)
        if dropblock_toggle==True: self.dropout=LinearScheduler(DropBlock3D(block_size=dropblock_blocks, drop_prob=0.),
                start_value=0.,
                stop_value=0.25,
                nr_steps=5)


    def forward(self, x):
        if self.down_size:
            x = self.max_pool(x)
        x = self.dropout(self.bn1(self.relu(self.conv1(x))))
        residual=x.clone()
        x = self.dropout(self.bn2(self.relu(self.conv2(x))))
        x = self.dropout(self.bn3(self.relu(self.conv3(x))))
        x=x+residual
        return x

class UNet_up_block(torch.nn.Module):
    def __init__(self, prev_channel, input_channel, output_channel):
        super(UNet_up_block, self).__init__()
#         self.up_sampling = torch.nn.functional.interpolate(scale_factor=2, mode='trilinear')
        self.conv1 = torch.nn.Conv3d(prev_channel + input_channel, output_channel, 3, padding=1)
        self.bn1 = my_norm(output_channel)
        self.conv2 = torch.nn.Conv3d(output_channel, output_channel, 3, padding=1)
        self.bn2 = my_norm(output_channel)
        self.conv3 = torch.nn.Conv3d(output_channel, output_channel, 3, padding=1)
        self.bn3 = my_norm(output_channel)
        self.relu = my_activation()
        if dropout_toggel==True: self.dropout=torch.nn.Dropout3d(dropout_amount)
        if dropblock_toggle==True: self.dropout=LinearScheduler(DropBlock3D(block_size=dropblock_blocks, drop_prob=0.),
                start_value=0.,
                stop_value=0.25,
                nr_steps=5)

        self.trans=torch.nn.ConvTranspose3d(input_channel,input_channel,2,2,padding=0)

    def forward(self, prev_feature_map, x):
        # print('before trans',x.size())
        x=self.trans(x)
        x = torch.cat((x, prev_feature_map), dim=1)
        x = self.dropout(self.bn1(self.relu(self.conv1(x))))
        residual=x.clone()
        x = self.dropout(self.bn1(self.relu(self.conv2(x))))
        x = self.dropout(self.bn1(self.relu(self.conv3(x))))
        x=x+residual
        return x


class UNet(torch.nn.Module):        # input -> shrunk in 1/2 then transposed at end. Loss can be with orig seg label / shurken if remove transpose
    def __init__(self):
        super(UNet, self).__init__()

        # self.bn0 = torch.nn.BatchNorm3d(4)

        self.down_block1 = UNet_down_block(4, 24, False)
        self.down_block2 = UNet_down_block(24, 72, True)
        self.down_block3 = UNet_down_block(72, 148, True)
        self.down_block4 = UNet_down_block(148, 244, True)
        self.max_pool = torch.nn.MaxPool3d(2, 2)

        self.mid_conv1 = torch.nn.Conv3d(244, 244, 3, padding=1)
        # self.bn1 = torch.nn.BatchNorm3d(144)
        self.mid_conv2 = torch.nn.Conv3d(244, 244, 3, padding=1)
        # self.bn2 = torch.nn.BatchNorm3d(144)
        self.mid_conv3 = torch.nn.Conv3d(244, 244, 3, padding=1)
        # self.bn3 = torch.nn.BatchNorm3d(144)

        self.up_block1 = UNet_up_block(244, 244, 148)
        self.up_block2 = UNet_up_block(148, 148, 72)
        self.up_block3 = UNet_up_block(72, 72, 24)
        self.up_block4 = UNet_up_block(24, 24, 8)


        self.last_conv1 = torch.nn.Conv3d(12, 8, 3, padding=1)
        self.last_bn = my_norm(8)
        self.last_conv2 = torch.nn.Conv3d(8, 4, 3, padding=1)
        self.last_bn2= my_norm(4)
        self.relu = my_activation()

        self.last_conv3 = torch.nn.Conv3d(4, 4, 3, padding=1)

 
         
    def forward(self, x):
        # print('input unet',x.size())
        inputs = x.clone()

        # print('first down',x.size())

        self.x1 = self.down_block1(x)
        # print("Block 1 shape:",self.x1.size())
        self.x2 = self.down_block2(self.x1)
    
        # print("Block 2 shape:",self.x2.size())
        self.x3 = self.down_block3(self.x2)
        # print("Block 3 shape:",self.x3.size())
    
        self.x4 = self.down_block4(self.x3)
        # print("Block 4 shape:",self.x4.size())
    
        self.xmid=self.max_pool(self.x4)
        self.xmid = self.relu(self.mid_conv1(self.xmid))
        residual=self.xmid.clone()
        self.xmid = self.relu(self.mid_conv2(self.xmid))
        self.xmid = self.relu(self.mid_conv3(self.xmid))
        self.xmid = self.xmid + residual
        # print("Block Mid shape:",self.xmid.size())
        
        x = self.up_block1(self.x4, self.xmid)
        # print("BlockU 1 shape:",x.size())
        x = self.up_block2(self.x3, x)
        # print("BlockU 2 shape:",x.size())
        
        x = self.up_block3(self.x2, x)
        # print("BlockU 3 shape:",x.size())
 
        x = self.up_block4(self.x1, x)
        # print("BlockU 4 shape:",x.size())
        
        # print('before concat'x.size())

        x = torch.cat((x, inputs), dim=1) # concat with input  * show x not inps


        x = self.last_bn(self.relu(self.last_conv1(x)))
        x = self.last_bn2(self.relu(self.last_conv2(x)))
        x = self.last_conv3(x)
        
        # print('unet output',x.size())

        return(x)


loss_weights=3*torch.tensor([0.00233549, 0.14019698, 0.04583725, 0.14620756],requires_grad=False).cuda()
def calc(iflat,tflat,i,loss_weights=loss_weights):
    smooth = 1.

    intersection = (iflat * loss_weights[i]*tflat).sum() 
    
    return 1-(((2. * intersection) /
              (iflat.sum() + loss_weights[i]*tflat.sum() + smooth)))
def calc_val(iflat,tflat):
    smooth = 1.

    intersection = (iflat * tflat).sum() 
    
    return 1-(((2. * intersection) /
              (iflat.sum() + tflat.sum() + smooth)))



def dice_loss(inpu, target):  # [bs,4,96,96,64]
    a=0;b=0;c=0;d=0 
    intersection=0; union=0
    

    ip=inpu[:,0,:,:,:].contiguous().view(-1)
    tar=target[:,0,:,:,:].contiguous().view(-1)
    intersection+=(ip * tar).sum() 
    union+= ip.sum() + tar.sum()
     
    ip=inpu[:,1,:,:,:].contiguous().view(-1)
    tar=target[:,1,:,:,:].contiguous().view(-1)
    intersection+=(ip * tar).sum() 
    union+= ip.sum() + tar.sum()
     
    ip=inpu[:,2,:,:,:].contiguous().view(-1)
    tar=target[:,2,:,:,:].contiguous().view(-1)
    intersection+=(ip * tar).sum() 
    union+= ip.sum() + tar.sum()
     
    ip=inpu[:,3,:,:,:].contiguous().view(-1)
    tar=target[:,3,:,:,:].contiguous().view(-1)
    intersection+=(ip * tar).sum() 
    union+= ip.sum() + tar.sum()
 
    raw_scores=1-2*(intersection/union)
    
    return raw_scores


def dice_loss_classes(inpu, target):  # [bs,4,96,96,64]
    # whole=1-dice_loss(inpu,target)
    
    

    ip=inpu[:,0,:,:,:].contiguous().view(-1)
    tar=target[:,0,:,:,:].contiguous().view(-1)
    intersection=(ip * tar).sum() 
    union= ip.sum() + tar.sum()
    score1=1-2*(intersection/union)
    

    ip=inpu[:,1,:,:,:].contiguous().view(-1)
    tar=target[:,1,:,:,:].contiguous().view(-1)
    intersection=(ip * tar).sum() 
    union= ip.sum() + tar.sum()
    score2=1-2*(intersection/union)
     
    ip=inpu[:,2,:,:,:].contiguous().view(-1)
    tar=target[:,2,:,:,:].contiguous().view(-1)
    intersection=(ip * tar).sum() 
    union= ip.sum() + tar.sum()
    score3=1-2*(intersection/union)
     
    ip=inpu[:,3,:,:,:].contiguous().view(-1)
    tar=target[:,3,:,:,:].contiguous().view(-1)
    intersection=(ip * tar).sum() 
    union= ip.sum() + tar.sum()
    score4=1-2*(intersection/union)
 
    
    
    return score1,score2,score3,score4



def dice_loss_val(inpu, target):
    a=0;b=0;c=0;d=0 

    ip=inpu[:,0,:,:,:].contiguous().view(-1)
    tar=target[:,0,:,:,:].contiguous().view(-1)
    a+=calc_val(ip,tar)
     
    ip=inpu[:,1,:,:,:].contiguous().view(-1)
    tar=target[:,1,:,:,:].contiguous().view(-1)
    b+=calc_val(ip,tar)
     
    ip=inpu[:,2,:,:,:].contiguous().view(-1)
    tar=target[:,2,:,:,:].contiguous().view(-1)
    c+=calc_val(ip,tar)
     
    ip=inpu[:,3,:,:,:].contiguous().view(-1)
    tar=target[:,3,:,:,:].contiguous().view(-1)
    d+=calc_val(ip,tar)
 
    raw_scores=( a +  b +  c +  d)/4
    
    return raw_scores

# def vis(soft_mask,batch_seg_orig,vis_count,mode='train'):
#     arg=torch.argmax(soft_mask,0)
#     if mode=='train': name='train' 
#     else: name= 'val'
#     for k in range(70,95):
#         writer.add_image(name+' Output', soft_mask[0,:,:,k], vis_count+k)
#         writer.add_image(name+' Target',batch_seg_orig[0,:,:,k] , vis_count+k)



L=[];Lc=[];Ld=[];Lv=[];Lvd=[];Lvc=[];L_r=[];Ldc=[];Lvdc=[]
model = UNet().cuda()
def train(): 

    forward_times=counter_t=vis_count=0

    dataset = BraTS_FLAIR(csv_dir,hgg_dir,transform=None,train_size=train_size)
    dataset_val=BraTS_FLAIR_val(csv_dir,hgg_dir)    #val
    data_loader = DataLoader(dataset, batch_size, shuffle=True, num_workers=2)
    val_loader=DataLoader(dataset_val,batch_size_val,shuffle=True, num_workers=2) 
    loaders={'train':data_loader,'val': val_loader}

 
    top_models = [(0, 10000)] * 5 # (epoch,loss)
    worst_val = 10000 # init val save loop

    loss_fn = torch.nn.CrossEntropyLoss(weight=loss_weights)  
    soft_max=torch.nn.Softmax(dim=1)
    
    opt = torch.optim.Adam(model.parameters(), lr=init_lr)
    scheduler = StepLR(opt, step_size=opt_step_size, gamma=opt_gamma)  #**
    # multisteplr 0.5,

    opt.zero_grad()

    for epoch in range(epochs):
        for e in loaders:
            if e=='train': counter_t+=1; model.train() ; grad=True #
            else: model.eval() ; grad=False


            with torch.set_grad_enabled(grad):    
                for idx, batch_data in enumerate(loaders[e]):
                    batch_input = Variable(batch_data['img'].float()).cuda()
                    batch_gt_mask = Variable(batch_data['mask'].float()).cuda()
                    batch_seg_orig=Variable(batch_data['seg_orig']).cuda()

                    pred_mask = model(batch_input)
                    if e=='train':forward_times += 1

                    ce = loss_fn(pred_mask, batch_seg_orig.long())
                    soft_mask=soft_max(pred_mask)
                    dice=dice_loss(soft_mask, batch_gt_mask)

                    a,b,c,d=dice_loss_classes(soft_mask, batch_gt_mask)

                    loss = ce+(a+b+c+d)/4 # +dice

                    print('Dice Losses: ',a.item(),b.item(),c.item(),d.item())
                    if e=='train': 
                        Lc.append(ce.item()); cross_moving_avg=sum(Lc)/len(Lc)
                        Ldc.append(np.array([a.item(),b.item(),c.item(),d.item()]))
                        Ld.append(dice.item()); dice_moving_avg=sum(Ld)/len(Ld)
                        L.append(loss.item()); loss_moving_avg=sum(L)/len(L)
                        loss.backward()
                        print('Epoch: ', epoch+1, ' Batch: ',idx+1,' lr: ',scheduler.get_lr()[-1], ' Dice: ', dice_moving_avg, ' CE: ',cross_moving_avg,' Loss :' ,loss_moving_avg)
                        if forward_times == grad_accu_times:
                            opt.step()
                            opt.zero_grad()
                            forward_times = 0
                            print('\nUpdate weights ... \n')

                        writer.add_scalar('Train Loss', loss.item(), counter_t)
                        writer.add_scalar('Train CE', ce.item(), counter_t)
                        writer.add_scalar('Train Dice', dice.item() , counter_t)
                        writer.add_scalar('D1', a.item(), counter_t)
                        writer.add_scalar('D2', b.item(), counter_t)
                        writer.add_scalar('D3', c.item() , counter_t)

                        writer.add_scalar('D4', d.item() , counter_t)
                        writer.add_scalar('Lr', scheduler.get_lr()[-1] , counter_t)

                        
                        # vis(soft_mask,batch_seg_orig,vis_count,mode='train')
                        # vis_count+=25 # += no of images in vis loop
                        scheduler.step()


                    else:
                        Lv.append(loss.item()); Lvd.append(dice.item()); Lvc.append(ce.item())
                        lv_avg=sum(Lv)/len(Lv); lvd_avg=sum(Lvd)/len(Lvd); lvc_avg=sum(Lvc)/len(Lvc)
                        Lvdc.append(np.array([a.item(),b.item(),c.item(),d.item()]))
                        Ldc.append(np.array([a.item(),b.item(),c.item(),d.item()]))

                        writer.add_scalar('Val Loss', loss.item(), counter_t)
                        writer.add_scalar('Val CE', ce.item(), counter_t)
                        writer.add_scalar('Val Dice', dice.item(), counter_t)
                        writer.add_scalar('D1v', a.item(), counter_t)
                        writer.add_scalar('D2v', b.item(), counter_t)
                        writer.add_scalar('D3v', c.item() , counter_t)
                        writer.add_scalar('D4', d.item() , counter_t)
                        # vis(soft_mask,batch_seg_orig,vis_count,mode='val')
                        print(save_initial)

                        print('Validation total Loss::::::::::', round(loss.item(),3) )
                        del batch_input, batch_gt_mask, batch_seg_orig, pred_mask

                        print('current n worst val: ',round(loss.item(),2),worst_val)

                        if epoch>55 and epoch%15==0: # save every 15- for logs- confilicting with down

                            checkpoint = {'epoch': epoch + 1, 'moving loss':L,'dice':Ld,'val':Lv,
                            'valc':Lvc,'vald':Lvd,'cross el':Lc,'Ldvc':Lvdc,'Ldc':Ldc, 'state_dict': model.state_dict(), 'optimizer' : opt.state_dict() }
                            torch.save(checkpoint, save_initial +'-' + str(round(loss,2))+'|'+str(epoch+1))
                            print('Saved at 25 : ', save_initial +'-' +str(round(loss,2))+'|'+str(epoch+1))


                        if loss<worst_val : 
                            # print('saving --------------------------------------',epoch)
                            top_models=sorted(top_models, key=lambda x: x[1]) # sort maybe not needed 
                            
                            checkpoint = {'epoch': epoch + 1, 'moving loss':L,'dice':Ld,'val':Lv,
                            'valc':Lvc,'vald':Lvd,'cross el':Lc, 'Ldvc':Lvdc,'Ldc':Ldc, 'state_dict': model.state_dict(), 'optimizer' : opt.state_dict() }
                            torch.save(checkpoint, save_initial +'-' +str(round(loss,2))+'|'+str(epoch+1))

                            to_be_deleted=save_initial +'-' +str(round(top_models[-1][1],2))+'|'+str(top_models[-1][0]) # ...loss|epoch
                            # print(to_be_deleted)

                            top_models.append((epoch+1,loss.item()))
                            
                            top_models=sorted(top_models, key=lambda x: x[1]) #sort after addition of new val
                            top_models.pop(-1)

                            print('top_models',top_models)

                            worst_val=top_models[-1][1]
                            if str(to_be_deleted)!=save_initial +'-' +'10000.0|0':     # first 5 epoch will be saved and no deletion this point
                                os.remove(to_be_deleted)
                                # print('sucess deleted------------------')
                                
                        break

train()
        

    