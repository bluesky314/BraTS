
import numpy as np
import pandas as pd
import gzip
import nibabel as nib
import matplotlib.pyplot as plt
import io
import torch
import torch.utils#.d

def get_dim(msk):
    x = msk.unsqueeze(0) # dimensions of non-zero Mask as cube

    non_zero = (x!=0).nonzero()  # Contains non-zero indices for all 4 dims
    h_min = non_zero[:, 1].min()
    h_max = non_zero[:, 1].max()
    w_min = non_zero[:, 2].min()
    w_max = non_zero[:, 2].max()
    d_min = non_zero[:, 3].min()
    d_max = non_zero[:, 3].max()
    return(h_min,h_max,w_min,w_max,d_min,d_max)
def tumor_patch(img,msk,seg_orig):
#     print(img.size(),msk.size())
    width=64
    height=64
    depth=64
    x = msk.unsqueeze(0) # dimensions of non-zero Mask as cube


    non_zero = (x!=0).nonzero()  # Contains non-zero indices for all 4 dims
    h_min = non_zero[:, 1].min()
    h_max = non_zero[:, 1].max()
    w_min = non_zero[:, 2].min()
    w_max = non_zero[:, 2].max()
    d_min = non_zero[:, 3].min()
    d_max = non_zero[:, 3].max()

    flag=False
    counter=0
    while flag==False:
        counter+=1
        try:
            wt1=np.random.randint(w_min,w_max-width)  # atlease n pixels of tumor present
        except:
            try:wt1=np.random.randint(w_min,w_max-(width//2))
            except: wt1=np.random.randint(w_min,w_max-(width//4))    
                
        wt2=wt1+width
        
        try:
            ht1=np.random.randint(h_min,h_max-height)  # atlease n pixels of tumor present
        except:
            try:
                ht1=np.random.randint(h_min,h_max-(height//2))
            except: 
                ht1=np.random.randint(h_min,h_max-(height//6))    
                
                
        try:
            ht1=np.random.randint(h_min,h_max-height)  # atlease n pixels of tumor present
        except:
            try:
                ht1=np.random.randint(h_min,h_max-(height//2))
            except: 
                ht1=np.random.randint(h_min,h_max-(height//6))    
                
        
        ht2=ht1+height
        
        try:
            dt1=np.random.randint(d_min,d_max-depth)  # atlease n pixels of tumor present
        except:
            try:
                dt1=np.random.randint(d_min,d_max-(depth//2))
            except: 
                dt1=np.random.randint(d_min,d_max-(depth//6))    
                

#         dt1=np.random.randint(d_min,d_max-depth)
        dt2=dt1+depth
#         print('w,h,d' ,wt1,wt2,ht1,ht2,dt1,dt2)

#         print(img.size())
#         print(np.linspace(ht1,ht2,ht2-ht1+1))
        not_tumor_img=img[:,ht1:ht2,wt1:wt2,dt1:dt2] #not_tumor is tumor
#         print(np.linspace(ht1,ht2,ht2-ht1+1))
#         print(not_tumor_img.size())
        black=torch.sum(not_tumor_img==0).float()

        total=not_tumor_img.size()[0]*not_tumor_img.size()[1]*not_tumor_img.size()[2]*not_tumor_img.size()[3]+0.2

        nonzero=total-black+0.2

#         print(nonzero,total)
        frac=nonzero/total
    
#         print('frac',frac)
        
        if counter>50:
            only_masks=msk[ht1:ht2,wt1:wt2,dt1:dt2]
#             print('counter return')
            return(not_tumor_img,only_masks)

        if frac < 0.68:flag=False; continue

        only_masks=msk[ht1:ht2,wt1:wt2,dt1:dt2]
        seg_orig_ret=seg_orig[:,ht1:ht2,wt1:wt2,dt1:dt2]

        flag=True
        return(not_tumor_img,only_masks,seg_orig_ret)
def non_tumor_patch(img,msk,seg_orig):
    width=64
    height=64
    depth=64
    
    x = msk.unsqueeze(0) # dimensions of non-zero Mask as cube
    non_zero = (x!=0).nonzero()  # Contains non-zero indices for all 4 dims
    h_min = non_zero[:, 1].min()
    h_max = non_zero[:, 1].max()
    w_min = non_zero[:, 2].min()
    w_max = non_zero[:, 2].max()
    d_min = non_zero[:, 3].min()
    d_max = non_zero[:, 3].max()


    counter=0
    flag=False
    while flag ==False:
        counter+=1
        w1=np.random.randint(0,192-width-5) +0.0
        w2=w1+width

        h1=np.random.randint(0,192-height-5)+0.0
        h2=h1+height

        d1=np.random.randint(0,128-depth-5)+0.0
        d2=d1+depth

        
        np.any(np.in1d(np.linspace(w1,w2,w2-w1+1), np.linspace(w_min,w_max,w_max-w_min+1)))
        
        if np.any(np.in1d(np.linspace(w1,w2,w2-w1+1), np.linspace(w_min,w_max,w_max-w_min+1)))==False:width_flag=True
        else: width_flag=False
        
        if np.any(np.in1d(np.linspace(h1,h2,h2-h1+1), np.linspace(h_min,h_max,h_max-h_min+1)))==False:height_flag=True
        else: height_flag=False

        flag=height_flag or width_flag
        if flag==False: continue
        

        w1,h1,d1,w2,h2,d2=int(w1),int(h1),int(d1),int(w2),int(h2),int(d2)
        
        not_tumor_img=img[:,h1:h2,w1:w2,d1:d2]

        if not_tumor_img.size()[1]!=height:
            continue
        if not_tumor_img.size()[2]!=width:
            continue
        if not_tumor_img.size()[3]!=depth:
            continue
        
        black=torch.sum(not_tumor_img==0).float()

        total=not_tumor_img.size()[0]*not_tumor_img.size()[1]*not_tumor_img.size()[2]*not_tumor_img.size()[3]+0.2

        nonzero=total-black+0.2

        frac=nonzero/total
#         print(nonzero,total)
#         print('frac',frac)
        
        if counter>50:
            only_masks=msk[ht1:ht2,wt1:wt2,dt1:dt2]
            return(not_tumor_img,only_masks)

        
        if frac < 0.70:flag=False; continue
            
        only_masks=msk[h1:h2,w1:w2,d1:d2]
        seg_orig_ret=seg_orig[:,h1:h2,w1:w2,d1:d2]

        flag=True
        
        return(not_tumor_img,only_masks,seg_orig_ret)
