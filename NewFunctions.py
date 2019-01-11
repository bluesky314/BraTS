def padder(img): 
    pad=torch.nn.ConstantPad2d(1,0)
    padded=pad(img.transpose(3,4).transpose(2,3))[:,:,:,:-1,:-1]
    return(padded.transpose(2,3).transpose(3,4))
    
[40:212,30:202,15:143] 

from skimage.transform import resize
resiz=resize(np.array(d['seg_orig']),[192/2, 192/2, 128],order=0,mode='constant', preserve_range=True)

def standardizer(img,st=standardizer_type,clamp=True):

    img=torch.tensor(img).float()

    if 'he' in st: img=exposure.equalize_hist(np.array(img)); img=torch.tensor(img)
    if 'mode' in st: 
        z=img.flatten()[img.flatten()>0]
        mean=z.float().flatten().mode()[0]  
        std=mode_std(z)
    if 'mean' in st : mean=img.mean(); std=img.std()
    if 'median' in st : mean=img.median(); std=img.std()
    
    img=img-mean
    img=img/std
    
    quartiles = percentile(img, [99.5])  # exact region
    if clamp: img=torch.clamp(img,img.min(),quartiles[0])

    return(img)
    
    def classify_label(msk): 
        v=np.zeros((msk.size()[-1],3))
        for i in range(msk.size()[-1]):
            if (msk[:,:,i]==1).any(): v[i,0]=1
            if (msk[:,:,i]==2).any(): v[i,1]=1
            if (msk[:,:,i]==3).any(): v[i,2]=1
        return(torch.tensor(v))
        
