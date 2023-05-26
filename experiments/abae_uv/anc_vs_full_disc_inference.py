# System paths and GPU vs. CPU
import os
import sys

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Append source folder to system path.  It uses the folder where the experiment runs.
# Since the template file is in 'src/templates' you only need to remove the last folder i.e. you only need '-1' in
# os.path.abspath(__file__).split('/')[:-1].  If you make your folder structure deeper, be sure to increase this value.
 
_MODEL_DIR = os.path.abspath(__file__).split('/')[:-2]
_SRC_DIR = os.path.join('/',*_MODEL_DIR[:-1])
_SRC_DIR = os.path.join(_SRC_DIR,'src')

sys.path.append(_SRC_DIR)


# Load modules
import numpy as np
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
import pickle as p
import matplotlib.pyplot as plt
import sunpy.map
import os
import pickle
import matplotlib.pyplot as plt
import cv2

##------------------------------------------------------------------------------------
## Random seed initialization
SEED_VALUE = 42
rng = np.random.default_rng(SEED_VALUE)


##------------------------------------------------------------------------------------
## Load CNN model and CNN coefficients
from models.model_HighResnet_ABAE_normal_init import make_CNN

# CNN options
ENSEMBLE_SIZE = 4  # no. CNNs in ensemble
REGULARIZATION = 'anc'  # type of regularisation to use - anc (anchoring) reg (regularised) free (unconstrained)
BATCH_SIZE = 10  # Batch Size
EPOCH0 = 1  # First epoch

DATA_NOISE = 0.1 # noise variance as mean of aia patch hist
W_VAR_I = 1 # variance of the anchor weights
W_LAMBDA_I = 0.001 # Strength of the regularization term for anchor weights
B_VAR_I = W_VAR_I # variance of the anchor biases 
B_LAMBDA_I = W_LAMBDA_I # Strength of the regularization term for anchor biases


##------------------------------------------------------------------------------------
## Patch location, data loader, and augmentation
from data_loaders.eit_aia_loader import imageIndexer, imageLoader

TRAIN_PATH = '/d1/patches/trn/'  # Training data path
VAL_PATH = '/d1/patches/trn/'  # Validation data path


# Augmentation
VFLIP = True  # Vertical flip
HFLIP = True  # Horizontal flip

##------------------------------------------------------------------------------------
## Optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001,beta_1=0.5)

#ANCHOR_PARAMS  = '_GLOROT_UNIF_last_LAMBDA_SCALED_001_ST_1_'
MODEL_FOLDER = '/d0/models/'
MODEL_FILE = 'eit_aia_sr_abae_medium_GLOROT_UNIF_last_LAMBDA_SCALED_001_ST_1_'

#model_path = '/d0/models/'
#model_names = ['eit_aia_sr_big_v3.h5','eit_aia_sr_big_v5.h5','eit_aia_sr_big_v6.h5','eit_aia_sr_big_v9.h5','eit_aia_sr_big_v10.h5','eit_aia_sr_big_v8.h5','eit_aia_sr_big_v11.h5','eit_aia_sr_big_v13.h5','eit_aia_sr_big_v7.h5','eit_aia_sr_big_v12.h5']
CNNs = []
ENSEMBLE_SIZE = 4
best_epochs = np.zeros(ENSEMBLE_SIZE)
for m in range(ENSEMBLE_SIZE):
    p = pickle.load(open(MODEL_FOLDER + MODEL_FILE + str(m+1).zfill(2) +'_'+'val_mse.p','rb'))
    best_epochs[m] = 1 + np.argmin(p)
for m in range(ENSEMBLE_SIZE):
    CNNs.append(make_CNN(reg=REGULARIZATION, features=32, rng=rng, W_var_i=W_VAR_I, W_lambda_i=W_LAMBDA_I, b_var_i=B_VAR_I, b_lambda_i=B_LAMBDA_I))
    CNNs[m].load_weights(MODEL_FOLDER + MODEL_FILE + str(m+1).zfill(2) +'_'+str(int(best_epochs[m])).zfill(2)+'.h5')




fd_path = '/d1/fd/val/eit_171/'
patch_path = '/d0/patches/val/'
files_e = []
files_d = []
file_list = []


for root,dirs,files in os.walk(fd_path):
    for file in files:
        files_e.append(file)
        files_d.append(file[3:11])



f = "eit_20130831"
idx = files_d.index(f[4:12])
f = files_e[idx]

eit_fd = np.zeros((64*14,64*14))
aia_fd = np.zeros((256*14,256*14))
fig, axs = plt.subplots(2,ENSEMBLE_SIZE, figsize=(5*ENSEMBLE_SIZE,5*2))
for n in range(196):
  x0 = 64*(n//14)
  y0 = 64*(n%14)
  y1 = y0+64
  x1 = x0+64
  patches_e = pickle.load(open(patch_path+'eit_'+f[3:11]+'_0_'+str(n).zfill(3)+'.p', "rb" ))
  patches_na = pickle.load(open(patch_path+'aia_'+f[3:11]+'_n_0_'+str(n).zfill(3)+'.p', "rb" ))
  e = sunpy.map.Map(fd_path+f)
  eit_fd[y0:y1,x0:x1] = patches_e[:,:,0]#*e.meta['exptime']*1000
  aia_fd[4*y0:4*y1,4*x0:4*x1] = patches_na[:,:]

xx0 = 64*10#(100//14)
yy0 = 64*7#(100%14)
yy1 = yy0+64
xx1 = xx0+64
low_res = 1.3*eit_fd - 0.92
low_res[low_res<0]=0
patch_lr = low_res[yy0:yy1,xx0:xx1].copy() 
baseline = 1.3*cv2.resize(eit_fd,(256*14,256*14),interpolation = cv2.INTER_CUBIC) - 0.92
baseline[baseline<0]=0
patch_bl = baseline[4*yy0:4*yy1,4*xx0:4*xx1].copy()
target = aia_fd
target[target<0]=0
patch_tr = target[4*yy0:4*yy1,4*xx0:4*xx1].copy()



ax = axs.ravel()

low = 0#np.min(target)
high = 1#np.max(target)


pred_Ensemble = np.zeros((256*14,256*14,ENSEMBLE_SIZE))
for k in range(ENSEMBLE_SIZE):
    model = CNNs[k]
    pred_fd = np.zeros((256*14,256*14))
    for n in range(196):
        x0 = 64*(n//14)
        y0 = 64*(n%14)
        y1 = y0+64
        x1 = x0+64
        patches_e = pickle.load(open(patch_path+'eit_'+f[3:11]+'_0_'+str(n).zfill(3)+'.p', "rb" ))
        data = np.zeros((1,64,64,2))
        e = sunpy.map.Map(fd_path+f)
        data[0,:,:,0] = patches_e[:,:,0]#*e.meta['exptime']*1000
        eit = patches_e[:,:,1]
        data[0,:,:,1] = eit
        data_pred = model.predict(data)
        pred_fd[4*y0:4*y1,4*x0:4*x1] = data_pred[0,:,:,0]
    predicted = pred_fd
    pred_Ensemble[:,:,k] = predicted
    #patch_pred = predicted[4*yy0:4*yy1,4*xx0:4*xx1].copy()
    #cv2.rectangle(predicted,(4*xx0,4*yy0), (4*xx1,4*yy1), (255,255,255), 16)
    ax[4+k].imshow(predicted[:128*14,:128*14]**0.5,cmap='magma',vmin = low,vmax = high)
    #ax[4+k].set_title(loss_names[k],fontsize=8)
    ax[4+k].set_xticks([])
    ax[4+k].set_yticks([])
    ax[4+k].set_xlabel('ANCHOR '+str(k+1)+ " OUTPUT",fontsize=20)
    # ax[6+k+len(model_names)].imshow(patch_pred**0.5,cmap='magma',vmin = low,vmax = high)
    # ax[6+k+len(model_names)].set_xticks([])
    # ax[6+k+len(model_names)].set_yticks([])


pred_Mean = np.mean(pred_Ensemble,axis=2)
pred_SDEV = np.std(pred_Ensemble,axis=2)
pred_rel = pred_SDEV/pred_Mean

ax[0].imshow(low_res[:32*14,:32*14]**0.5,cmap='magma',vmin = low,vmax = high)
ax[0].set_xticks([])
ax[0].set_yticks([])
ax[0].set_title('SoHO/EIT 171 $\AA$ [Input]',fontsize=20)

ax[1].imshow(target[:128*14,:128*14]**0.5,cmap='magma',vmin = low,vmax = high)
ax[1].set_xticks([])
ax[1].set_yticks([])
ax[1].set_title('SDO/AIA 171 $\AA$ [Target]',fontsize=20)

ax[2].imshow(pred_Mean[:128*14,:128*14]**0.5,cmap='magma',vmin = low,vmax = high)
#ax[2].contour(pred_rel[:128*14,:128*14], levels=[0.1,0.2,0.3,0.4], colors='white', alpha=0.5)
ax[2].set_xticks([])
ax[2].set_yticks([])
ax[2].set_title('ENSEMBLE MEAN',fontsize=20)

ax[3].imshow(pred_rel[:128*14,:128*14],cmap='viridis',vmin = low,vmax=0.4)
ax[3].set_xticks([])
ax[3].set_yticks([])
ax[3].set_title('ENSEMBLE RELATIVE SD',fontsize=20)


plt.subplots_adjust(wspace=0.0, hspace=0.0)#0.4)
#plt.show()
plt.savefig('/home/subhamoy/mycodes/euv_homogenization/ANC_ENSEMBLE_EUV_INFERENCE.pdf',dpi=300)