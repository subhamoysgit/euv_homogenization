import os
import pickle
import numpy as np

def imageIndexer(patchPath, trainPath, valPath):

	# Index batches
	patch_num = []
	for root,dirs,files in os.walk(patchPath):
		for file in files:
				if file.startswith('eit'):
					patches_e = pickle.load(open(root+file, "rb" ))
					if np.sum(patches_e[:,:,1]<1)>=0:
						patch_num.append(file[13:18])

	# Find number of training batches
	file_list = []
	for root,dirs,files in os.walk(trainPath):
		for file in files:
			if file[:3]=='eit' and file[13:18] in patch_num:
				file_list.append(root+file)
	nTrain = len(file_list)

	# Find number of tvalidation batches
	file_list = []
	for root,dirs,files in os.walk(valPath):
		for file in files:
			if file[:3]=='eit' and file[13:18] in patch_num:
				file_list.append(root+file)
	nVal = len(file_list)							

	return patch_num, nTrain, nVal


def imageLoader(file_path, batch_size, patch_num, rng=None, vflip=False, hflip=False):

	# Initialize random generator if not passed
	if rng is None:
		rng = np.random.default_rng()

	# Add the number of augmentations used
	nAugmentations = 0
	if hflip:
		nAugmentations += 1
	if vflip:
		nAugmentations += 1

	file_list = []
	for root,dirs,files in os.walk(file_path):
		for file in files:
			if file[:3]=='eit' and file[13:18] in patch_num:
				file_list.append(root+file)
	file_list = file_list[:batch_size*(len(file_list)//batch_size)]

	rng.shuffle(file_list)

	k = 0
	num = len(file_list)//batch_size
	X = np.zeros(((2**nAugmentations)*batch_size,64,64,2),dtype=float)
	Y = np.zeros(((2**nAugmentations)*batch_size,256,256,1),dtype=float)

	while True:
		k = k % num
		for i in range(batch_size):
			eit = pickle.load(open(file_list[k*batch_size+i], "rb" ))
			aia = pickle.load(open(file_list[k*batch_size+i][:16]+'aia'+file_list[k*batch_size+i][19:], "rb" ))
			prof = eit[:,:,1]
			X[i,:,:,1] = prof[:,:]
			Y[i,:,:,0] = aia[:,:]

		## Augmentation within the batch

		if vflip:
			X[batch_size:2*batch_size,:,:,:] = np.flip(X[:batch_size,:,:,:].copy(),1)
			Y[batch_size:2*batch_size,:,:,0] = np.flip(Y[:batch_size,:,:,0].copy(),1)

		if hflip:
			X[(1+vflip)*batch_size:(1+vflip)*2*batch_size,:,:,:] = np.flip(X[:batch_size*(1+vflip),:,:,:].copy(),2)
			Y[(1+vflip)*batch_size:(1+vflip)*2*batch_size,:,:,0] = np.flip(Y[:batch_size*(1+vflip),:,:,0].copy(),2)

		k = k+1
		if k == num:
			rng.shuffle(file_list)

		yield(X,Y)                    