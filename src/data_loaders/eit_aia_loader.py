import os
import pickle
import numpy as np

def imageIndexer(trainPath, valPath, trainDateRange = None, valDateRange = None):

	# Find number of training batches
	file_list = []
	for root,dirs,files in os.walk(trainPath):
		for file in files:
			if file[:3]=='eit':
				if trainDateRange:
					if int(file[4:12])>=trainDateRange[0] and int(file[4:12])<=trainDateRange[1]:
						file_list.append(root+file)
				else:
					file_list.append(root+file)
	nTrain = len(file_list)

	# Find number of tvalidation batches
	file_list = []
	for root,dirs,files in os.walk(valPath):
		for file in files:
			if file[:3]=='eit':
				if valDateRange:
					if int(file[4:12])>=valDateRange[0] and int(file[4:12])<=valDateRange[1]:
						file_list.append(root+file)
				else:
					file_list.append(root+file)
	nVal = len(file_list)							

	return nTrain, nVal


def imageLoader(file_path, batch_size, DateRange = False, rng=None, vflip=False, hflip=False):
    # DateRange if not False should be an integer list [DateInitial, DateFinal] in YYYYMMDD format
	# Initialize random generator if not passed
	if rng is None:
		rng = np.random.default_rng()

	file_list = []
	for root,dirs,files in os.walk(file_path):
		for file in files:
			if file[:3]=='eit':
				if DateRange:
					if int(file[4:12])>=DateRange[0] and int(file[4:12])<=DateRange[1]:
						file_list.append(root+file)
				else:
					file_list.append(root+file)
	file_list = file_list[:batch_size*(len(file_list)//batch_size)]

	rng.shuffle(file_list)

	k = 0
	num = len(file_list)//batch_size
	X = np.zeros(((2**(hflip+vflip))*batch_size,64,64,2),dtype=float)
	Y = np.zeros(((2**(hflip+vflip))*batch_size,256,256,1),dtype=float)

	while True:
		k = k % num
		for i in range(batch_size):
			eit = pickle.load(open(file_list[k*batch_size+i], "rb" ))
			aia = pickle.load(open(file_list[k*batch_size+i][:16]+'aia'+file_list[k*batch_size+i][19:], "rb" ))
			prof = eit[:,:,1]
			X[i,:,:,0] = eit[:,:,0]
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