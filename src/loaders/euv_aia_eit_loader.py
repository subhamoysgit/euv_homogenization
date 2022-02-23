import os
import numpy as np


def imageLoader(file_path, batch_size,patch_num,fd_path, rng):
	files_e = []
	files_d = []
	for root,dirs,files in os.walk(fd_path):
		for file in files:
				files_e.append(file)
				files_d.append(file[3:11])

	file_list = []
	for root,dirs,files in os.walk(file_path):
		for file in files:
			if file[:3]=='eit' and file[13:18] in patch_num:
				file_list.append(root+file)
	file_list = file_list[:batch_size*(len(file_list)//batch_size)]

	rng.shuffle(file_list)
	k = 0
	num = len(file_list)//batch_size
	X = np.zeros((4*batch_size,64,64,2),dtype=float)
	Y = np.zeros((4*batch_size,256,256,1),dtype=float)
	while True:
		k = k % num
		for i in range(batch_size):
			idx = files_d.index(file_list[k*batch_size+i][20:28]) 
			e = sunpy.map.Map(fd_path+files_e[idx])
			eit = pickle.load(open(file_list[k*batch_size+i], "rb" ))
			aia = pickle.load(open(file_list[k*batch_size+i][:16]+'aia'+file_list[k*batch_size+i][19:], "rb" ))
			X[i,:,:,0] = eit[:,:,0]*1000*e.meta['exptime']
			prof = eit[:,:,1]
			X[i,:,:,1] = prof[:,:]
			Y[i,:,:,0] = aia[:,:]
		X[batch_size:2*batch_size,:,:,:] = np.flip(X[:batch_size,:,:,:].copy(),1)
		X[2*batch_size:3*batch_size,:,:,:] = np.flip(X[:batch_size,:,:,:].copy(),2)
		X[3*batch_size:4*batch_size,:,:,:] = np.flip(np.flip(X[:batch_size,:,:,:].copy(),1),2)
		Y[batch_size:2*batch_size,:,:,0] = np.flip(Y[:batch_size,:,:,0].copy(),1)
		Y[2*batch_size:3*batch_size,:,:,0] = np.flip(Y[:batch_size,:,:,0].copy(),2)
		Y[3*batch_size:4*batch_size,:,:,0] = np.flip(np.flip(Y[:batch_size,:,:,0].copy(),1),2)
		k = k+1
		if k == num:
			rng.shuffle(file_list)

		yield(X,Y)
