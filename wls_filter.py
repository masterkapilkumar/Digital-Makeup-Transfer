import cv2
import sys
import numpy as np

import matplotlib.pyplot as plt
from scipy.sparse import spdiags
from scipy.sparse.linalg import spsolve

def wlsfilter_layer(image_orig, cmat, beta=0.2 ,lambda_=0.2):
	

	gmat = cv2.GaussianBlur(cmat,(5,5),0)

	image = image_orig.astype(np.float)/255.0
	image1 = image.flatten(1)
	s = image.shape
	k = np.prod(s)

	dy = np.diff(image, 1, 0)
	dx = np.diff(image, 1, 1)
	
	dy = -beta*lambda_ / ((np.absolute(dy) ** 1.2 + 0.0001))
	dx = -beta*lambda_ / ((np.absolute(dx) ** 1.2 + 0.0001))
	
	for y in xrange(s[0]-1):
		for x in xrange(s[1]-1):
			dy[y][x] = (gmat[y][x]==1)*dy[y][x]
			dx[y][x] = (gmat[y][x]==1)*dx[y][x]

	dy = np.vstack((dy, np.zeros(s[1], )))
	dx = np.hstack((dx, np.zeros(s[0], )[:, np.newaxis]))

	dy = dy.flatten(1)
	dx = dx.flatten(1)
	
	d = 1 - (dx + np.roll(dx, s[0]) + dy + np.roll(dy, 1))

	a = spdiags(np.vstack((dx, dy)), [-s[0], -1], k, k)
	a = a + a.T + spdiags(d, 0, k, k)
	
	temp = spsolve(a, image1).reshape(s[::-1])
	
	base = np.rollaxis(temp,1)
	detail = image - base
	
	return (base*255.0), (detail*255.0)


























