import numpy as np
import math
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from time import time
pi = math.pi

def show(grayImg):
	imgplot = plt.imshow(grayImg)
	plt.show()

def refX(cX, r, fi):
	phi = 2*pi/16
	return np.int16(np.round(cX+r*math.cos(phi*fi)))

def refY(cY, r, fi):
	phi = 2*pi/16
	return np.int16(np.round(cY+r*math.sin(phi*fi)))

class kernelOut():
	def __init__(self, R, r, d):
		self.h = np.int16(2*R+1)
		self.w = np.int16(2*R+1)
		cX = R
		cY = R
		self.MapX = np.zeros(d*16, np.int16)
		self.MapY = np.zeros(d*16, np.int16)
		i = 0
		for rad in range(R-d, R):
			for fi in range(16):
				self.MapX[i] = refX(cX, rad, fi)
				self.MapY[i] = refY(cY, rad, fi)
				i += 1
	
	def get(self):
		return self.h, self.w, self.MapX, self.MapY

class kernelIn():
	def __init__(self, R, r, d):
		self.h = np.int16(2*R+1)
		self.w = np.int16(2*R+1)
		cX = np.int16(R+1)
		cY = np.int16(R+1)
		self.MapX = np.zeros(d*16, np.int16)
		self.MapY = np.zeros(d*16, np.int16)
		i = np.int16(0)
		for rad in range(r-d, r):
			for fi in range(16):
				self.MapX[i] = refX(cX, rad, fi)
				self.MapY[i] = refY(cY, rad, fi)
				i += np.int16(1)
	
	def get(self):
		return self.h, self.w, self.MapX, self.MapY

	def convolution(grayImg, hKern, wKern, MapXIn, MapYIn, MapXOut, MapYOut):
		grayImg = grayImg.astype(np.int16)
		resH = grayImg.shape[0]-hKern
		resW = grayImg.shape[1]-wKern
		resImg = np.zeros((resH, resW), np.int16)
		for j in np.arange(resH, dtype=np.int16):
			for i in np.arange(resW, dtype=np.int16):
				resImg[j][i] = abs(np.sum(grayImg[MapYIn+j, MapXIn+i])-np.sum(grayImg[MapYOut+j, MapXOut+i]))
		maximum = np.max(resImg)
		cv2.normalize(resImg, resImg, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
		return resImg, maximum

		
	def API(grayImg, R):
		r = R-2
		kIn = kernelIn(R, r, 1)
		hKern, wKern, MapXIn, MapYIn = kIn.get()
		kOut = kernelOut(R, r, 1)
		hKern, wKern, MapXOut, MapYOut = kOut.get()
		Im = convolution(grayImg, hKern, wKern, MapXIn, MapYIn, MapXOut, MapYOut)
		return Im
	
	folder = "C:/Users/Maria/Desktop/Work/jhdf.jpg"
	grayImg = cv2.imread(folder, 0)
	R = 50
	Im, maximum = API(grayImg, R)
	print(maximum)
	print(np.mean(Im))
	cv2.imwrite("C:/Users/Maria/Desktop/Work/convoluted.png", Im)
	plt.imshow(Im)
	plt.imshow(grayImg)