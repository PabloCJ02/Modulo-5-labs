#Laboratorio 5_1
import cv2
import urllib.request
import os
from matplotlib import pyplot as plt
#%matplotlib inline
import numpy as np
from pylab import rcParams

minions_url = "https://www.cleverfiles.com/howto/wp-content/uploads/2018/03/minion.jpg"
minions_filename = "minions.jpg"
urllib.request.urlretrieve(minions_url, minions_filename)

minions = cv2.imread(minions_filename)
#plt.axis("off")
#img_corrected = cv2.cvtColor(minions, cv2.COLOR_BGR2RGB)
#plt.imshow(img_corrected)
#plt.show()


gray_minions = cv2.cvtColor(minions, cv2.COLOR_BGR2GRAY)
#plt.imshow(gray_minions, cmap = "gray")
#plt.axis("off") #remove axes ticks
#plt.title('Grayscale Minions')
#plt.show()


#rcParams['figure.figsize'] = 8,4

#plt.hist(gray_minions.ravel(),256,[0,256])
#plt.title('Histogram of Grayscale minions.jpg')
#plt.show()


rcParams['figure.figsize'] = 8, 4

color = ('b','g','r')
for i,col in enumerate(color):
    histr = cv2.calcHist([minions],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
plt.show()