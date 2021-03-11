import urllib.request
import cv2
import os
import math
import matplotlib.pyplot as plt
#%matplotlib inline
from sklearn.cluster import KMeans
import numpy as np

##############################    Pez      ############################################


#fish_image_url = "http://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/CV0101/Dataset/fish.png"
#urllib.request.urlretrieve(fish_image_url, "fish.png") # downloads file as "fish.png"
#im2 = cv2.imread("fish.png")
#fish_im_corrected = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)
#plt.axis('off')
#plt.imshow(fish_im_corrected)
#plt.show()
#print("Original size of fish image is: {} Kilo Bytes".format(str(math.ceil((os.stat('fish.png').st_size)/1000))))



#num_rows_fish = im2.shape[0]
#num_cols_fish = im2.shape[1]
#transform_fish_image_for_KMeans = im2.reshape(num_rows_fish * num_cols_fish, 3)

#kmeans_fish = KMeans(n_clusters=8)
#kmeans_fish.fit(transform_fish_image_for_KMeans)
#cluster_centroids_fish = np.asarray(kmeans_fish.cluster_centers_,dtype=np.uint8)

#labels_fish = np.asarray(kmeans_fish.labels_,dtype=np.uint8 )
#labels_fish = labels_fish.reshape(num_rows_fish,num_cols_fish)

#compressed_image_fish = np.ones((num_rows_fish, num_cols_fish, 3), dtype=np.uint8)
#for r in range(num_rows_fish):
#    for c in range(num_cols_fish):
#        compressed_image_fish[r, c, :] = cluster_centroids_fish[labels_fish[r, c], :]
#cv2.imwrite("compressed_fish.png", compressed_image_fish)
#compressed_fish_im = cv2.imread('compressed_fish.png')
#compressed_fish_im_corrected = cv2.cvtColor(compressed_fish_im, cv2.COLOR_BGR2RGB)
#plt.axis('off')
#plt.imshow(compressed_fish_im_corrected)
#plt.show()
#print("Compressed size of fish's image is: {} Kilo Bytes".format(str(math.ceil((os.stat('compressed_fish.png').st_size)/1000))))



################                  Mariposa            #################################

butterfly_image_url = "http://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/CV0101/Dataset/butterfly.png"
urllib.request.urlretrieve(butterfly_image_url, "butterfly.png") # downloads file as "butterfly.png"
im3 = cv2.imread("butterfly.png")
butterfly_im_corrected = cv2.cvtColor(im3, cv2.COLOR_BGR2RGB)
#plt.axis('off')
#plt.imshow(butterfly_im_corrected)
#plt.show()
#print("Original size of butterfly image is: {} Kilo Bytes".format(str(math.ceil((os.stat('butterfly.png').st_size)/1000))))



num_rows_butterfly = im3.shape[0]
num_cols_butterfly = im3.shape[1]
transform_butterfly_image_for_KMeans = im3.reshape(num_rows_butterfly * num_cols_butterfly, 3)




kmeans_butterfly = KMeans(n_clusters=8)
kmeans_butterfly.fit(transform_butterfly_image_for_KMeans)
cluster_centroids_butterfly = np.asarray(kmeans_butterfly.cluster_centers_,dtype=np.uint8)


labels_butterfly = np.asarray(kmeans_butterfly.labels_,dtype=np.uint8 )
labels_butterfly = labels_butterfly.reshape(num_rows_butterfly,num_cols_butterfly)



compressed_image_butterfly = np.ones((num_rows_butterfly, num_cols_butterfly, 3), dtype=np.uint8)
for r in range(num_rows_butterfly):
    for c in range(num_cols_butterfly):
        compressed_image_butterfly[r, c, :] = cluster_centroids_butterfly[labels_butterfly[r, c], :]
cv2.imwrite("compressed_image_butterfly.png", compressed_image_butterfly)
compressed_butterfly_im = cv2.imread('compressed_image_butterfly.png')
compressed_butterfly_im_corrected = cv2.cvtColor(compressed_butterfly_im, cv2.COLOR_BGR2RGB)
plt.axis('off')
plt.imshow(compressed_butterfly_im_corrected)
plt.show()
print("Compressed size of butterfly image is: {} Kilo Bytes".format(str(math.ceil((os.stat('compressed_image_butterfly.png').st_size)/1000))))



