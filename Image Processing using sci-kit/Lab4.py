##TASK 1
import skimage
import os
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from skimage import io
from skimage.transform import rescale, resize
import numpy as np
from skimage.exposure import histogram
from skimage.util import img_as_ubyte
from skimage import data
from skimage.feature import match_template
from skimage import data
from skimage.feature import match_template

filename_path = 'C:\Users\Umut\Documents\Python Scripts/Lab4'
filename = os.path.join(filename_path,'coins.jpg')
filename2 = os.path.join(filename_path,'astronaut.jpg')

image = io.imread(filename)
image2 = io.imread(filename2)
print("\ncoins picture information before conversion:")
print(image.shape)
print(image.dtype)
print(image[1,100])
io.imshow(image)
io.show()

print("\nastronaut picture information before conversion:")
print(image2.shape)
print(image2.dtype)
print(image2[1,100, 0])
io.imshow(image2)
io.show()


##TASK 2

grayscale = rgb2gray(image2)
print("\nastronaut picture information after conversion to grayscale:")
print(grayscale.shape)
print(grayscale.dtype)
print(grayscale[1,100])
io.imshow(grayscale)
io.show()

##TASK 3

image_rescaled = rescale(grayscale, 0.25, anti_aliasing=False)
print("\nastronaut grayscale picture information after rescaling:")
print(image_rescaled.shape)
print(image_rescaled.dtype)
print(image_rescaled[1,30])
io.imshow(image_rescaled)
io.show()


image_resized=resize(grayscale, (2048, 2048),
anti_aliasing=True)
print("\nastronaut grayscale picture information after resizing:")
print(image_resized.shape)
print(image_resized.dtype)
print(image_resized[1,300])
io.imshow(image_resized)
io.show()

##TASK 4

ax = plt.hist(grayscale.ravel(), bins = 256)
t = 0.6
binary = grayscale < t
fig, ax = plt.subplots()
plt.imshow(binary, cmap="gray")
plt.show()


##TASK 5

image = data.coins()
coin = image[170:220, 75:130]
result = match_template(image, coin)
ij = np.unravel_index(np.argmax(result), result.shape)
x, y = ij[::-1]
fig = plt.figure(figsize=(8, 3))
ax1 = plt.subplot(1, 3, 1)
ax2 = plt.subplot(1, 3, 2)
ax3 = plt.subplot(1, 3, 3, sharex=ax2, sharey=ax2)
ax1.imshow(coin, cmap=plt.cm.gray)
ax1.set_axis_off()
ax1.set_title('template')
ax2.imshow(image, cmap=plt.cm.gray)
ax2.set_axis_off()
ax2.set_title('image')
# highlight matched region
hcoin, wcoin = coin.shape
rect = plt.Rectangle((x, y), wcoin, hcoin, edgecolor='r', facecolor='none')
ax2.add_patch(rect)
ax3.imshow(result)
ax3.set_axis_off()
ax3.set_title('`match_template`\nresult')
# highlight matched region
ax3.autoscale(False)
ax3.plot(x, y, 'o', markeredgecolor='r', markerfacecolor='none', markersize=10)
plt.show()