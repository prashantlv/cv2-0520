import matplotlib.pyplot as plt 
import numpy as np
import cv2

image = cv2.imread('/home/prashant/Pictures/ginger.jpg') # image with blue background

print('This image is With dimension :', image.shape)

image_copy = np.copy(image)
image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)
plt.imshow(image_copy)

#define color threshold

lower_blue = np.array([0,0,181])
upper_blue = np.array([250,250,255])

# define a mask

mask = cv2.inRange(image_copy, lower_blue, upper_blue)
#plt.imshow(mask, cmap = 'gray')

# Mask the image to let the girl show through
masked_image = np.copy(image_copy)
masked_image[mask !=0] = [0,0,0]
plt.imshow(masked_image)

#load in as background image
background_image = cv2.imread('/home/prashant/Pictures/space.jpeg')
#plt.imshow(background_image)
backgroung_image = cv2.cvtColor(background_image, cv2.COLOR_BGR2RGB)

#crop it to right size
crop_background = background_image[0:426, 0:626]
# Mask the cropped background so that the pizza area is blocked
crop_background[mask == 0 ]	 = [0, 0, 0]
#plt.imshow(crop_background)


# Add the two images together to create a complete image!
complete_image = masked_image + crop_background

# Display the result
plt.imshow(complete_image)



plt.show()

