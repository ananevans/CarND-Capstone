# Mohamed Elbanhawi (mohamedbanhawi@gmail.com)

import cv2
import numpy as np
import random

class DataAugmentation:

    def __self__():
        print ('New DataAugmentation Class')

    # augmentation helper functions
    def blur_dataset(self, images, alpha):
        dim = len(images.shape)
        if dim ==4:
            for i in range(images.shape[0]):
                im = np.copy(images[i])
                dst = cv2.blur(im,(alpha,alpha))
                images[i] = dst
        elif dim ==3:
            im = np.copy(images)
            dst = cv2.blur(im,(alpha,alpha))
            images = dst
        return images

    def adjust_gamma(self, images, gamma=1.0):
        dim = len(images.shape)
        
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
          for i in np.arange(0, 256)]).astype("uint8")
        
        if dim ==3:
            images = cv2.LUT(images, table)
        elif dim ==4:
            for i in range(images.shape[0]):
                im = images[i]
                im = cv2.LUT(im, table)
                images[i] = im
        return images

    def rotate_dataset(self, images, angle):
        dim = len(images.shape)
        if dim ==4:
            for i in range(images.shape[0]):
                im = np.copy(images[i])
                rows,cols,channel = im.shape
                M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
                dst = cv2.warpAffine(im,M,(cols,rows))
                images[i] = np.copy(dst)
        elif dim == 3:
            im = np.copy(images)
            rows,cols,channel = im.shape
            M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
            dst = cv2.warpAffine(im,M,(cols,rows))
            images = np.copy(dst)
        return images

    def shift_dataset(self, images, vector):
        dim = len(images.shape)
        if dim == 4:
            for i in range(images.shape[0]):
                im = np.copy(images[i])
                rows,cols,channel = im.shape
                M = np.float32([[1,0,vector],[0,1,vector]])
                dst = cv2.warpAffine(im,M,(cols,rows))
                images[i] = dst
        elif dim ==3:
            im = np.copy(images)
            rows,cols,channel = im.shape
            M = np.float32([[1,0,vector],[0,1,vector]])
            dst = cv2.warpAffine(im,M,(cols,rows))
            images = dst
        return images

    def random_jitter(self, x):
        jitter_func = random.randint(1,4) 
        
        if jitter_func == 1:
            alpha = 1
            if random.random() > 0.5:
                alpha = 3
            x = blur_dataset(x, alpha)

        if jitter_func == 2:
            theta = random.random()*30 - 15  # rotate [-15,15]
            x = rotate_dataset(x, theta)
            
        if jitter_func == 3:
            dx = random.random()*20 - 10 # translate [-10,10]
            x = shift_dataset(x, dx)
            
        if jitter_func == 4:
            alpha = 0.35
            if random.random() > 0.5:
                alpha = 5
            x = adjust_gamma(x, gamma)
        return x