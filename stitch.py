import numpy as np
import os
import glob
from skimage import io, color

imgA_dir = './train_blur/'
imgB_dir = './train_crop/'
out_dir = './train/'

if not os.path.exists(out_dir):
    os.mkdir(out_dir)

# filename list
imgA_filename = os.listdir(imgA_dir)
imgA_filename.sort()
numA = len(imgA_filename)
imgB_filename = os.listdir(imgB_dir)
imgB_filename.sort()
numB = len(imgB_filename)
if numA != numB:
    print("numA != numB")
    exit()

for i in range(numA):
    if (imgA_filename[i][-4:] == '.jpg') or (imgA_filename[i][-5:] == '.JPEG') or (imgA_filename[i][-4:] == '.png') or (imgA_filename[i][-4:] == '.JPG'):
        imgA = io.imread(imgA_dir + imgA_filename[i])
        ##imgA = color.gray2rgb(imgA)
    if (imgB_filename[i][-4:] == '.jpg') or (imgB_filename[i][-5:] == '.JPEG') or (imgB_filename[i][-4:] == '.png') or (imgB_filename[i][-4:] == '.JPG'):
        imgB = io.imread(imgB_dir + imgB_filename[i])
        ##imgB = color.gray2rgb(imgB)
    #print(imgA_filename[i])
    #print(imgA.shape)
    #print(imgB.shape)
    out_img = np.concatenate((imgA, imgB),axis=1)
    new_name = imgA_filename[i].replace('JPEG', 'jpg')
    io.imsave(out_dir + new_name, out_img)
