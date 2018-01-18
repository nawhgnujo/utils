import os
##import sys
import cv2

base = '../ILSVRC/'
src_path_test = base + 'test.ori/'
tgt_path_test = base + 'test/'
src_path_val = base + 'val.ori/'
tgt_path_val = base + 'val/'


file_list = os.listdir(src_path_val)
##file_list.sort()


i = 0
for file in file_list:
  f_path = src_path_val + file
  img = cv2.imread(f_path)
  h, w = img.shape[:2]
  if h < 300 or w < 300: 
    continue
  ##print(f_path)
  crop_h = int(h/2)
  crop_w = int(w/2)
  ##print('{}:{}, h:{},crop {} w:{},crop {}'.format(i, file, h, crop_h, w, crop_w))
  crop = img[crop_h-128:crop_h+128, crop_w-128:crop_w+128]
  blur = cv2.GaussianBlur(crop, (7,7), 0)
  target = '{}val_crop/{}'.format(base, file)
  cv2.imwrite(target, crop)
  target = '{}val_blur/{}'.format(base, file)
  cv2.imwrite(target, blur)
  i += 1
  if i % 200 == 0:
    break
