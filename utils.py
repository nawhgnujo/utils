import sys
import cv2
import numpy as np
import scipy.misc
import math
import os
import glob
import hashlib
from scipy.ndimage import gaussian_filter

base = os.getcwd()
print(base)
work_path = base + '/' + sys.argv[1]
print(base)

path1 = work_path + 'inputs/' ##realA0.jpg
path2 = work_path + 'outputs/' ##GenB0_0.00296422.jpg
path3 = work_path + 'targets/' ##realB0_0.215984.jpg

##################################################################
def sha256_checksum(filename, block_size=65536):
  sha256 = hashlib.sha256()
  with open(filename, 'rb') as f:
    for block in iter(lambda: f.read(block_size), b''):
      sha256.update(block)
  return sha256.hexdigest()
##################################################################
def reorder_files():
  list1 = os.listdir(path1)
  for file in list1:
    checksum = sha256_checksum(path1 + file)
    new = checksum + '.jpg'
    os.rename(path1 + file, path1 + new)
    os.rename(path2 + file, path2 + new)
    os.rename(path3 + file, path3 + new)
  return 0
##################################################################
def change_filename():
  list1 = os.listdir(path1)
  list2 = os.listdir(path2)
  list3 = os.listdir(path3)

  for file in list1:
    split1 = file.split('.')
    split2 = split1[0].split('_')
    new = split2[0] + split2[2] + '.jpg'
    os.rename(path1 + file, path1 + new)

  for file in list2:
    split1 = file.split('.')
    split2 = split1[0].split('_')
    new = split2[0] + split2[2] + '.jpg'
    os.rename(path2 + file, path2 + new)

  for file in list3:
    split1 = file.split('.')
    split2 = split1[0].split('_')
    new = split2[0] + split2[2] + '.jpg'
    os.rename(path3 + file, path3 + new)
##################################################################
def change_filename2(fpath):
  for file in glob.glob(fpath):
    new_name = file.replace('-outputs.png', '.png')
    os.rename(file, new_name)
##################################################################
def change_filename3():

  if os.path.isfile(work_path + 'index.html'):
    print('exist')
    exit()
  else:
    print('change file names')

  list1 = os.listdir(path1)
  list2 = os.listdir(path2)
  list3 = os.listdir(path3)

  os.chdir(path1)
  for file in list1:
    new = file[5:]
    os.rename(file, new)
  os.chdir(base)

  for file in list2:
    split = file.split('_')
    new = split[0][4:] + '.jpg'
    os.rename(path2 + file, path2 + new)

  for file in list3:
    split = file.split('_')
    new = split[0][5:] + '.jpg'
    os.rename(path3 + file, path3 + new)
##################################################################
def psnr(img1, img2):
  mse = np.mean( (img1 - img2) ** 2 )
  if mse == 0:
    return 100
  PIXEL_MAX = 255.0
  return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
##################################################################
def ssim(img1, img2, sd=1.5, C1=0.01**2, C2=0.03**2):
  mu1 = gaussian_filter(img1, sd)
  mu2 = gaussian_filter(img2, sd)

  mu1_sq = mu1 * mu1
  mu2_sq = mu2 * mu2
  mu1_mu2 = mu1 * mu2

  sigma1_sq = gaussian_filter(img1 * img1, sd) - mu1_sq
  sigma2_sq = gaussian_filter(img2 * img2, sd) - mu2_sq
  sigma12 = gaussian_filter(img1 * img2, sd) - mu1_mu2

  ssim_num = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2))
  ssim_den = ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
  ssim_map = ssim_num / ssim_den

  return np.mean(ssim_map)
##################################################################
def cv2_read_write():
  image = cv2.imread('ztest.jpg')
  h, w = image.shape[:2]
  print('h:{}, w:{}'.format(h, w))

  cv2.imshow('original', image)
  cv2.waitKey(0)
  aaa = image[0:255, 0:255]
  bbb = image[0:255, 256:511]
  psnr_value = psnr(aaa, bbb)
##################################################################
def make_html():
  if os.path.isfile(work_path + 'index.html'):
    print('exist')
    exit()

  print('make html file')

  html = open(work_path + 'index.html', 'w')
  html.write('<html><body><table>')
  html.write('<tr><th>NAME</th><th>INPUT(blur)</th><th>OUTPUT(U-Net)</th><th>OUPUT(ResNet)</th><th>TARGET</th></tr>')

  psnr_count = [0, 0, 0]
  ssim_count = [0, 0, 0]

  filelist = os.listdir(path1)
  filelist.sort()

  avg_psnr = [0, 0, 0]

  index = 0
  for file in filelist:
    img_path1 = './inputs/' + file ##os.path.join('./', 'inputs/' + file)
    img_path2 = './outputs/' + file ##os.path.join('./', 'outputs/' + file)
    img_path3 = './outputs/' + file ##os.path.join('./', 'outputs/' + file)
    img_path4 = './targets/' + file ##os.path.join('./', 'targets/' + file)

    img1 = scipy.misc.imread(work_path + img_path1, flatten=True).astype(np.float32)
    img2 = scipy.misc.imread(work_path + img_path2, flatten=True).astype(np.float32)
    img3 = scipy.misc.imread(work_path + img_path3, flatten=True).astype(np.float32)
    img4 = scipy.misc.imread(work_path + img_path4, flatten=True).astype(np.float32)

    psnr1 = psnr(img1, img4)
    psnr2 = psnr(img2, img4)
    psnr3 = psnr(img3, img4)
    ssim1 = ssim(img1/255, img4/255)
    ssim2 = ssim(img2/255, img4/255)
    ssim3 = ssim(img3/255, img4/255)

    avg_psnr[0] += psnr1
    avg_psnr[1] += psnr2
    avg_psnr[2] += psnr3

    psnr_values = [psnr1, psnr2, psnr3]
    max_psnr_index = psnr_values.index( max(psnr_values) )
    ssim_values = [ssim1, ssim2, ssim3]
    max_ssim_index = ssim_values.index( max(ssim_values) )

    psnr_color = ['black', 'black', 'black']
    psnr_color[max_psnr_index] = 'red'
    ssim_color = ['black', 'black', 'black']
    ssim_color[max_ssim_index] = 'red'

    psnr_count[max_psnr_index] += 1
    ssim_count[max_ssim_index] += 1

    html.write('<tr>')
    html.write('<td>IMG%d</td>' % index)
    index += 1
    html.write("<td><img src='%s'></td>" % img_path1)
    html.write("<td><img src='%s'></td>" % img_path2)
    html.write("<td><img src='%s'></td>" % img_path3)
    html.write("<td><img src='%s'></td>" % img_path4)
    html.write('</tr><tr>')

    str = "<td></td><td><b><font color='{}'>PSNR : {}</font><br><font color='{}'>SSIM : {}</font></b></td>".format(psnr_color[0], psnr1, ssim_color[0], ssim1)
    str = str + "<td><b><font color='{}'>PSNR : {}</font><br><font color='{}'>SSIM : {}</font></b></td>".format(psnr_color[1], psnr2, ssim_color[1], ssim2)
    str = str + "<td><b><font color='{}'>PSNR : {}</font><br><font color='{}'>SSIM : {}</font></b></td>".format(psnr_color[2], psnr3, ssim_color[2], ssim3)
    str = str + "<td></td>"
    html.write(str)
    html.write('</tr>')


  html.write('<tr>')
  html.write('<td colspan=5 height=50></td></tr><tr>')

  str = "<td></td><td><b>PSNR wins: {}<br>SSIM wins: {}</b></td>".format(psnr_count[0], ssim_count[0])
  str = str + "<td><b>PSNR wins: {}<br>SSIM wins: {}</b></td>".format(psnr_count[1], ssim_count[1])
  str = str + "<td><b>PSNR wins: {}<br>SSIM wins: {}</b></td><td></td>".format(psnr_count[2], ssim_count[2])
  html.write(str)
  html.write('</tr>')
  html.write('</table></body></html>')

  print('PSNR wins = {}, {}, {}'.format(psnr_count[0], psnr_count[1], psnr_count[2]))
  print('SSIM wins = {}, {}, {}'.format(ssim_count[0], ssim_count[1], ssim_count[2]))
  print('AVG PSNR = {}, {}'.format(avg_psnr[0]/200, avg_psnr[1]/200))
##################################################################
def qualities():

  filelist = os.listdir(path1)
  filelist.sort()

  avg_psnr = [0, 0, 0]
  avg_ssim = [0, 0, 0]

  index = 0
  for file in filelist:
    img_path1 = './inputs/' + file ##os.path.join('./', 'inputs/' + file)
    img_path2 = './outputs/' + file ##os.path.join('./', 'outputs/' + file)
    img_path3 = './outputs/' + file ##os.path.join('./', 'outputs/' + file)
    img_path4 = './targets/' + file ##os.path.join('./', 'targets/' + file)

    img1 = scipy.misc.imread(work_path + img_path1, flatten=True).astype(np.float32)
    img2 = scipy.misc.imread(work_path + img_path2, flatten=True).astype(np.float32)
    img3 = scipy.misc.imread(work_path + img_path3, flatten=True).astype(np.float32)
    img4 = scipy.misc.imread(work_path + img_path4, flatten=True).astype(np.float32)

    psnr1 = psnr(img1, img4)
    psnr2 = psnr(img2, img4)
    psnr3 = psnr(img3, img4)
    ssim1 = ssim(img1/255, img4/255)
    ssim2 = ssim(img2/255, img4/255)
    ssim3 = ssim(img3/255, img4/255)

    avg_psnr[0] += psnr1
    avg_psnr[1] += psnr2
    avg_psnr[2] += psnr3

    avg_ssim[0] += ssim1
    avg_ssim[1] += ssim2
    avg_ssim[2] += ssim3

  print('AVG PSNR = {}, {}'.format(avg_psnr[0]/200, avg_psnr[1]/200))
  print('AVG SSIM = {}, {}'.format(avg_ssim[0]/200, avg_ssim[1]/200))
##################################################################
def test():
  a = 123
  b = 222
  c = 23
  l = [a, b, c,]
  print('{}'.format(max(l)))
  print('{}'.format( l.index(max(l)) ))

  dd = {1 : 'aaaa', 2 : 'bbbb'}.get(3, 'cccc')
  print(dd)
##################################################################


change_filename()
##make_html()
##qualities()
