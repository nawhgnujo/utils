import os
import sys

base = os.getcwd()
print(base)
base = base + '/' + sys.argv[1]
print(base)

path1 = base + '/inputs/' ##realA0.jpg
path2 = base + '/outputs/' ##GenB0_0.00296422.jpg
path3 = base + '/targets/' ##realB0_0.215984.jpg

list1 = os.listdir(path1)
list2 = os.listdir(path2)
list3 = os.listdir(path3)

os.chdir(path1)
for file in list1:
  new = file[5:]
  os.rename(file, new)

os.chdir(path2)
for file in list2:
  split = file.split('_')
  new = split[0][4:] + '.jpg'
  os.rename(file, new)

os.chdir(path3)
for file in list3:
  split = file.split('_')
  new = split[0][5:] + '.jpg'
  os.rename(file, new)
