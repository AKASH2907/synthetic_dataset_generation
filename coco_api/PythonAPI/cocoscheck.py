
from os.path import isfile, join
from os import rename, listdir, rename, makedirs
import skimage.io as io

train = './train2017'
files = listdir(train)

print(len(files))

c=0
j=0
for i in files:
	img = io.imread(join(train, i))
	# print(img.shape)
	c+=1
	if(c%1000)==0:
		j+=1
		print( j,"000 done")