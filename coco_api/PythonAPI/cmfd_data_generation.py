#!/usr/bin/env python
# coding: utf-8

# In[190]:


get_ipython().run_line_magic('matplotlib', 'inline')
from pycocotools.coco import COCO
import numpy as np
import cv2
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
pylab.rcParams['figure.figsize'] = (8.0, 10.0)
from matplotlib.patches import Polygon
import os
import random


# In[18]:


dataDir='../..'

train_data='train2017'
train_ann='{}/annotations/instances_{}.json'.format(dataDir, train_data)
val_data='val2017'
val_ann='{}/annotations/instances_{}.json'.format(dataDir,val_data)


# In[19]:


# initialize COCO api for instance annotations
# coco=COCO(train_ann)
coco = COCO(train_ann)


# In[20]:


# display COCO categories and supercategories
cats = coco.loadCats(coco.getCatIds())
nms=[cat['name'] for cat in cats]
print('COCO categories: \n{}\n'.format(' '.join(nms)))

super_cats = set([cat['supercategory'] for cat in cats])
print('COCO supercategories: \n{}'.format(' '.join(super_cats)))


# In[243]:


category = ['appliance']
print(type(category[0]))
print(len(nms))


# In[244]:


# get all images containing given categories, select one at random
catIds = coco.getCatIds(catNms=category)
imgIds = coco.getImgIds(catIds=catIds );
print(len(imgIds))


# In[246]:


for i in range(12):
    imgId = coco.getImgIds(imgIds=imgIds[i])
#     img = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]
    img = coco.loadImgs(imgIds[i])[0]
    print(imgId)
#     print(img)
    
    I = io.imread('%s/%s/%s'%(dataDir,train_data,img['file_name']))
#     I = io.imread(img['coco_url'])
#     plt.axis('off')
#     plt.figure(i+1)
#     plt.imshow(I)
    
plt.show()


# In[247]:


for i in range(15):
    imgId = coco.getImgIds(imgIds=imgIds[i])
#     img = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]
    img = coco.loadImgs(imgIds[i])[0]
    print(imgId)
#     print(img)
    
    I = io.imread('%s/%s/%s'%(dataDir,train_data,img['file_name']))
#     I = io.imread(img['coco_url'])
#     plt.figure(i+1)
#     plt.imshow(I); plt.axis('off')
#     annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
#     anns = coco.loadAnns(annIds)
#     conts = coco.showAnns(anns)


# In[248]:


# print(conts)
# wht_image = cv2.bitwise_not(np.zeros(I.shape, np.uint8))
# I = io.imread('%s/%s/%s'%(dataDir,dataType,img['file_name']))
# I = io.imread(img['coco_url'])
plt.figure(i+1)
plt.imshow(I); plt.axis('off')
annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
anns = coco.loadAnns(annIds)
conts = coco.showAnns(anns)

print(len(conts))

conts = np.asarray(conts)
# print(type(conts))
# print(conts.ndim)
# print(conts.shape)

cont = []

for i in range(conts.shape[0]):
    cn = conts[i]
#     cn = cn.astype(int)
#     print(cn.shape)
#     print(cn)
    cn = np.asarray(cn)
#     print(cn)
    cn = np.reshape(cn, (cn.shape[0], 1, cn.shape[1]))
    cn = cn.astype(int)
    print(cn.shape)
    cont+=[cn]
#     break

cont = sorted(cont, key = cv2.contourArea, reverse=True)
# for i in range(3):
#     print(cv2.contourArea(cont[i]))

c1 = cont[0]
# print(c1)

c1 = np.asarray(c1)
# print(c1.shape)
c1 = np.reshape(c1, (c1.shape[0], c1.shape[2]))
# print(c1)
# print(c1.shape)
# cont = np.asarray(cont)
# print(cont.shape)
if conts.shape[0]==1:
    cnt=conts
else:
    
    conts = np.reshape(conts, (conts.shape[0],1))
#     print(conts.shape)
    cnt = conts[0]
# print(conts)
# print(cnt.shape)
# print(cnt)
# print(type(cnt))
# print(cnt[0].shape)
# print(cnt[0])

# cnt = np.asarray(cnt)
# cntt = np.reshape(cnt[0], (cnt[0].shape[0], 1, cnt[0].shape[1]))
# print(cntt)
# cntt = cntt.astype(int)
# print(type(cntt))
# print(cntt.shape)
# print(cv2.contourArea(cntt))
# print(cntt[0][0])
# print(cnt)
# print(cnt)

# m = int(cnt)
# print(I[m])

# cv2.drawContours(I, np.int32([cnt[0]]), -1, (0,0,0), -1)

# plt.axis('off')
# plt.imshow(I)
# plt.show()
# io.imsave('forged_image.png', I)


# In[249]:


pts = c1
pts = pts.astype(int)
# print(pts)
# print(c1)
# print(pts)

## (1) Crop the bounding rect
rect = cv2.boundingRect(pts)
x,y,w,h = rect
print(x,y,w,h)
print(I.shape)

roi = I[0:h, 0:w]
croped = I[y:y+h, x:x+w].copy()
croped_2 = I.copy()



## (2) make mask
pts = pts - pts.min(axis=0)

c1 = c1 - c1.min(axis=0)
print(pts)
print(c1)
mask = np.zeros(croped.shape[:2], np.uint8)
mask_inv = cv2.bitwise_not(mask)
mask_2 = np.zeros(croped_2.shape[:2], np.uint8)
src_mask = np.zeros(croped.shape, croped.dtype)

cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)
cv2.fillPoly(src_mask, [pts], (255, 255, 255))
cv2.drawContours(mask_2, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)


## (3) do bit-op
dst = cv2.bitwise_and(croped, croped, mask=mask)
# dst_2 = cv2.bitwise_and(croped_2, croped_2, mask=mask_2)

# img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
# img2_fg = cv2.bitwise_and(croped, croped, mask=mask_inv)
# dst_1 = cv2.bitwise_and(I, I, mask=src_mask)
# dst_f = cv2.add(img1_bg, img2_fg)

# dst_w = cv2.addWeighted(img1_bg, 0.2, img2_fg, 0.8, 0)

# I3 = I.copy()
# I3[0:h, 0:w] = dst_f


## (4) add the white background
bg = np.ones_like(croped, np.uint8)*255
print(bg.shape)
cv2.bitwise_not(bg,bg, mask=mask)
dst2 = bg+ dst

# bg_2 = np.ones_like(I, np.uint8)*255
# print(bg_2.shape)
# cv2.bitwise_not(bg_2, bg_2, mask=mask)
# dst3 = bg_2 + dst
# io.imsave("dst3.png", dst3)

# io.imsave("croped.png", croped)
# io.imsave("mask.png", mask)
# io.imsave("dst.png", dst)
# io.imsave("dst2.png", dst2)
# io.imsave("bg.png", img1_bg)
io.imsave("full_mask.png", mask_2)
plt.figure()
plt.imshow(dst)
plt.figure()
plt.imshow(mask, cmap=plt.cm.gray)
# plt.figure()
# plt.imshow(img1_bg)
# plt.figure()
# plt.imshow(img2_fg)
# plt.figure()
# plt.imshow(I3)


# In[250]:


from skimage.transform import resize

# dst_resized = resize(dst, (dst.shape[0]//2, dst.shape[1]//2), anti_aliasing=True )
# print(dst_resized.max())
# dst_resized=  dst_resized*255
# print(dst_resized.max())
# dst2_resized = resize(dst2, (dst2.shape[0]//2, dst2.shape[1]//2), anti_aliasing=True)
# print(dst_resized.shape)
# plt.figure()
# plt.imshow(dst_resized)


scale_percent = 60 # percent of original size
width = int(croped.shape[1] * scale_percent / 100)
height = int(croped.shape[0] * scale_percent / 100)
dim = (width, height)

croped_resized = cv2.resize(croped, dim, interpolation = cv2.INTER_AREA)
dst_resized = cv2.resize(dst, dim, interpolation = cv2.INTER_AREA)
src_mask_resized = cv2.resize(src_mask, dim, interpolation = cv2.INTER_AREA)

plt.figure()
plt.imshow(dst_resized)
print(dst_resized.shape)
plt.figure()
plt.imshow(src_mask_resized)
print(src_mask_resized.shape)


# In[251]:


# print(cnt[0])
# adds = cv2.add(I, dst_resized)
# plt.imshow(adds)
# mask_resized = resize(mask, (mask.shape[0]//2, mask.shape[1]//2), anti_aliasing=True)
# croped_resized = resize(croped, (croped.shape[0]//2, croped.shape[1]//2), anti_aliasing=True)
x_offset=y_offset=50
# I[y_offset:y_offset+(croped_resized.shape[0]), x_offset:x_offset+ (croped_resized.shape[1])] = croped_resized
I2 = I.copy()
I2[I2.shape[0] - (croped_resized.shape[0]) - 10:I2.shape[0] - 10, 25:25 + (croped_resized.shape[1])] = croped_resized
# plt.imshow(cv2.cvtColor(I2, cv2.COLOR_BGR2RGB))
plt.imshow(I2)
# io.imsave("cmfd_image.png", I2)
plt.figure()
plt.imshow(I)


# In[233]:


# center1 = (150, 150)
# output_normal = cv2.seamlessClone(croped_resized, I, src_mask_resized, center1, cv2.NORMAL_CLONE)

# center2 = (400, 150)
# output_mixed = cv2.seamlessClone(dst_resized, I, src_mask_resized, center2, cv2.MIXED_CLONE)

# plt.figure()
# plt.imshow(output_normal)

# plt.figure()
# plt.imshow(output_mixed)

# io.imsave('normal_clone.png', output_normal)
# io.imsave('mixed_clone.png', output_mixed)




iii = []
for k in new_list:
    print(k)
    c = coco.getCatIds(catNms=k)
    iIds = coco.getImgIds(catIds=c)
    print(len(iIds))
    iii +=[len(iIds)]
print(sorted(iii))




#     for i in range(1):
#         imgId = coco.getImgIds(imgIds=iIds[i])
#     #     img = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]
#         img = coco.loadImgs(iIds[i])[0]
#         print(imgId)
#         print(img)

#         I = io.imread('%s/%s/%s'%(dataDir,train_data,img['file_name']))
#     #     I = io.imread(img['coco_url'])
#         plt.axis('off')
#         plt.figure(i+1)
#         plt.imshow(I)
#         annIds = coco.getAnnIds(imgIds=img['id'], catIds=c, iscrowd=None)
#         anns = coco.loadAnns(annIds)
#         conts = coco.showAnns(anns)
    
#     plt.show()

# g = nms.index("hot dog")
# print(g)
# new_list = nms.copy()
# new_list.pop(g)
# print(new_list.index("hot dog"))
# g = new_list.index("toaster")
# print(g)
# new_list.pop(g)
print((new_list))
print(iii)


# In[303]:


from os import listdir

remove_imgs = '../../coco_corrupted_images/'
r = listdir(remove_imgs)
print(len(r))
b =0
# for i in r:
# #     print(i)
#     k = i.split('.')[0]
# #     print(k, int(k))
#     print(b)
#     imgIds.remove(int(k))
#     b+=1
# imgIds.remove(398858)
# imgIds.remove(533408)
# imgIds.remove(405740)
# print(len(imgIds))
# print(len(new_list))
gg = new_list.index('microwave')
print(gg)
nn = new_list[gg:]
for k in nn:
    print(k)
#     k = ['motorcycle']
    categories = coco.getCatIds(catNms=k)
    imgIds = coco.getImgIds(catIds=categories)
    print(len(imgIds))
    for p in r:
        l = p.split('.')[0]
        l = int(l)
    #         print(l, int(l))
        if l in imgIds:
            imgIds.remove(l)
    print(len(imgIds))
    for i in range(705):
        imgId = coco.getImgIds(imgIds=imgIds[i])
    #     img = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]
        img = coco.loadImgs(imgIds[i])[0]
    #     print(imgId)
    #     print(img)
#         print(img['file_name'])
        I = io.imread('%s/%s/%s'%(dataDir,train_data,img['file_name']))
    
#     break


# In[302]:


remove_imgs = '../../coco_corrupted_images/'
r = listdir(remove_imgs)
k = ['microwave']
categories = coco.getCatIds(catNms=k)
imgIds = coco.getImgIds(catIds=categories)
print(len(imgIds))
for p in r:
    l = p.split('.')[0]
    l = int(l)
#         print(l, int(l))
    if l in imgIds:
        imgIds.remove(l)
print(len(imgIds))
for i in range(705):
    imgId = coco.getImgIds(imgIds=imgIds[i])
#     img = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]
    img = coco.loadImgs(imgIds[i])[0]
#     print(imgId)
#     print(img)
    print(img['file_name'])
    I = io.imread('%s/%s/%s'%(dataDir,train_data,img['file_name']))


# In[304]:


# for _ in 
if not os.path.exists('./cmf/authentic'):
    os.makedirs('./cmf/authentic')
if not os.path.exists('./cmf/tamper'):
    os.makedirs('./cmf/tamper')

remove_imgs = '../../coco_corrupted_images/'
r = listdir(remove_imgs)
    
    
for k in new_list:
    print(k)
    categories = coco.getCatIds(catNms=k)
    iIds = coco.getImgIds(catIds=categories)
    for p in r:
        l = p.split('.')[0]
        l = int(l)
    #         print(l, int(l))
        if l in iIds:
            iIds.remove(l)
    print(len(iIds))
    for i in range(705):
        imgId = coco.getImgIds(imgIds=iIds[i])
    #     img = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]
        img = coco.loadImgs(iIds[i])[0]
    #     print(imgId)
    #     print(img)

        I = io.imread('%s/%s/%s'%(dataDir,train_data,img['file_name']))
#         I = io.imread(img['coco_url'])
    #     I_resize = cv2.resize(I, (384, 256))
        I2 = I.copy()
        annIds = coco.getAnnIds(imgIds=img['id'], catIds=categories, iscrowd=None)
        anns = coco.loadAnns(annIds)
        conts = coco.showAnns(anns)
        conts = np.asarray(conts)
        # print(conts)
    #     print(conts.ndim)
    #     print(conts.shape)
        cont = []
    #     print(conts.shape)
        for j in range(conts.shape[0]):
            cn = conts[j]
        #     cn = cn.astype(int)
        #     print(cn.shape)
        #     print(cn)
            cn = np.asarray(cn)
        #     print(cn)
            cn = np.reshape(cn, (cn.shape[0], 1, cn.shape[1]))
            cn = cn.astype(int)
    #         print(cn.shape)
            cont+=[cn]
        #     break

        cont = sorted(cont, key = cv2.contourArea, reverse=True)

    #     if conts.shape[0]==1:
    #         cnt=conts
    #     else:

    #         conts = np.reshape(conts, (conts.shape[0],1))
    # #         print(conts.shape)
    #         cnt = conts[0]
    # #     print(cnt.shape)
    #     # print(cnt[0])
    # #     print(cnt[0].shape)
    #     cnt = np.asarray(cnt)
        c1 = cont[0]
        print(cv2.contourArea(c1))
        if cv2.contourArea(c1)>=4000:
            c1 = np.asarray(c1)
            # print(c1.shape)
            c1 = np.reshape(c1, (c1.shape[0], c1.shape[2]))
            pts = c1
            pts = pts.astype(int)

            # print(pts)

            ## (1) Crop the bounding rect
            rect = cv2.boundingRect(pts)
            x,y,w,h = rect
            print("area:" ,np.multiply(w,h))
        #     print(x,y,w,h)
        #     print(I.shape)

            roi = I[0:h, 0:w]
            croped = I[y:y+h, x:x+w].copy()
        #     croped_2 = I.copy()


            ## (2) make mask
            pts = pts - pts.min(axis=0)


            mask = np.zeros(croped.shape[:2], np.uint8)
            mask_inv = cv2.bitwise_not(mask)
        #     mask_2 = np.zeros(croped_2.shape[:2], np.uint8)
        #     src_mask = np.zeros(croped.shape, croped.dtype)

            cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)
        #     cv2.fillPoly(src_mask, [pts], (255, 255, 255))
        #     cv2.drawContours(mask_2, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)


            ## (3) do bit-op
            dst = cv2.bitwise_and(croped, croped, mask=mask)
            # dst_2 = cv2.bitwise_and(croped_2, croped_2, mask=mask_2)

        #     img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
        #     img2_fg = cv2.bitwise_and(croped, croped, mask=mask_inv)
            # dst_1 = cv2.bitwise_and(I, I, mask=src_mask)
        #     dst_f = cv2.add(img1_bg, img2_fg)

        #     dst_w = cv2.addWeighted(img1_bg, 0.2, img2_fg, 0.8, 0)

        #     I3 = I.copy()
        #     I3[0:h, 0:w] = dst_f


            ## (4) add the white background
            bg = np.ones_like(croped, np.uint8)*255
            cv2.bitwise_not(bg,bg, mask=mask)
            dst2 = bg+ dst

            scale_percent = 60 # percent of original size
            width = int(croped.shape[1] * scale_percent / 100)
            height = int(croped.shape[0] * scale_percent / 100)
            dim = (width, height)

            croped_resized = cv2.resize(croped, dim, interpolation = cv2.INTER_AREA)
        #     dst_resized = cv2.resize(dst, dim, interpolation = cv2.INTER_AREA)
        #     src_mask_resized = cv2.resize(src_mask, dim, interpolation = cv2.INTER_AREA)


#             y_offset=50 + random.randint(10, 100)
            
            # I[y_offset:y_offset+(croped_resized.shape[0]), x_offset:x_offset+ (croped_resized.shape[1])] = croped_resized
            y_offset = random.randint(I2.shape[0]//4, I2.shape[0] - croped_resized.shape[0])
#             y_offset = random.randint(0, I2.shape[0] - croped_resized.shape[0])
            x_offset = random.randint(0, I2.shape[1] - croped_resized.shape[1])
        #     I2[10:10+(croped_resized.shape[0]), 25:25 + (croped_resized.shape[1])] = croped_resized Fixed
#             I2[I2.shape[0] - (croped_resized.shape[0]) - y_offset:I2.shape[0] - y_offset, I2.shape[1] - x_offset - (croped_resized.shape[1]): I2.shape[1] - x_offset] = croped_resized # right
#             I2[I2.shape[0] - (croped_resized.shape[0]) - y_offset:I2.shape[0] - y_offset, x_offset: x_offset + (croped_resized.shape[1])] = croped_resized # left
            I2[y_offset: y_offset + (croped_resized.shape[0]), x_offset: x_offset + (croped_resized.shape[1])] = croped_resized
        # plt.imshow(cv2.cvtColor(I2, cv2.COLOR_BGR2RGB))
        #     plt.figure(i+1)
        #     plt.imshow(I2)


            io.imsave("./cmf/authentic/" + "cmfd_" + k + "_" + str(i) + ".png", I)
            io.imsave("./cmf/tamper/" + "cmfdr_"+ k + "_" + str(i) + ".png", I2)

        #     plt.figure(i+1)
        #     plt.imshow(I); plt.axis('off')
        #     annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
        #     anns = coco.loadAnns(annIds)
        #     conts = coco.showAnns(anns)


# In[ ]:


# print(dst2_resized)
M = cv2.moments(cnt[0])
print(M)
cX = int(M["m10"] / M["m00"])
cY = int(M["m01"] / M["m00"])
print(cX, cY)


# In[ ]:


# plt.axis('off')
# plt.imshow(wht_image)
# plt.show()
# io.imsave('mask_forged.png', wht_image)

# black_image = cv2.bitwise_not(np.ones(I.shape, np.uint8))
black_image = np.zeros(I.shape, np.uint8)
plt.figure()
# plt.imshow(black_image)
cv2.drawContours(black_image, np.int32([cnt[0]]), -1, (255, 255, 255), -1)
plt.axis('off')
plt.imshow(black_image)
plt.show()
io.imsave('mask_forged_inverse.png', black_image)

# for cnt in conts:
#     print(cnt.shape)
# #     cnt = np.array(cnt).reshape((-1,46,2)).astype(numpy.int32)
#     cv2.drawContours(I, np.int32([cnt[0]]), 0, (0,255,0), 2)
#     plt.imshow(I)
#     plt.show()
# #     cv2.waitKey()
# Polygon()
# print()
# p = Pat


# In[ ]:


mask = coco.annToMask(anns[0])
print(mask)
for i in range(len(anns)):
    mask += coco.annToMask(anns[i])
plt.imshow(mask, cmap=plt.cm.gray)
plt.axis('off')
# plt.savefig('asjdbkba.png')
plt.imsave('mask _mofo.png', mask, cmap=plt.cm.gray)
mnb = cv2.resize(mask, (256, 256))
plt.imsave('resizesssadas.png', mnb, cmap = plt.cm.gray)


# In[ ]:


print(len(anns))
mask = coco.annToMask(anns[0])
print(mask)
plt.imshow(mask) 
new_mask = cv2.resize(mask, (256, 256))
plt.imsave('check_size.png', new_mask, cmap=plt.cm.gray)


# In[ ]:


if len(anns)>1:
    mask+=coco.annToMask(anns[1])
# else:
#     continue
plt.imshow(mask, cmap=plt.cm.gray)


# In[ ]:


print(imgId)
plt.imshow(I);
annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
print(len(annIds))
anns = coco.loadAnns(annIds[0])
print(len(anns))
coco.showAnns(anns)
print(type(anns))


# In[ ]:





# In[ ]:




