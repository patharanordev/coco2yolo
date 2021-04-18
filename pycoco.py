# %matplotlib inline
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import requests
import os
import cv2
import shutil

def get_yolo_image_box(line, img_width, img_height):
    columns = line.split(' ')
    class_id = columns[0]
    center_x = float(columns[1]) * img_width
    center_y = float(columns[2]) * img_height
    obj_half_width = float(columns[3]) * img_width / 2
    obj_half_height = float(columns[4]) * img_height / 2
    x_min = int(center_x - obj_half_width)
    y_min = int(center_y - obj_half_height)
    x_max = int(center_x + obj_half_width)
    y_max = int(center_y + obj_half_height)
    return class_id, x_min, y_min, x_max, y_max

# pylab.rcParams['figure.figsize'] = (8.0, 10.0)

output = './output'
dataDir='/Users/patharanor/Desktop/datasets/coco_dataset'
dataType='train2017'
annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)

# initialize COCO api for instance annotations
coco=COCO(annFile)

# display COCO categories and supercategories
cats = coco.loadCats(coco.getCatIds())
nms=[cat['name'] for cat in cats]
print('COCO categories: \n{}\n'.format(' '.join(nms)))

nms = set([cat['supercategory'] for cat in cats])
print('COCO supercategories: \n{}'.format(' '.join(nms)))

# get all images containing given categories, select one at random
reCatIds = [3,4]
catNms = ['person','cell phone']
catIds = coco.getCatIds(catNms=catNms)
# print('category id:', catIds)

if os.path.isdir(output):
    shutil.rmtree(output)

# for cls_name in catNms:
#     os.makedirs('{}/{}'.format(output, cls_name))

annIds = coco.getAnnIds(catIds=catIds, iscrowd=None)
all_anns = coco.loadAnns(annIds)

for i in range(0, len(all_anns)):
    cur_ann = all_anns[i]
    # print(cur_ann)

    segmentation = cur_ann['segmentation']
    area = cur_ann['area']
    iscrowd = cur_ann['iscrowd']
    category_id = cur_ann['category_id']
    category_name = catNms[catIds.index(category_id)]
    cbbox = cur_ann['bbox']
    cimg_info = coco.loadImgs(cur_ann['image_id'])

    if len(cimg_info) > 1:
        print('Found more than one image')
        break
        
    print('current category info :', cimg_info)
    coco_url  = cimg_info[0]["coco_url"]
    fname_ext = cimg_info[0]["file_name"]
    fname     = fname_ext.split('.')
    width     = cimg_info[0]["width"]
    height    = cimg_info[0]["height"]
    xmin      = int(cbbox[0])
    ymin      = int(cbbox[1])
    xmax      = min(int(xmin + cbbox[2]), width-1)
    ymax      = min(int(ymin + cbbox[3]), height-1)

    fext      = fname.pop()
    fname     = ''.join(fname)

    norm_center_x = ((xmax + xmin)/2)/width
    norm_center_y = ((ymax + ymin)/2)/height
    norm_width = (xmax - xmin)/width
    norm_height = (ymax - ymin)/height

    # print('current category name :', category_name)
    one_line = '{} {} {} {} {}'.format(
        reCatIds[catIds.index(category_id)], 
        norm_center_x, norm_center_y, norm_width, norm_height
    )

    print('{} : {}'.format(category_name, one_line))

    # class_id, x_min, y_min, x_max, y_max = get_yolo_image_box(one_line, width, height)
    # print('x_min:', x_min==xmin)
    # print('y_min:', y_min==ymin)
    # print('x_max:', x_max==xmax)
    # print('y_max:', y_max==ymax)

    img_new_fname = '{}/COCO_{}.{}'.format(output, fname, fext)
    txt_new_fname = '{}/COCO_{}.txt'.format(output, fname)

    if not os.path.isfile(img_new_fname):
        response = requests.get(coco_url)
        with open(img_new_fname, 'wb') as f:
            f.write(response.content)

    with open(txt_new_fname, 'a+') as f:
        f.write('{}\n'.format(one_line))

    # break

print('total:', len(all_anns))

# imgIds = coco.getImgIds(catIds=catIds )
# # imgIds = coco.getImgIds(imgIds = [324158])
# img = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]

# # load and display image
# # I = io.imread('%s/images/%s/%s'%(dataDir,dataType,img['file_name']))
# # use url to load image
# I = io.imread(img['coco_url'])
# plt.axis('off')
# plt.imshow(I)
# plt.show()

# # load and display instance annotations
# plt.imshow(I); plt.axis('off')
# annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
# anns = coco.loadAnns(annIds)
# coco.showAnns(anns)

# # initialize COCO api for person keypoints annotations
# annFile = '{}/annotations/person_keypoints_{}.json'.format(dataDir,dataType)
# coco_kps=COCO(annFile)

# # load and display keypoints annotations
# plt.imshow(I); plt.axis('off')
# ax = plt.gca()
# annIds = coco_kps.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
# anns = coco_kps.loadAnns(annIds)
# coco_kps.showAnns(anns)

# # initialize COCO api for caption annotations
# annFile = '{}/annotations/captions_{}.json'.format(dataDir,dataType)
# coco_caps=COCO(annFile)

# # load and display caption annotations
# annIds = coco_caps.getAnnIds(imgIds=img['id'])
# anns = coco_caps.loadAnns(annIds)
# coco_caps.showAnns(anns)
# plt.imshow(I); plt.axis('off'); plt.show()