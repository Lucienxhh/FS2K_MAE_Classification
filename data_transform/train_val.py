########### to transform the format of dataset to torchvision.datasets
import os
import json
import numpy as np
import shutil
import random

def makeDir(set):
    def make(path):
        if not os.path.exists(path):
            os.makedirs(path)
    make('hair/'+set)
    make('hair/'+set+'No/')
    make('hair/'+set+'Yes/')
    
    make('color/'+set)
    make('color/'+set+'Brown/')
    make('color/'+set+'Black/')
    make('color/'+set+'Red/')
    make('color/'+set+'No/')
    make('color/'+set+'Golden/')

    make('sex/'+set)
    make('sex/'+set+'Male/')
    make('sex/'+set+'Female/')

    make('earring/'+set)
    make('earring/'+set+'No/')
    make('earring/'+set+'Yes/')

    make('smile/'+set)
    make('smile/'+set+'No/')
    make('smile/'+set+'Yes/')

    make('frontal_face/'+set)
    make('frontal_face/'+set+'Up/')
    make('frontal_face/'+set+'Low/')

json_file_train = 'anno_train.json'


with open(json_file_train, 'r') as f:
    json_data = json.loads(f.read())

inputDir = 'train/sketch/'
makeDir(set='train/')
makeDir(set='val/')


attrs = {}
for attr in json_data[0].keys():
    attrs[attr] = []
for idx_fs, fs in enumerate(json_data):
    for attr in fs:
        attrs[attr].append(fs[attr])

photo_paths = []
for photo_dir in os.listdir(inputDir):
    photo_paths.append(inputDir+photo_dir)

total = len(attrs['image_name'])
ran = random.sample(range(0, total), total)
val = ran[0:int(total/3)]
train = ran[int(total/3):]

for idx_image_name, image_name in enumerate(attrs['image_name']):
    if idx_image_name < 0:
        continue
    print('{}/{},'.format(idx_image_name+1, total), image_name)
    image_name = image_name.replace("photo1", "sketch1_sketch")
    image_name = image_name.replace("photo2", "sketch2_sketch")
    image_name = image_name.replace("photo3", "sketch3_sketch")
    image_name = image_name.replace("/image", "")
    for photo_path in photo_paths:
        if image_name in photo_path:
            src_path = photo_path
            suffix = photo_path[-4:]
            break
    dst_file_name = image_name + suffix
    hair = int(attrs['hair'][idx_image_name])
    if 'nan' not in attrs['lip_color'][idx_image_name]:
        hair_color = np.array(attrs['hair_color'][idx_image_name]).astype(np.uint8)
    gender = int(attrs['gender'][idx_image_name])
    earring = int(attrs['earring'][idx_image_name])
    smile = int(attrs['smile'][idx_image_name])
    frontal_face = int(attrs['frontal_face'][idx_image_name])
    if idx_image_name in val:
        set = 'val/'
    else:
        set = 'train/'
    if hair == 0:
        shutil.copy2(src_path, 'hair/'+set+'Yes/'+dst_file_name)
    else:
        shutil.copy2(src_path, 'hair/'+set+'No/'+dst_file_name)
    
    if hair_color == 0:
        shutil.copy2(src_path, 'color/'+set+'Brown/'+dst_file_name)
    elif hair_color == 1:
        shutil.copy2(src_path, 'color/'+set+'Black/'+dst_file_name)
    elif hair_color == 2:
        shutil.copy2(src_path, 'color/'+set+'Red/'+dst_file_name)
    elif hair_color == 3:
        shutil.copy2(src_path, 'color/'+set+'No/'+dst_file_name)
    elif hair_color == 4:
        shutil.copy2(src_path, 'color/'+set+'Golden/'+dst_file_name)
        
    if gender == 0:
        shutil.copy2(src_path, 'sex/'+set+'Male/'+dst_file_name)
    else:
        shutil.copy2(src_path, 'sex/'+set+'Female/'+dst_file_name)
    
    if earring == 0:
        shutil.copy2(src_path, 'earring/'+set+'Yes/'+dst_file_name)
    else:
        shutil.copy2(src_path, 'earring/'+set+'No/'+dst_file_name)
    
    if smile == 0:
        shutil.copy2(src_path, 'smile/'+set+'No/'+dst_file_name)
    else:
        shutil.copy2(src_path, 'smile/'+set+'Yes/'+dst_file_name)
    
    if frontal_face == 0:
        shutil.copy2(src_path, 'frontal_face/'+set+'Low/'+dst_file_name)
    else:
        shutil.copy2(src_path, 'frontal_face/'+set+'Up/'+dst_file_name)





