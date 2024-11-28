#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 17:07:36 2022

@author: soyeonmin
"""
import os
import tqdm
DATA_FOLDER=os.environ['DATA_DIR']
DATA_FOLDER='/home/soyeonm/projects/Alexa_april_17/teach_data_collection/teach_depth_data_collection/TEACH_DOWNLOAD'
import json
from pathlib import Path
from itertools import chain
all_train_edh_files = list(
    (Path(DATA_FOLDER)/"edh_instances/train").glob("*.json")
)
all_valid_unseen_edh_files = list(chain(
    (Path(DATA_FOLDER)/"edh_instances/valid_unseen").glob("*.json")
))

game_id_seen = {}
for edh_files in tqdm.tqdm(all_train_edh_files ):
   game_id = json.load(edh_files.open("r"))['game_id']
   game_id_seen[game_id] =1

game_id_unseen = {}
for edh_files in tqdm.tqdm(all_valid_unseen_edh_files):
   game_id = json.load(edh_files.open("r"))['game_id']
   game_id_unseen[game_id] =1

#Divide game files in current horizon


from glob import glob
game_folders_depth = glob('Horizon_30/depth/*')
game_folders_hor = [d.replace('depth', 'hor') for d in game_folders_depth]
game_folders_rgb = [d.replace('depth', 'rgb') for d in game_folders_depth]


#Just get the ones that have all three

invalid = []
for gf in game_folders_depth:
    d_pickles =  glob(gf + '/*')
    for d in d_pickles:
        if os.path.exists(d.replace('depth', 'rgb') + "ng") and os.path.exists(d.replace('depth', 'hor').replace('hor_', 'horizon_')):
            pass
        else:
            invalid.append(d)
            
#Remove the invalids
assert len(invalid) == 0

#Divide now
game_folders_depth = glob('Horizon_30/depth/*')
game_folders_hor = [d.replace('depth', 'hor') for d in game_folders_depth]
game_folders_rgb = [d.replace('depth', 'rgb') for d in game_folders_depth]

weird = []
for gf in game_folders_depth:
    if gf.split('/')[-1].split('.')[0] in game_id_seen:
        pass
    elif gf.split('/')[-1].split('.')[0] in game_id_unseen:
        pass
    else:
        weird.append(gf)

weird = set(weird)

import shutil

train_new_loc = '/home/soyeonm/projects/Alexa_april_17/teach_data_collection/teach_depth_data_collection/Depth_Formatted/Horizon_30/sync/'
val_new_loc = '/home/soyeonm/projects/Alexa_april_17/teach_data_collection/teach_depth_data_collection/Depth_Formatted/Horizon_30/test/'
#Now for the gf's not in weird, move them
assert len(game_folders_rgb) == len(game_folders_depth)
assert len(game_folders_hor) == len(game_folders_depth)

not_exsits_hor = []
not_exsits_rgb= []
for i, gf in enumerate(game_folders_depth):
    if not(gf in weird):
        if gf.split('/')[-1].split('.')[0] in game_id_seen:
            new_file = train_new_loc + '/'.join(gf.split('/')[1:])
            new_dir = '/'.join(new_file.split('/')[:-1])
            if not (os.path.exists(new_dir)):
                os.makedirs(new_dir)
            shutil.move(gf,  new_file)
            
            #Do the same for hor
            gf_hor = game_folders_hor[i]
            new_file = train_new_loc + '/'.join(gf_hor.split('/')[1:])
            new_dir = '/'.join(new_file.split('/')[:-1])
            if not (os.path.exists(new_dir)):
                os.makedirs(new_dir)
            if not(os.path.exists(gf_hor)):
                not_exsits_hor.append(gf_hor)
            else:
                shutil.move(gf_hor,  new_file)
            
            gf_rgb = game_folders_rgb[i]
            new_file = train_new_loc + '/'.join(gf_rgb.split('/')[1:])
            new_dir = '/'.join(new_file.split('/')[:-1])
            if not (os.path.exists(new_dir)):
                os.makedirs(new_dir)
            if not(os.path.exists(gf_rgb)):
                not_exsits_rgb.append(gf_rgb)
            else:
                shutil.move(gf_rgb,  new_file)
            
            
        elif gf.split('/')[-1].split('.')[0] in game_id_unseen:
            new_file = val_new_loc + '/'.join(gf.split('/')[1:])
            new_dir = '/'.join(new_file.split('/')[:-1])
            if not (os.path.exists(new_dir)):
                os.makedirs(new_dir)
            shutil.move(gf,  new_file)
            
            gf_hor = game_folders_hor[i]
            new_file = val_new_loc + '/'.join(gf_hor.split('/')[1:])
            new_dir = '/'.join(new_file.split('/')[:-1])
            if not (os.path.exists(new_dir)):
                os.makedirs(new_dir)
            if not(os.path.exists(gf_hor)):
                not_exsits_hor.append(gf_hor)
            else:
                shutil.move(gf_hor,  new_file)
            
            gf_rgb = game_folders_rgb[i]
            new_file = val_new_loc + '/'.join(gf_rgb.split('/')[1:])
            new_dir = '/'.join(new_file.split('/')[:-1])
            if not (os.path.exists(new_dir)):
                os.makedirs(new_dir)
            if not(os.path.exists(gf_rgb)):
                not_exsits_rgb.append(gf_rgb)
            else:
                shutil.move(gf_rgb,  new_file)
            
        else:
            raise Exception("weird!")
            
            


        