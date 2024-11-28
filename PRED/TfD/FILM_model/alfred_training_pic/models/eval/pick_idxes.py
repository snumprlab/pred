#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 17 03:11:47 2021

@author: soyeonmin
"""

#Get what to put in for alfred

training_pic_save_task_ids

train_files = json.load(open("/Volumes/Transcend/gittest/OGN/alfred_data_small/splits/oct21.json"))['train']
same_scenes = {}
#sliced = {}
#microwave = {}
for i , scene_name in enumerate(train_files):
    if i %3 == 0:
        #scene_name = files[i]
        loaded = json.load(open('/Volumes/Transcend/gittest/OGN/alfred_data_all/json_2.1.0/' + scene_name['task'] + '/pp/ann_' + str(scene_name['repeat_idx']) + '.json'))
        #if loaded['pddl_params']['object_sliced'] or loaded['task_type']=='pick_heat_then_place_in_recep':
        if not(loaded['scene']['floor_plan'][9:] in same_scenes):
            same_scenes[loaded['scene']['floor_plan'][9:]] = []
        same_scenes[loaded['scene']['floor_plan'][9:]].append(i)
        
import numpy as np
chosen = {k:[] for k in same_scenes}
for k, v in same_scenes.items():
    #scene 하나 당 5개만 고르기
    #get the rest from alfred
    counter = 0
    while len(chosen[k]) <10:
        np.random.seed(int(k) + counter)
        ch = np.random.choice(len(v))
        if not(v[ch] in chosen[k]):
            chosen[k].append(v[ch])
        counter +=1
        
all_chosens = []
for k, v in chosen.items():
    all_chosens += v
all_chosens = {v:1 for v in all_chosens}

all_chosens_final = {}
for i , scene_name in enumerate(train_files):
    loaded = json.load(open('/Volumes/Transcend/gittest/OGN/alfred_data_all/json_2.1.0/' + scene_name['task'] + '/pp/ann_' + str(scene_name['repeat_idx']) + '.json'))
    if i in all_chosens:
        all_chosens_final[(loaded['task_id'], loaded['repeat_idx'])] = 1

all_chosens_final_small = {}
c = 0
for i , scene_name in enumerate(train_files):
    loaded = json.load(open('/Volumes/Transcend/gittest/OGN/alfred_data_all/json_2.1.0/' + scene_name['task'] + '/pp/ann_' + str(scene_name['repeat_idx']) + '.json'))
    if i in all_chosens:
        all_chosens_final_small[(loaded['task_id'], loaded['repeat_idx'])] = 1
        c +=1
    if c==3:
        break

pickle.dump(all_chosens_final, open('/volumes/Transcend/alfred/alfred_training_pic_idxes_10.p', 'wb'))
pickle.dump(all_chosens_final, open('/volumes/Transcend/alfred/alfred_training_pic_idxes.p', 'wb'))
pickle.dump(all_chosens_final_small, open('/volumes/Transcend/alfred/alfred_training_pic_idxes_small.p', 'wb'))