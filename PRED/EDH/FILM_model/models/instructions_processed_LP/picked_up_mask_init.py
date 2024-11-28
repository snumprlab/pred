#Get picked up mask temporary script
import os
os.chdir('/Users/soyeonmin/Documents/Alexa/teach/')
teach_dir = '/Users/soyeonmin/Documents/Alexa/teach/'
DATA_FOLDER='/Volumes/untitled2/teach_download/'

import sys
sys.path.append(teach_dir +'src/teach/inference/FILM_refactor_april1/models/instructions_processed_LP/')
from init_pickup_mask_helper import init_pickup_mask_helper
import cv2
import pickle
#Repeat for all edh_instances and save pickedup next scene mask, image mask, and last image
import ipdb

from glob import glob
import json
import numpy as np
all_valid_seen_edh_files = glob(DATA_FOLDER + "edh_instances/valid_seen/*")
os.makedirs("pick_up_mask/")
os.makedirs("pick_up_mask/True")
os.makedirs("pick_up_mask/False")

nones = []
last_pickup = []
last_place = []


for ei, edh_file in enumerate(all_valid_seen_edh_files): 
    if ei %10 ==0:
        print("ei is ", ei)
    edh_instance = json.load(open(edh_file, 'r'))
    
    #edh_instance = pickle.load(open(teach_dir +'src/teach/inference/FILM_refactor_april1/models/instructions_processed_LP/'+'edh_instance.p', 'rb'))

    assert len(edh_instance['driver_action_history']) == len(edh_instance['driver_image_history'])
    #assert not(edh_instance['driver_action_history'][-1]['action_name'] in ['Pickup', 'Place'])
    if edh_instance['driver_action_history'][-1]['action_name'] == "Pickup":
        last_pickup.append(edh_instance['instance_id'])

    if edh_instance['driver_action_history'][-1]['action_name'] == "Place":
        last_place.append(edh_instance['instance_id'])

    if not(edh_instance['driver_action_history'][-1]['action_name'] in ['Pickup', 'Place']):

        init_helper = init_pickup_mask_helper()
        image_dir = DATA_FOLDER + "images/valid_seen/" + edh_instance['instance_id'].split('.')[0] 

        #IMPORTANT: each image history is before taking that option
        #ASSERT THAT the last action is not pickup

        #Get the last picked up action instance that succeded
        last_picked_i = -1
        for i, action_dict in enumerate(edh_instance['driver_action_history']):
            if action_dict['action_name'] == "Pickup":
                prev_rgb = cv2.resize(cv2.imread(os.path.join(image_dir, edh_instance['driver_image_history'][i])), (300,300))
                img =  cv2.resize(cv2.imread(os.path.join(image_dir,edh_instance['driver_image_history'][i+1])), (300,300))
                assert not(img is  None) and not(prev_rgb is None)
                if init_helper._get_approximate_success(prev_rgb, img, "PickupObject"):
                    last_picked_i = i

        last_put = -1
        for i, action_dict in enumerate(edh_instance['driver_action_history']):
            if action_dict['action_name'] == "Place":
                prev_rgb = cv2.resize(cv2.imread(os.path.join(image_dir, edh_instance['driver_image_history'][i])), (300,300))
                img =  cv2.resize(cv2.imread(os.path.join(image_dir,edh_instance['driver_image_history'][i+1])), (300,300))
                assert not(img is  None) and not(prev_rgb is None)
                if init_helper._get_approximate_success(prev_rgb, img, "PutObject"):
                    last_put = i


        case = None
        if last_picked_i ==-1:
            case = False
        elif last_put > last_picked_i :
            case=False
        else:
            case = True
            #get the largest one
            goal_name = edh_instance['driver_action_history'][last_picked_i]['oid'].split('|')[0]
            rgb = cv2.resize(cv2.imread(os.path.join(image_dir, edh_instance['driver_image_history'][last_picked_i+1])), (300,300))
            rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            assert not(rgb is None)
            picked_up = True
            init_helper.get_sem_pred(rgb, goal_name)
            picked_up_mask = init_helper.sem_seg_get_instance_mask_from_obj_type_largest_only(goal_name)
            #if picked_up_mask is None:
            #    ipdb.set_trace()
            if picked_up_mask is None:
                print(edh_instance['instance_id'], " Gave None!")
                nones.append(edh_instance['instance_id'])
            #else:
            #    ipdb.set_trace()



        #Save
        if case == True:
            save_dir = "pick_up_mask/True/" + edh_instance['instance_id']
            os.makedirs("pick_up_mask/True/" + edh_instance['instance_id'])
        else:
            save_dir = "pick_up_mask/False/" + edh_instance['instance_id']
            os.makedirs("pick_up_mask/False/" + edh_instance['instance_id'])

        #Last image
        cv2.imwrite(save_dir + "/last_image.png", cv2.resize(cv2.imread(os.path.join(image_dir, edh_instance['driver_image_history'][-1])) , (300,300)))
        if case == True:
            if not(picked_up_mask) is None:
                p = np.zeros((300,300, 3)); p[:, :, 0] = picked_up_mask; p[:, :, 1] = picked_up_mask; p[:, :, 2] = picked_up_mask
                picked_up_mask = p.astype('uint8') * 255
                #ipdb.set_trace()
                cv2.imwrite(save_dir + "/picked_up_mask.png", picked_up_mask)
            cv2.imwrite(save_dir + "/picked_up_img.png", rgb)



ipdb.set_trace()


# #Get all the edh instances
# edh_instance_by_type = {t:[] for t in task_types}
# for edh_files in tqdm.tqdm(all_valid_edh_files):
#     edh_instance = json.load(edh_files.open("r"))
#     tmp = _get_task_type_and_params(edh_instance)
#     task_type, obj_count, obj_target, parent_target = tmp["task_type"], tmp["obj_count"], tmp["obj_target"], tmp["parent_target"]
#     edh_instance_by_type[task_type].append(edh_instance)

# weird = []
# for edh in edh_instance_by_type["Water Plant"]:
#     objs =  list(edh['state_changes']['objects'].keys())
#     objs = [obj.split('|')[0] for obj in objs]
#     if objs!= ['HousePlant']:
#         weird.append(edh) #0

# Type "Coffee"
# #Get the num state changes for Mug
# #num state changes for Coffeemachine
# #for Faucet

# #ids that succeded with nothing
# succeeded_ids = ['3c82a4284ec95f08_f378.edh9', '38e166ac4f59b7a7_cf45.edh4', '413f4dccd527e74b_892a.edh1', 'de134e044f715c4f_2817.edh0', '86e4a253884d4582_acc0.edh6', 'd8ae4bdf08ad1cab_4dbf.edh1', '66a2b3be9faab959_4530.edh2', '8f94869f774ba6af_417e.edh1', 'e8b90ae5532bbf3f_1c7b.edh9', '66a2b3be9faab959_4530.edh0', '1fe53640e664654a_f63d.edh1', 'd6adf9446daff5a0_d60b.edh5', '40d8beb71dbef7cb_f241.edh2', 'b32df1dd6e121adb_0c9f.edh0', 'f80a282a8367bd72_a4e7.edh0', 'ab92949b1ed12dff_ad77.edh0', '277a21f6f0bb6150_5a97.edh7', '61d740111c7f4a7e_7c5f.edh3', '58e15b751d6b9c3a_692f.edh1', 'cab79739695381c7_98ee.edh0', 'cf51a495607a66dc_9bf0.edh4', '61d740111c7f4a7e_7c5f.edh2', '99193efbc05bc4b3_215f.edh0', '12541129ceafe873_d2d8.edh1', 'e2d23874ea3c73b1_2f3e.edh0', '45d5698613e90f24_1b91.edh4', '7f8000bdc9a6df46_e4c4.edh1', '9469c520db9f0cfd_87fa.edh0', 'a157945e35db4825_7aa8.edh0', 'f9145b3a51dbaf55_1e75.edh1', 'b0aea68d80b2b19d_d5c9.edh0', '12541129ceafe873_d2d8.edh0', '3c82a4284ec95f08_f378.edh0', '1c2df1e8ff105c52_6ac7.edh9', '1f3729c7d90e4976_6ac9.edh1', '277a21f6f0bb6150_5a97.edh5', '4304ea1a0444cea2_7156.edh0', '5ad807df9d1f9966_aab9.edh1', '3974a05de94d42b5_c851.edh0', '028c162b607cc751_ac5a.edh0', 'fca2df65168229c0_bb02.edh0', '193f64286be0ba2c_0e02.edh1', '5ddceaefa7a8af94_5eda.edh0', 'be351712c44aeb66_b6d8.edh2', 'bf824d471acac29a_6dbb.edh1']

# succeeded_ids =set(succeeded_ids)
# valid_seen_edh_files = (Path(DATA_FOLDER)/"edh_instances/valid_seen").glob("*.json")

# edh_instances_succeeded_with_nothing = []
# task_types_succeed_with_nothing = {t:0 for t in task_types}
# for edh_file in valid_seen_edh_files:
#     edh_instance = json.load(edh_file.open("r"))
#     tmp = _get_task_type_and_params(edh_instance)
#     task_type, obj_count, obj_target, parent_target = tmp["task_type"], tmp["obj_count"], tmp["obj_target"], tmp["parent_target"]
    
#     if edh_instance['instance_id'] in succeeded_ids:
#         edh_instances_succeeded_with_nothing.append(edh_instance)
#         task_types_succeed_with_nothing[task_type] +=1


