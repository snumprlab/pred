import openai
import os
import json
import random
from FILM_model.models.instructions_processed_LP.get_arguments_from_bert import GetArguments
ArgPredictor = GetArguments()
from FILM_model.models.segmentation import alfworld_constants
from FILM_model.models.instructions_processed_LP.object_state import ObjectState
import copy

#############################################
#############################################

############################################

######## argument ####
tfd = True
data_dir = "/media/user/data/TEACH/project/TEACH_FILM_for_jhc/teach-dataset"
splits = ["valid_seen", "valid_unseen"]
# splits = ["valid_seen"]
# split = "train"   # choices=["train", "valid_seen", "valid_unseen", "test_seen", "test_unseen"],
Task_name = 'Plate Of Toast'  #'Coffee' : #'Boil X' :#'Water Plant':
dn = 'template+task_relevant_context'
dn = 'template'
gpt_subgoal_flag = False
extracted_info_flag = False

########################
Task_names = ['Coffee','Water Plant','Boil X', 'Clean All X', 'N Cooked Slices Of X In Y', 'N Slices Of X In Y', 'Plate Of Toast', 'Put All X In One Y', 'Put All X On Y', 'Salad', 'Sandwich']
# Task_names = ['Coffee','Water Plant','Boil X','N Slices Of X In Y','Put All X In One Y', 'Put All X On Y']
# Task_names = ['Put All X In One Y', 'Put All X On Y', 'Salad', 'Sandwich']
# Task_names = ['Coffee']
########################
gt_task_names = dict()

task_correct_num_total = 0
ep_num_total = 0


all_results = dict()





large = alfworld_constants.STATIC_RECEPTACLES
large_objects2idx = {k:i for i, k in enumerate(large)}

small = alfworld_constants.OBJECTS_DETECTOR
small_objects2idx = {k:i for i, k in enumerate(small)}
INRECEP = ['Fridge', 'Cabinet', 'Microwave', 'Drawer', 'LaundryHamper']


gpt_result_game_id_all = dict()
for split in splits:
    accuracy_result = dict()
    
    gpt_result_split = dict()
    for Task_name in Task_names:

        task_correct_num = 0
        print(split)
        print(Task_name)
        ep_num = 0
        
        gpt_result = dict()
        ########### Data loader  ##########################
        if tfd:
            edh_instance_files = [
                os.path.join(data_dir, "tfd_instances",split, f)
                for f in os.listdir(os.path.join(data_dir, "tfd_instances", split))
                # if f not in finished_edh_instance_files
            ]
        else:
            edh_instance_files = [
                os.path.join(data_dir, "edh_instances", split, f)
                for f in os.listdir(os.path.join(data_dir, "edh_instances", split))
                # if f not in finished_edh_instance_files
            ]
        ##################################################    
        for tfd_instance_dir in edh_instance_files:
            try:
                with open(tfd_instance_dir) as json_file:
                    json_data = json.load(json_file)
                    
                if json_data['game']['tasks'][0]['task_name'] != Task_name: #'Coffee' : #'Boil X' :#'Water Plant': 
                    continue

                game_id = json_data['game_id']
                dialog = json_data['dialog_history']
                task_name = json_data['game']['tasks'][0]['task_name']

                task_type, obj_count, obj_target, parent_target = ArgPredictor.get_pred(json_data)
                
                edh_instance = json_data



                ###LANGUAGE PLANNING######################################################################################################################################
                if obj_target == "Bottle":
                    obj_target = "Glassbottle"

                #If you ever get an object target or parent target not in , just replace them with "Mug"
                if not(obj_target in small_objects2idx) and not(obj_target in large_objects2idx):
                    if not(obj_target  in ['Drinkware', 'Dishware', 'SportsEquipment', 'Condiments', 'Silverware', 'Cookware', 'Fruit', 'SmallHandheldObjects', 'Tableware'] ):
                        obj_target = "Mug"

                if not(parent_target in small_objects2idx) and not(parent_target in large_objects2idx):
                    if parent_target not in ['Tables', 'Chairs', 'Furniture']:
                        parent_target = "Desk"

                real_obj_target = copy.deepcopy(obj_target); real_parent_target = copy.deepcopy(parent_target)
                if obj_target in ['Drinkware', 'Dishware', 'SportsEquipment', 'Condiments', 'Silverware', 'Cookware', 'Fruit', 'SmallHandheldObjects', 'Tableware']:
                    if obj_target == 'Drinkware':
                        obj_target = "Mug"
                    elif obj_target == 'Dishware':
                        obj_target = "Plate"
                    elif obj_target == 'SportsEquipment':
                        obj_target = 'BaseballBat'
                    elif obj_target == 'Condiments':
                        obj_target = 'SaltShaker'
                    elif obj_target == "Silverware":
                        obj_target = "Fork"
                    elif obj_target == "Cookware":
                        obj_target = 'Kettle'
                    elif obj_target == 'Fruit':
                        obj_target = 'Apple'
                    elif obj_target =='Tableware':
                        obj_target = "Plate"
                    elif obj_target == 'SmallHandheldObjects':
                        obj_target = "RemoteControl"


                if parent_target in ["Tables", "Chairs", "Furniture"]:
                    if parent_target == "Tables":
                        parent_target = "DiningTable"
                    elif parent_target == "Chairs":
                        parent_target = 'ArmChair'
                    elif parent_target == "Furniture":
                        parent_target = "Sofa"                

                intermediate_obj_gpt = None
                if gpt_subgoal_flag:
                    # gpt direct subgoal new
                    with open('/media/user/data/TEACH/project/TEACH_FILM_for_jhc/gpt_direct_subgoal_new/gpt_result_all_object_revised.json') as f:
                        gpt_subgoal = json.load(f)

                    game_id = edh_instance['game_id']
                    if game_id in gpt_subgoal:
                        list_of_actions_gpt = gpt_subgoal[game_id]
                        if task_type == "Water Plant" :
                            # change intermediate_obj
                            for i in list_of_actions_gpt:
                                if i[0] in ['Mug', 'Cup', 'Bowl']:
                                    intermediate_obj_gpt = i[0]
                                    break            
                
                prev_actions = edh_instance['driver_action_history']
                obj_state = ObjectState(edh_instance['dialog_history'], prev_actions, task_type, obj_count=obj_count, obj_target=obj_target, parent_target=parent_target, intermediate_obj_gpt = intermediate_obj_gpt)
                # obj_state = ObjectState(edh_instance['dialog_history'], prev_actions, task_type, obj_count=obj_count, obj_target=obj_target, parent_target=parent_target)
                sliced = obj_state.slice_needs_to_happen
                #sliced= False
                #list_of_actions, categories_in_inst, second_object, caution_pointers = get_list_of_highlevel_actions(task_type, obj_count, obj_target, parent_target, sliced)
                #####################
                list_of_actions = obj_state.future_list_of_highlevel_actions 
                categories_in_inst = obj_state.categories_in_inst
                second_object = obj_state.second_object
                caution_pointers = obj_state.caution_pointers

            
                # gpt subgoal (direct)
                ###################################################################################
                if gpt_subgoal_flag:
                    with open('FILM_model/classes.txt', 'r') as file:
                        content = file.read().strip()  # 파일 내용 읽기 및 공백 제거
                        word_list = content.split(', ')			
                    ### list of actions ###
                    # # with open('FILM_model/gpt_result_valid_seen_con2subgoal.json') as f:
                    # with open('FILM_model/gpt_result_valid_direct_1211.json') as f:
                    # 	gpt_subgoal = json.load(f)
        
                    # # Retrieved subgoal
                    # with open('FILM_model/retrieved_subgoals.json') as f:
                    # 	gpt_subgoal = json.load(f)	# retrieved subgoal
        
                    # gpt direct subgoal new
                    with open('/media/user/data/TEACH/project/TEACH_FILM_for_jhc/gpt_direct_subgoal_new/gpt_result_all_object_revised.json') as f:
                        gpt_subgoal = json.load(f)
            
                    game_id = edh_instance['game_id']
                    if game_id in gpt_subgoal:
                        list_of_actions = gpt_subgoal[game_id]
                        
                        # invalid action name check
                        for i in list_of_actions:
                            if i[1] not in ["PickupObject", "PutObject", "OpenObject", "CloseObject", "ToggleObjectOn", "ToggleObjectOff", "PourObject", "SliceObject"] \
                            or i[0] not in word_list:
                                list_of_actions = [[obj_target, 'PickupObject']]
                        for i in list_of_actions:
                            if i[0] == 'Sink':
                                i[0] = 'SinkBasin'

                        ###########################################################################
                        # SinkBasin ToggleOff
                        list_of_actions_ = []
                        faucettoggledon = True
                        faucetshouldbetoggledon = False
                        muginsinkbasin = False
                        for idx, i in enumerate(list_of_actions):
                            if (i[0] == "Faucet") and (list_of_actions[idx-1][0] not in ["SinkBasin", "Faucet"]):
                                list_of_actions_.append(["SinkBasin", "ToggleObjectOff"])
                                faucettoggledon = False

                            ## 물 차있고 안 차있고 고려해서 sinkbasin toggleoff 추가 / 나중에 물 차있는지 attribute로 판단하면 아래 지우면 됨
                            elif (i[0] == 'SinkBasin' and i[1] == "PutObject") :
                                if faucettoggledon :
                                    list_of_actions_.append(['SinkBasin', 'ToggleObjectOff'])
                                    faucetshouldbetoggledon = True
                                muginsinkbasin = True

                            if i[1] == "PickupObject" and muginsinkbasin:
                                if (list_of_actions[idx-1][0] =="Faucet" and list_of_actions[idx-1][1] == "ToggleObjectOn"):
                                    list_of_actions_.append(["Faucet", "ToggleObjectOff"])
                                    muginsinkbasin = False
                                    faucettoggledon = False
                                    faucetshouldbetoggledon = True          
                                elif faucetshouldbetoggledon:
                                    list_of_actions_.append(["Faucet", "ToggleObjectOn"])
                                    list_of_actions_.append(["Faucet", "ToggleObjectOff"])
                                    faucettoggledon = False
                                    muginsinkbasin = False


                            if i[0] == "Faucet" and i[1] == "ToggleObjectOn":
                                faucettoggledon = True; faucetshouldbetoggledon = False
                            if i[0] in ["Faucet", "SinkBasin"] and i[1] == "ToggleObjectOff":
                                faucettoggledon = False
                            list_of_actions_.append(i)
                        list_of_actions = list_of_actions_
                        ###########################################################################


                    else:
                        list_of_actions = [[obj_target, 'PickupObject']]
                    
                game_id = edh_instance['game_id']
                ###################################################################################
                if extracted_info_flag:
                    with open('FILM_model/condition_revised/extracted_info_gpt4.json') as f:
                        ext_info = json.load(f)
                    new_actions = []
                    if task_type == "N Slices Of X In Y" and game_id in ext_info:
                        print(ext_info[game_id])
                        location_flag = False
                        sliced_flag = False
                        pickup_put = False
                        target_object = None
                        location_open = []
                        for i in list_of_actions:
                            if i[0] in ext_info[game_id] and i[1] in ["PickupObject", "SliceObject", "PutObject"] and i[0] not in location_open:
                                if "location" in ext_info[game_id][i[0]]:
                                    if ext_info[game_id][i[0]]["location"] in INRECEP:
                                        new_actions.append([ext_info[game_id][i[0]]["location"], "OpenObject"])
                                        location_open.append(i[0])
                            if i[1] == "SliceObject":
                                sliced_flag = True
                                target_object = i[0]
                            # if sliced_flag and new_actions[-1][0] == target_object and new_actions[-1][1] == "PickupObject":

                            if target_object in ext_info[game_id]:
                                if "quantity_of_slices" in ext_info[game_id][target_object]:
                                    if isinstance(ext_info[game_id][target_object]["quantity_of_slices"], int):
                                        for _ in range(int(ext_info[game_id][target_object]["quantity_of_slices"])-1):
                                            new_actions.append(list_of_actions[3])
                                            new_actions.append(list_of_actions[4])
                            new_actions.append(i)
                        list_of_actions = new_actions
            
                    elif task_type == "N Cooked Slices Of X In Y" and game_id in ext_info:
                        print(ext_info[game_id])
                        location_flag = False
                        sliced_flag = False
                        microwave_open = False
                        target_object = None
                        location_open = []
                        for i in list_of_actions:
                            if i[0] in ext_info[game_id] and i[1] in ["PickupObject", "SliceObject", "PutObject"] and i[0] not in location_open:
                                if "location" in ext_info[game_id][i[0]]:
                                    if ext_info[game_id][i[0]]["location"] in INRECEP:
                                        new_actions.append([ext_info[game_id][i[0]]["location"], "OpenObject"])
                                        location_open.append(i[0])
                            if i[1] == "SliceObject":
                                sliced_flag = True
                                target_object = i[0]
                            # if sliced_flag and new_actions[-1][0] == target_object and new_actions[-1][1] == "PickupObject":
                            if i == ('Microwave', 'OpenObject') and not microwave_open:
                                new_actions.append(list_of_actions[9])
                                if target_object in ext_info[game_id]:
                                    if "quantity_of_slices" in ext_info[game_id][target_object]:
                                        # pickup = new_actions[-1]
                                        # put = i
                                        for _ in range(int(ext_info[game_id][target_object]["quantity_of_slices"])-1):
                                            new_actions.append(list_of_actions[3])
                                            new_actions.append(list_of_actions[9])
                                new_actions.append((list_of_actions[9][0], 'PickupObject'))
                                microwave_open = True
                            new_actions.append(i)
                        new_actions = new_actions[:-3]
                        list_of_actions = new_actions				

                    elif task_type == "Put All X In One Y" and game_id in ext_info:
                        print(ext_info[game_id])
                        microwave_open = False
                        target_object = None
                        location_open = dict()
                        for i in list_of_actions:
                            if i[0] in ext_info[game_id] and i[1] in ["PickupObject"]:
                                target_object = i[0]
                                if i[0] not in location_open:
                                    if "location" in ext_info[game_id][i[0]]:
                                        location_open[i[0]] = []
                                        if isinstance(ext_info[game_id][i[0]]["location"], list):
                                            for loc in ext_info[game_id][i[0]]["location"]:
                                                
                                                if loc in INRECEP:
                                                    # new_actions.append([ext_info[game_id][i[0]]["location"], "OpenObject"])
                                                    location_open[i[0]].append(loc)
                                                location_open[i[0]] = list(set(location_open[i[0]]))
                                                if len(location_open[i[0]]) > 0:
                                                    new_actions.append([location_open[i[0]].pop(), "OpenObject"])
                                        elif ext_info[game_id][i[0]]["location"] in INRECEP:
                                            new_actions.append([ext_info[game_id][i[0]]["location"], "OpenObject"])
                                            
                            new_actions.append(i)
                        if target_object in ext_info[game_id]:
                            if "quantity" in ext_info[game_id][target_object]:
                                if isinstance(ext_info[game_id][target_object]['quantity'], int):
                                    for _ in range(int(ext_info[game_id][target_object]["quantity"])-1):
                                        if len(location_open[list_of_actions[0][0]]) > 0:
                                            new_actions.append([list_of_actions[0][0].pop(), "OpenObject"])
                                        new_actions.append(list_of_actions[0])
                                        new_actions.append(list_of_actions[1])
                        list_of_actions = new_actions

                    elif task_type == "Put All X On Y" and game_id in ext_info:
                        print(ext_info[game_id])
                        location_flag = False
                        sliced_flag = False
                        microwave_open = False
                        target_object = None
                        location_open = dict()
                        set_X = set()
                        for i in list_of_actions:
                            if i[0] in ext_info[game_id] and i[1] in ["PickupObject"]:
                                target_object = i[0]
                                if i[0] not in location_open:
                                    if "location" in ext_info[game_id][i[0]]:
                                        location_open[i[0]] = []
                                        if isinstance(ext_info[game_id][i[0]]["location"], list):
                                            for loc in ext_info[game_id][i[0]]["location"]:
                                                
                                                if loc in INRECEP:
                                                    # new_actions.append([ext_info[game_id][i[0]]["location"], "OpenObject"])
                                                    location_open[i[0]].append(loc)
                                                location_open[i[0]] = list(set(location_open[i[0]]))
                                                if len(location_open[i[0]]) > 0:
                                                    new_actions.append([location_open[i[0]].pop(), "OpenObject"])
                                        elif ext_info[game_id][i[0]]["location"] in INRECEP:
                                            new_actions.append([ext_info[game_id][i[0]]["location"], "OpenObject"])
                                            
                            new_actions.append(i)
                        if target_object in ext_info[game_id]:
                            if "quantity" in ext_info[game_id][target_object]:
                                if isinstance(ext_info[game_id][target_object]['quantity'], int):
                                    for _ in range(int(ext_info[game_id][target_object]["quantity"])-1):
                                        if len(location_open[list_of_actions[0][0]]) > 0:
                                            new_actions.append([list_of_actions[0][0].pop(), "OpenObject"])
                                        new_actions.append(list_of_actions[0])
                                        new_actions.append(list_of_actions[1])
                        list_of_actions = new_actions

                    elif task_type == "Coffee" and game_id in ext_info:
                        print(ext_info[game_id])
                        if "Mug" in ext_info[game_id]:
                            if "location" in ext_info[game_id]["Mug"]:
                                if ext_info[game_id]["Mug"]["location"] in INRECEP:
                                    new_actions.append((ext_info[game_id]["Mug"]["location"], "OpenObject"))
                        for i in list_of_actions:
                            new_actions.append(i)
                        list_of_actions = new_actions

                    elif task_type == "Water Plant" and game_id in ext_info:
                        print(ext_info[game_id])
                        opened = False
                        if list_of_actions[0][0] in ext_info[game_id]:
                            if "location" in ext_info[game_id][list_of_actions[0][0]]:
                                if ext_info[game_id][list_of_actions[0][0]]["location"] in INRECEP:
                                    new_actions.append((ext_info[game_id][list_of_actions[0][0]]["location"], "OpenObject"))
                        for i in list_of_actions:
                            new_actions.append(i)
                        list_of_actions = new_actions

                    elif task_type == "Boil X" and game_id in ext_info:
                        print(ext_info[game_id])
                        if "Mug" in ext_info[game_id]:
                            if "location" in ext_info[game_id]["Mug"]:
                                if ext_info[game_id]["Mug"]["location"] in INRECEP:
                                    new_actions.append((ext_info[game_id]["Mug"]["location"], "OpenObject"))
                        for i in list_of_actions:
                            new_actions.append(i)
                        list_of_actions = new_actions                


                gpt_result[game_id] = list_of_actions
                gpt_result_game_id_all[game_id] = list_of_actions

            except Exception as e: print(e)

        # try:
        #     with open('/media/user/data/TEACH/project/TEACH_FILM_for_jhc/template_subgoal/template_' + Task_name + '_' + split + '_' + dn + '.json') as f:
        #         result = json.load(f)
        #     for i in list(gpt_result.keys()):
        #         result[i] = gpt_result[i]                
        #     with open('/media/user/data/TEACH/project/TEACH_FILM_for_jhc/template_subgoal/template_' + Task_name + '_' + split + '_' + dn + '.json', 'w') as f:
        #         json.dump(result,f,indent=4)
        # except:
        #     with open('/media/user/data/TEACH/project/TEACH_FILM_for_jhc/template_subgoal/template_' + Task_name + '_' + split + '_' + dn + '.json', 'w') as f:
        #         json.dump(gpt_result,f,indent=4)                    
            
        print()
        print(split)
        print(Task_name)

        print("=============================================\n")
        
        # accuracy_result[Task_name] = str(round(task_correct_num/ep_num*100,2))+'  '+str(task_correct_num)+'/'+str(ep_num)

    # all_results[split] = accuracy_result
    
try:
    with open('/media/user/data/TEACH/project/TEACH_FILM_for_jhc/template_subgoal/'+dn+'_all.json') as f:
        result = json.load(f)
    for i in list(gpt_result_game_id_all.keys()):
        result[i] = gpt_result_game_id_all[i]
    with open('/media/user/data/TEACH/project/TEACH_FILM_for_jhc/template_subgoal/'+dn+'_all.json','w') as f:
        json.dump(result,f,indent=4)        
except:
    with open('/media/user/data/TEACH/project/TEACH_FILM_for_jhc/template_subgoal/'+dn+'_all.json', 'w') as f:
        json.dump(gpt_result_game_id_all, f, indent = 4)
    
    
    
    
# print()
# for split in list(all_results.keys()):
#     print("=============================================")
#     print(split)
#     for t in list(all_results[split].keys()):
#         print(t)
#         print(all_results[split][t])
#         print()
# print('prediction accuracy_condition: ', round(task_correct_num_total/ep_num_total*100,2))