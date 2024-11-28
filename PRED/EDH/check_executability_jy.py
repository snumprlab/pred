import tqdm
import pickle
import json
import os
import FILM_model.alfred_utils.gen.constants as constants

########################################### #Executability ##############################

# with open('gpt_result_direct_subgoal/gpt_result_valid_direct.json') as f:
#     pred = json.load(f)
# with open('gpt_result_condition_to_compact_subgoal/gpt_result_Coffee_train_direct.json') as f:
#     pred = json.load(f)




########################################################################
# Arguments
########################################################################
splits = ['train']
dn = "compact_detailed"
# Task_names = ['Coffee','Water Plant','Boil X', 'Clean All X', 'N Cooked Slices Of X In Y', 'N Slices Of X In Y', 'Plate Of Toast', 'Put All X In One Y', 'Put All X On Y', 'Salad', 'Sandwich']
Task_names = ['Water Plant']
tfd = True
data_dir = "/media/user/data/TEACH/project/TEACH_FILM_for_jhc/teach-dataset"
subgoal_file_path = "검사하려는 파일 경로"
########################################################################
########################################################################




INRECEP = ['Fridge', 'Cabinet', 'Microwave', 'Drawer', 'LaundryHamper']
# gt subgoal
gt_path = 'gt_condition_valid.json'
gt_path = 'gt_condition_train.json'
with open(gt_path) as f:
    gt = json.load(f)

def openlist(obj, Open_Obj):
    f = False
    if obj in Open_Obj:
        if obj in INRECEP:
                f = True
    return f

passed = dict()


def reorder(action):
    act = action[1]
    obj = action[0]
    if act == "Pickup": act = "PickupObject"
    if act == "Place": act = "PutObject"
    if act == "Open": act = "OpenObject"
    if act == "Close": act = "CloseObject"
    if act == "Slice": act = "SliceObject"
    if act == "ToggleOn": act = "ToggleObjectOn"
    if act == "ToggleOff": act = "ToggleObjectOff"
    if act == "Pour": act = "PourObject"
    return [obj, act]
class_correct = dict()

for split in splits:
    correct_num  = 0
    total_num    = 0

    action_sequence_correct_Num = 0
    Cls_correct_Num = 0
    Cls_Location_correct_Num  = 0

    pourfailed = []
    failed = []
    class_correct[split] = []
    alreadyfilled = []
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

    gpt_result_split = dict()
    print("====================================================================================================")
    print(split)
    for Task_name in Task_names:
            
        task_correct_num = 0
        
        print(Task_name)
        ep_num = 0
        
        gpt_result = dict()

        for tfd_instance_dir in edh_instance_files:
            #### load data
            with open(tfd_instance_dir) as json_file:
                json_data = json.load(json_file)
                
            if json_data['game']['tasks'][0]['task_name'] != Task_name: #'Coffee' : #'Boil X' :#'Water Plant': 
                continue
            
            game_id = json_data['game_id']
            dialog = json_data['dialog_history']
            task_name = json_data['game']['tasks'][0]['task_name']
        

            traj_data = json_data
            instruction = dialog


            LLM_result = dict()
            LLM_result['seen_objs'] = list()
            #############################################################################################
            #############################################################################################            
            # 검사하려는 subgoal 불러오는 부분
            with open(subgoal_file_path) as f:
                pred = json.load(f)
            if game_id not in pred:
                continue
            subgoal = pred[game_id]
            for i in range(len(subgoal)):
                subgoal[i] = reorder(subgoal[i])
            #############################################################################################
            #############################################################################################

            for i in subgoal:
                LLM_result['seen_objs'].append(i[0])

            LLM_result['seen_objs'] = list(set(LLM_result['seen_objs']))

            LLM_result['triplet'] = list()
            for i in subgoal:
                LLM_result['triplet'].append([i[1],i[0], "recep"])
            
            initiallyopened = []
            # append env obj
            env_obj_list = []
            for env_obj in traj_data['game']['tasks'][0]['episodes'][0]['initial_state']['objects'] :
                env_obj_list.append(env_obj['objectType'])
                if env_obj['isOpen']:
                    initiallyopened.append(env_obj['objectType'])



            ### init parameter
            last_cls_condition =  True
            sliced = False

            total_num+=1
            
            
            ################## Check cls in env and instance num
            for obj_list in LLM_result['seen_objs'] :
                if not 'Sliced' in obj_list:
                    last_cls_condition *= (obj_list in  env_obj_list)
                    if obj_list in  env_obj_list :
                        env_obj_list.remove(obj_list)
                    else :
                        print(obj_list +" not in Env")
                else:
                    sliced = True

            if last_cls_condition :
                Cls_correct_Num +=1
                class_correct[split].append(game_id)


##############################################
            # predicted subgoal에서 mug가 어떤 recep 안에/위에 있는지 뱉은 게 있으면 진짜 있는지 없는지 확인하고, 없으면 fail 처리
            Location_condition = False
            actionset = set()
            filled_objs = dict()
            insinkbasin = []
            opened=[]
            ToggleOn_obj = []
            act_flag = False
            for i in subgoal:
                obj = i[0]
                act = i[1]
                if act == "OpenObject":
                    opened.append(obj)
                    act_flag = True
                if act == "CloseObject" and obj in opened:
                    opened.remove(obj)
                    act_flag = True
                if act in ["PickupObject", "SliceObject"]:
                    act_flag = True
                    for env_obj in traj_data['game']['tasks'][0]['episodes'][0]['initial_state']['objects'] :
                        if obj == env_obj['objectType']:
                            if env_obj['parentReceptacles'] is not None :
                                for p in env_obj['parentReceptacles']:
                                    if p.split('|')[0] not in INRECEP:
                                        Location_condition = True
                                        break
                                    if p.split('|')[0] in opened:
                                        Location_condition = True
                                        break
                                    if p.split('|')[0] in initiallyopened:
                                        Location_condition = True
                                        break
                            else:
                                print("what happend")
                if not act_flag:
                    Location_condition = True
            if Location_condition :
                Cls_Location_correct_Num +=1
                
            for env_obj in traj_data['game']['tasks'][0]['episodes'][0]['initial_state']['objects'] :
                if env_obj['isFilledWithLiquid']:
                    filled_objs[env_obj['objectType']] = True
                if env_obj['parentReceptacles'] is not None:
                    for p in env_obj['parentReceptacles']:
                        if p.split('|')[0] == 'SinkBasin' or p.split('|')[0] == 'Sink':
                            insinkbasin.append(env_obj['objectType'])
                if env_obj['isToggled']:
                    ToggleOn_obj.append(env_obj['objectType'])
            
            for objinsink in insinkbasin:
                if "Faucet" in ToggleOn_obj:
                    filled_objs[objinsink] = True
            ############################################################
            ### agent init state
            agent_state ={
                "Pickable"     : True,
                "Putable"      : False,
                "Openable"     : True,
                "Closable"     : False,
                "Sliceable"    : False,
                "PickSlicable" : False,
                "Toggleable"   : True,
                "Pourable"     : False
            }
            low_actions =[]
            for i in subgoal:
                obj = i[0]
                act = i[1]
                if obj == 'Sink':
                    obj = 'SinkBasin'
                low_actions.append([act, obj])

            ######################### pick action in recep executability only
            action_sequnce_condition = True

            ######################### action sequnce executability

            Pickup_obj = '0'
            Open_Obj = []
            Sliced_obj ='0'
            for idx, low_action in enumerate(low_actions):
                ##------- Pickup Obj on recep
                if low_action[0] == 'PickupObject' and agent_state["Pickable"] \
                    and "Sliced" not in low_action[1] :
                    # and (low_action[1] in  constants.NON_RECEPTACLES or low_action[1] in  constants.MOVABLE_RECEPTACLES ):
                
                    agent_state["Pickable" ] = False
                    agent_state["Putable" ] = True
                    Pickup_obj = low_action[1]
                    if low_action[1] == 'Knife' or low_action[1] == 'ButterKnife' :
                            agent_state['Sliceable'] = True


                ##-------- Pickup SlicedObj on recep
                elif low_action[0] == 'PickupObject' and agent_state["Pickable"] \
                    and "Sliced" in low_action[1] and agent_state["PickSlicable"]\
                    and Sliced_obj in low_action[1]:

                    agent_state["Pickable" ] = False
                    agent_state["Putable" ] = True
                    Pickup_obj = low_action[1]

                ##-------- Put on recep (TEACH)
                elif low_action[0] == 'PutObject' and low_action[1] not in INRECEP and agent_state['Putable']:
                        agent_state["Pickable" ] = True
                        agent_state["Putable" ] = False                    
                        Pickup_obj = '0'
                        if agent_state['Sliceable']  :
                            agent_state['Sliceable'] = False 

                ##-------- Put in recep (TEACH)
                elif low_action[0] == 'PutObject' and low_action[1] in INRECEP+['Safe'] and  agent_state['Putable']\
                    and (low_action[1] in Open_Obj  or  low_action[1] in initiallyopened):
                    
                    agent_state["Pickable" ] = True
                    agent_state["Putable" ] = False

                    Pickup_obj = '0'
                    if agent_state['Sliceable']  :
                        agent_state['Sliceable'] = False                    

                ##------- Slice obj	
                elif low_action[0] == 'SliceObject' and agent_state["Sliceable"] :
                            agent_state["PickSlicable" ] = True
                            Sliced_obj = low_action[1]


                ##--------- Open recep (a number of  recep)
                elif low_action[0] == 'OpenObject' \
                    and low_action[1]  in  INRECEP+['Safe']\
                    and not openlist(low_action[1], ['Fridge','Microwave', 'Safe']):
                    agent_state["Closable" ] = True
                    Open_Obj.append(low_action[1])

                ##--------- Open recep (0nly one recep)
                elif low_action[0] == 'OpenObject' \
                    and low_action[1]  in  INRECEP+['Safe']\
                    and (openlist(low_action[1],['Fridge','Microwave', 'Safe']) and low_action[1] not in Open_Obj): # and agent_state["Openable"] ## 물건이 하나인 경우 중복해서 open X 
                    agent_state["Closable" ] = True
                    # Open_Obj = low_action[1]
                    Open_Obj.append(low_action[1])

                ##--------- Close recep
                elif low_action[0] == 'CloseObject'\
                        and (low_action[1] in Open_Obj or low_action[1] in initiallyopened): #  and agent_state["Closable"]
                    agent_state["Closable" ] = False 

                    if low_action[1] in Open_Obj:
                        Open_Obj.remove(low_action[1])
                    elif low_action[1] in initiallyopened:
                        initiallyopened.remove(low_action[1])

                ##--------- ToggleOn
                elif low_action[0] == "ToggleObjectOn" :
                    # and (low_action[1] in constants.VAL_ACTION_OBJECTS['Toggleable'] or low_action[1] in ['Microwave','Faucet', 'CoffeeMachine'] ):
                    agent_state["Toggleable"] = False

                    if low_action[1] == "CoffeeMachine":
                        picked_ = ""; filled_ = False; putinsink = False; poured_ = False
                        for i, l in enumerate(low_actions):
                            if i < idx:
                                o = l[1]; a = l[0]
                                if a == "PickupObject":
                                    picked_ = o
                                if picked_ != "" and a == "PutObject" and o in ['Sink', 'SinkBasin']:
                                    putinsink = True
                                if putinsink and a == "ToggleObjectOn" and o == "Faucet":
                                    filled_ = True
                                if filled_ and a == "PourObject":
                                    poured_ = True
                                    filled_ = False
                        if filled_:
                            action_sequnce_condition = False
                            alreadyfilled.append(game_id)
                            

                ##--------- ToggleOFF
                elif low_action[0] == "ToggleObjectOff" :
                    # and (low_action[1] in constants.VAL_ACTION_OBJECTS['Toggleable'] or low_action[1] in ['Microwave','Faucet', 'CoffeeMachine'] ):
                    agent_state["Toggleable"] = True
                    
                ##--------- Pour
                elif low_action[0] == "PourObject":
                    if Pickup_obj not in filled_objs:
                        filled = False
                        picked = False
                        put = False
                        ToggleOn = False

                        for i, l in enumerate(low_actions):
                            if i<idx:
                                o = l[1]; a = l[0]
                                if a == "ToggleObjectOff" and o =="Faucet":
                                    ToggleOn = False
                                if a == "ToggleObjectOn" and o == "Faucet":
                                    ToggleOn = True
                                if (put  or Pickup_obj in insinkbasin) and a == "ToggleObjectOn" and o =="Faucet":
                                    filled = True
                                    ToggleOn = True
                                if  picked and a == "PutObject" and o in ['Sink', 'SinkBasin']:
                                    put = True
                                    if ToggleOn or "Faucet" in ToggleOn_obj:
                                        filled = True

                                if o == Pickup_obj and a == "PickupObject":
                                    picked = True
                        if not filled:
                            action_sequnce_condition = False
                            pourfailed.append(game_id)
                else :
                    action_sequnce_condition = False
                    
            if action_sequnce_condition :
                action_sequence_correct_Num +=1

            if 	last_cls_condition and Location_condition and action_sequnce_condition:
                correct_num +=1

            else:
                failed.append(game_id)

    
        print("")
        print("pourfailed:", pourfailed)
        print("alreadyfilled:", alreadyfilled)
        print("failed: ", failed)

        
        print("==================================================")
        print("ACC          :  "  +  str((correct_num/total_num)*100))
        print("Total Num    :  "  +  str(total_num))
        print("Correct Num  :  "  +  str(correct_num))
        print()
        print("Class predict failed                  :  "  +  str(total_num-Cls_correct_Num) )
        print("Invalid action with object's location :  "  +  str(total_num-Cls_Location_correct_Num) )
        print("Invalid action sequence               :  "  +  str(total_num-action_sequence_correct_Num) )
        print()
        print("pourfailed: ", len(pourfailed))
        print("failed: ", len(failed))
        print("====================================================================================================")

############################################################
