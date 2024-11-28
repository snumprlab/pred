import json
import os


INRECEP = ['Fridge', 'Cabinet', 'Microwave', 'Drawer', 'LaundryHamper']
FILLABLE_OBJS = ['Kettle', 'Glassbottle', "WineBottle", "WateringCan", 'Mug', 'Cup', 'Bowl']
PICKUPABLE_OBJS = ['Towel', 'HandTowel', 'SoapBar', 'ToiletPaper', 'SoapBottle', 'Candle', 'ScrubBrush', 'Plunger', 'Cloth', 'SprayBottle', 'Book', 'BasketBall', 'Pen', 'Pillow', 'Pencil', 'CellPhone', 'KeyChain', 'CreditCard', 'AlarmClock', 'CD', 'Laptop', 'Watch',  'WateringCan', 'Newspaper', 'RemoteControl', 'Statue', 'BaseballBat', 'TennisRacket', 'Mug', 'Apple', 'Lettuce', 'Bottle', 'Egg', 'Fork', 'WineBottle', 'Spatula', 'Bread', 'Tomato', 'Pan', 'Cup', 'Pot', 'SaltShaker', 'Potato', 'PepperShaker', 'ButterKnife', 'DishSponge', 'Spoon', 'Plate', 'Knife', 'Bowl', 'Vase', 'TissueBox', 'Boots', 'PaperTowelRoll', 'Ladle', 'Kettle', 'GarbageBag', 'TeddyBear', 'Dumbbell', 'AluminumFoil']



def high_to_low(act, highlevel_idx, ext_info, game_id, opened, sliced, shouldbeopened, list_of_actions, idx):
    Targets = list(ext_info[game_id].keys())
    new_actions = []
    shouldbeclosed = False
    
    if act[0] in ["Clean", "Fill"]:
        if act[1] in shouldbeopened:
            new_actions.append([shouldbeopened[act[1]], 'OpenObject', act, highlevel_idx])
            shouldbeclosed = True
            
        new_actions.append([act[1], 'PickupObject', act, highlevel_idx])
        if shouldbeclosed:
            new_actions.append([shouldbeopened[act[1]], 'CloseObject', act, highlevel_idx])
            shouldbeclosed = False
            del shouldbeopened[act[1]]
            opened.append(act[1])
        new_actions.append(['SinkBasin', 'PutObject', act, highlevel_idx])
        new_actions.append(['Faucet', 'ToggleObjectOn', act, highlevel_idx])
        new_actions.append(['Faucet', 'ToggleObjectOff', act, highlevel_idx])

    elif act[0] == "Heat":
        if act[1] in shouldbeopened:
            new_actions.append([shouldbeopened[act[1]], 'OpenObject', act, highlevel_idx])
            shouldbeclosed = True

        new_actions.append([act[1], 'PickupObject', act, highlevel_idx])
        if shouldbeclosed:
            new_actions.append([shouldbeopened[act[1]], 'CloseObject', act, highlevel_idx])
            shouldbeclosed = False
            del shouldbeopened[act[1]]
            opened.append(act[1])
        new_actions.append(['Microwave', 'OpenObject', act, highlevel_idx])
        new_actions.append(['Microwave', 'PutObject', act, highlevel_idx])
        new_actions.append(['Microwave', 'CloseObject', act, highlevel_idx])
        new_actions.append(['Microwave', 'ToggleObjectOn', act, highlevel_idx])
        new_actions.append(['Microwave', 'ToggleObjectOff', act, highlevel_idx])
        new_actions.append(['Microwave', 'OpenObject', act, highlevel_idx])
    
    elif act[0] == "Cut":
        if 'Knife' in shouldbeopened:
            new_actions.append([shouldbeopened['Knife'], 'OpenObject', act, highlevel_idx])
            shouldbeclosed = True
        new_actions.append(['Knife', 'PickupObject', act, highlevel_idx])
        if shouldbeclosed:
            new_actions.append([shouldbeopened['Knife'], 'CloseObject', act, highlevel_idx])
            shouldbeclosed = False
            del shouldbeopened['Knife']
            opened.append(act[1])
                    
        if act[1] in shouldbeopened:
            new_actions.append([shouldbeopened[act[1]], 'OpenObject', act, highlevel_idx])
            shouldbeclosed = True

        new_actions.append([act[1], 'SliceObject', act, highlevel_idx])
        if shouldbeclosed:
            new_actions.append([shouldbeopened[act[1]], 'CloseObject', act, highlevel_idx])
            shouldbeclosed = False
            if list_of_actions[idx+1][0] == "Move" and list_of_actions[idx+1][1] == act[1]:
                pass
            else:
                del shouldbeopened[act[1]]
            
            opened.append(act[1])
        new_actions.append(['CounterTop', 'PutObject', act, highlevel_idx])
        sliced.append(act[1])
    
    elif act[0] == "Move":
        target = act[1]
        quantity = 1
        open_needed = False; open_needed_inrecep = False

        if act[2] in shouldbeopened:
            open_needed = True
        if act[2] in INRECEP and act[2]:
            open_needed_inrecep = True
 
        for i in range(quantity):
            if i>0:
                highlevel_idx += 1
            if act[1] in shouldbeopened:
                new_actions.append([shouldbeopened[act[1]], 'OpenObject', act, highlevel_idx])
                shouldbeclosed = True

            new_actions.append([act[1], 'PickupObject', act, highlevel_idx])
            if shouldbeclosed:
                new_actions.append([shouldbeopened[act[1]], 'CloseObject', act, highlevel_idx])
                shouldbeclosed = False
                del shouldbeopened[act[1]]
                opened.append(act[1])
                
            if open_needed:
                new_actions.append([shouldbeopened[act[2]], 'OpenObject', act, highlevel_idx])
            if open_needed_inrecep:
                new_actions.append([act[2], 'OpenObject', act, highlevel_idx])                
            new_actions.append([act[2], 'PutObject', act, highlevel_idx])
            if open_needed:
                new_actions.append([shouldbeopened[act[2]], 'CloseObject', act, highlevel_idx])          
            if open_needed_inrecep:
                new_actions.append([act[2], 'CloseObject', act, highlevel_idx])
        
    elif act[0] == "Pickup":
        if act[1] in shouldbeopened:
            new_actions.append([shouldbeopened[act[1]], 'OpenObject', act, highlevel_idx])

            shouldbeclosed = True
        new_actions.append([act[1], 'PickupObject', act, highlevel_idx])
        if shouldbeclosed:
            new_actions.append([shouldbeopened[act[1]], 'CloseObject', act, highlevel_idx])
            shouldbeclosed = False
            del shouldbeopened[act[1]]
            opened.append(act[1])
                
    elif act[0] == "Put":
        if act[2] in INRECEP:
            new_actions.append([act[2], 'OpenObject', act, highlevel_idx])

        new_actions.append([act[2], 'PutObject', act, highlevel_idx])
        if act[2] in INRECEP:
            new_actions.append([act[2], 'CloseObject', act, highlevel_idx])            
        
    elif act[0] == "Pour":
        new_actions.append([act[2], 'PourObject', act, highlevel_idx])
        
    elif act[0] == "ToggleOn":
        new_actions.append([act[1], 'ToggleObjectOn', act, highlevel_idx])

    elif act[0] == "ToggleOff":
        new_actions.append([act[1], 'ToggleObjectOff', act, highlevel_idx])
        
    elif act[0] == "Open":
        new_actions.append([act[1], 'OpenObject', act, highlevel_idx])

    elif act[0] == "Close":
        new_actions.append([act[1], 'CloseObject', act, highlevel_idx])        
        
    return new_actions, opened, sliced, shouldbeopened, highlevel_idx

def new_action_generate(ext_info, game_id, list_of_actions_):
    # print('\n',ext_info[game_id])
    Targets = list(ext_info[game_id].keys())
    opened = []; sliced = []; shouldbeopened = {}; new_actions = []
    highlevel_idx = -1
    for idx, act in enumerate(list_of_actions_):
        highlevel_idx += 1
        objects = []
        if act[0] == "Cut":
            objects.append("Knife")
            objects.append(act[1])
        else:
            for i in act:
                if i in PICKUPABLE_OBJS:
                    objects.append(i)
        for obj in objects:
            if obj not in opened:
                if obj in Targets:
                    if "location" in ext_info[game_id][obj]:
                        locs = ext_info[game_id][obj]["location"]
                        if isinstance(locs, str):
                            if locs != "X" and locs in INRECEP and locs not in opened:
                                shouldbeopened[obj] = locs
                        elif isinstance(locs, list):
                            allinrecep = True
                            for loc in locs:
                                if loc != "X" and loc not in INRECEP:
                                    allinrecep = False
                            if allinrecep:
                                inrecep = ""
                                for loc in locs:
                                    if loc != "X":
                                        inrecep = loc
                                if inrecep != "" and inrecep not in opened:
                                    shouldbeopened[obj] = inrecep
        new_actions_, opened, sliced, shouldbeopened, highlevel_idx = high_to_low(act, highlevel_idx, ext_info, game_id, opened, sliced, shouldbeopened, list_of_actions_, idx)
        new_actions = new_actions + new_actions_
    return new_actions



def convert( game_id, new_actions, picked_up_obj):
    fname = 'results/exception_files/' +"else_error"+ '.json'
    if not os.path.isfile(fname) :
        users = {"ErrorID" : [game_id] }
        with open(fname, 'w') as f:
            json.dump(users, f)
    else :
        data = open(fname, 'r').read()
        data = json.loads(data)
        error_list = data['ErrorID']
        error_list.append(game_id)
        users = {"ErrorID" : error_list}
        with open(fname, 'w') as f:
            json.dump(data, f)
    
    if 'tfd' in game_id:
        game_id = game_id.split('.')[0]
    with open('new_template/ext_info_all_0216.json') as f:
        ext_info = json.load(f)
        new_actions = new_action_generate(ext_info, game_id, new_actions)
        if picked_up_obj != "None":
            if new_actions[0][1] == "PickupObject" and picked_up_obj == new_actions[0][0]:
                new_actions = new_actions[1:]
            else:
                new_actions_ = []
                new_actions_.append(['CounterTop', 'PutObject', ['Put', picked_up_obj, 'Countertop'], 0])
                for a in new_actions:
                    a[-1] += 1
                    new_actions_.append(a)
                new_actions = new_actions_

        low_action_format = []
        for new_action in new_actions:
            low_action_format.append([new_action[0],new_action[1],new_action[2][0],new_action[3]])
            
        return(low_action_format)

