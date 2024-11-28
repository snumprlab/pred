#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 16:49:38 2021

@author: soyeonmin
"""
import pickle
import alfred_utils.gen.constants as constants
import string

import os
import json
from pathlib import Path
from itertools import chain

import sys
sys.path.append('src/')

from teach.dataset.definitions import Definitions
from teach.dataset.dataset import Dataset
from teach.dataset.actions import Action_Keyboard, Action_ObjectInteraction
definitions = Definitions(version="2.0")


from collections import OrderedDict
import copy

exclude = set(string.punctuation)
task_types = {'Boil X',
 'Breakfast',
 'Clean All X',
 'Coffee',
 'N Cooked Slices Of X In Y',
 'N Slices Of X In Y',
 'Plate Of Toast',
 'Put All X In One Y',
 'Put All X On Y',
 'Salad',
 'Sandwich',
 'Water Plant'}

DEFAULT_NONE_DICT = {'On': None, 'Clean': None, 'Sliced': None, 'Pickedup': None, 'Cooked': None, 'Boiled': None, 'Coffee': None, 'Watered': None, 'Toggled': None}
FILLABLE_OBJS = ['Pot', 'Mug', 'Kettle', "HousePlant", 'Cup', 'Bowl', "Bottle", "WineBottle", "WateringCan"]

DATA_FOLDER=os.environ['DATA_DIR']#'/Volumes/untitled2/teach_download/'


from typing import Dict, Union
from functools import lru_cache
def _get_task_type_and_params(edh_instance: dict) -> Dict[str, Union[str, int]]:
    @lru_cache()
    def _internal(game_id: str) -> Dict[str, Union[str, int]]:
        game = json.load((Path(DATA_FOLDER)/"all_game_files"/f"{game_id}.game.json").open("r"))
        task_type = game["tasks"][0]["task_name"]  # assuming each game has only one task
        if not "X" in task_type.split(" "): # hacky
            # no param for this task
            obj_count, obj_target, parent_target = None, None, None
        elif not "Y" in task_type.split(" "): # hacky as well
            # 1 param for this task
            obj_count, obj_target = None, game["tasks"][0]["task_params"][0]
            parent_target = None
        elif not "N" in task_type.split(" "): # hacky as well
            # 2 params for this task
            obj_count, obj_target, parent_target = None, game["tasks"][0]["task_params"][0], game["tasks"][0]["task_params"][2]
        else:
            # 3 params for this task
            obj_count, obj_target, parent_target = game["tasks"][0]["task_params"][0], game["tasks"][0]["task_params"][1], game["tasks"][0]["task_params"][3]
        return dict(task_type=task_type, obj_count=obj_count, obj_target=obj_target, parent_target=parent_target)
    game_id = edh_instance["game_id"]
    return _internal(game_id)


def get_task_type_and_params(edh_instance):
    tmp = _get_task_type_and_params(edh_instance)
    task_type, obj_count, obj_target, parent_target = tmp["task_type"], tmp["obj_count"], tmp["obj_target"], tmp["parent_target"]
    return task_type, obj_count, obj_target, parent_target

def add_target(target, target_action, list_of_actions):
    if target in [a for a in constants.OPENABLE_CLASS_LIST if not(a == 'Box')]:
        list_of_actions.append((target, "OpenObject"))
    list_of_actions.append((target, target_action))
    if target in [a for a in constants.OPENABLE_CLASS_LIST if not(a == 'Box')]:
        list_of_actions.append((target, "CloseObject"))
    return list_of_actions


def desired_objects_by_type(task_type, obj_count=None, obj_target=None, parent_target=None):
    if task_type == 'Boil X':
        assert obj_target !=None
        return [obj_target] #simbotIsBoiled (cooked with water)
    elif task_type == 'Breakfast':
        assert parent_target !=None
        return ['Bread', 'Plate', parent_target] #Forget about this for now. It's too complicated
    elif task_type == 'Clean All X':
        assert obj_target !=None
        return [obj_target] #All obj target must be clean
    elif task_type == 'Coffee':
        return ['Mug'] #Filled with coffee and clean
    elif task_type == 'N Cooked Slices Of X In Y':
        assert obj_target !=None
        assert parent_target !=None
        return [obj_target, parent_target] #obj_target cooked and sliced, parent_target clean
    elif task_type == 'N Slices Of X In Y':
        assert obj_target !=None
        assert parent_target !=None
        return [obj_target, parent_target] #obj_target sliced, parent_target clean
    elif task_type == 'Plate Of Toast':
        return ['Bread', 'Plate'] #Bread needs to be sliced, cooked and Plate needs to be clean
    elif task_type == 'Put All X In One Y':
        assert obj_target !=None
        assert parent_target !=None
        return [obj_target, parent_target] #all obj_target in Y
    elif task_type == 'Put All X On Y':
        assert obj_target !=None
        assert parent_target !=None
        return [obj_target, parent_target] #all obj_target in Y
    elif task_type == 'Salad':
        return ['Lettuce', 'Tomato', 'Potato', 'Plate'] #Lettuce sliced, Tomato sliced, Potato sliced & cooked, Plate clean
    elif task_type == 'Sandwich':
        return ['Bread', 'Lettuce', 'Tomato', 'Plate'] #Lettuce sliced, Tomato sliced, Bread sliced & cooked, Plate clean
    elif task_type == 'Water Plant':
        return ['HousePlant'] #This needs to be filled with liquid
    else:
        raise Exception("Task type not defined!")

def desired_object_state_for_type(task_type, obj_count, obj_target, parent_target):
    desired_objects = desired_objects_by_type(task_type, obj_count, obj_target, parent_target)
    desired_object_states = {o: {'On': None, 'Clean': None, 'Sliced': None, 'Pickedup': None, 'Cooked': None, 'Boiled': None, 'Coffee': None, 'Watered': None}}

    if task_type == 'Boil X':
        desired_object_states[obj_target]['Boiled']= True
    elif task_type == 'Breakfast':
        desired_object_states['Bread']['Sliced'] = True #Give up
    elif task_type == 'Clean All X':
        desired_object_states[obj_target]['Clean'] = True
    elif task_type == 'Coffee':
        desired_object_states['Mug']['Clean'] = True
        desired_object_states['Mug']['Coffee'] = True
    elif task_type == 'N Cooked Slices Of X In Y': #NEED to address N
        desired_object_states[obj_target]['Cooked'] = True
        desired_object_states[obj_target]['Sliced'] = True
        desired_object_states[obj_target]['On'] = parent_target
    elif task_type == 'N Slices Of X In Y':        #NEED to address N
        desired_object_states[obj_target]['Sliced'] = True
        desired_object_states[obj_target]['On'] = parent_target
    elif task_type == 'Plate Of Toast':
        desired_object_states['Bread']['Sliced'] = True
        desired_object_states['Bread']['Cooked'] = True
        desired_object_states['Bread']['On'] = 'Plate'
        #desired_object_states['Plate']['Clean'] = True Ignore
    elif task_type == 'Put All X In One Y':
        return [obj_target, parent_target] #all obj_target in Y
    elif task_type == 'Put All X On Y':
        return [obj_target, parent_target] #all obj_target in Y
    elif task_type == 'Salad':
        return ['Lettuce', 'Tomato', 'Potato', 'Plate'] #Lettuce sliced, Tomato sliced, Potato sliced & cooked, Plate clean
    elif task_type == 'Sandwich':
        return ['Bread', 'Lettuce', 'Tomato', 'Plate'] #Lettuce sliced, Tomato sliced, Bread sliced & cooked, Plate clean
    elif task_type == 'Water Plant':
        return ['HousePlant'] #This needs to be filled with liquid
    else:
        raise Exception("Task type not defined!")



def get_currently_pickedup_object(obj_state_dict):
    picked_up = 0
    picked_up_obj = None
    for o in obj_state_dict:
        if obj_state_dict[o]['Pickedup']:
            picked_up +=1
            picked_up_obj = o

    assert picked_up in [1,0], "picked up "+ str(picked_up) + " obj state dict is " + str(obj_state_dict) 
    return picked_up_obj

def get_currently_in_recep_Y(recep_Y, obj_state_dict):
    in_recep_Y = OrderedDict()
    #if recep_Y== 'SinkBasin':
    #    recep_Y== 'Sink'
    #if recep_Y == 'BathtubBasin':
    #    recepY == 'Bathtub'
    for o in obj_state_dict:
        if obj_state_dict[o]['On'] == recep_Y:
            in_recep_Y[o] = 1

    return list(in_recep_Y.keys())

def recepY_convert(recepY):
    if recepY == 'Sink':
        return 'SinkBasin'
    elif recepY == 'Bathtub':
        return 'BathtubBasin'
    return recepY

def _pickup(obj):
    pass
    
def _place(obj):
    pass

def parse_action_to_state(single_action_pointer, entire_action, obj_state_dict):
    #obj_state_dict = {'On': None, 'Clean': None, 'Sliced': None, 'Pickedup': None, 'Cooked': None, 'Boiled': None, 'Coffee': None, 'Watered': None, 'Toggled': None}
    single_action = entire_action[single_action_pointer]
    action = single_action[0]
    obj = single_action[1]

    if action == 'Pickup':
        obj_state_dict[obj]['Pickedup'] = True
        obj_state_dict[obj]['On'] = None

    elif action == 'Place':
        #Find the most recently picked obj
        pickedup_obj = get_currently_pickedup_object(obj_state_dict)
        assert pickedup_obj != None
        obj_state_dict[pickedup_obj]['Pickedup'] = False
        obj_state_dict[pickedup_obj]['On'] = obj

    elif action == 'Slice':
        obj_state_dict[obj]['Sliced'] = True

    elif action == 'ToggleOn': #Facuet toggled on and off #Previous action was faucet on
        if obj =='Faucet':
            clean_objs = get_currently_in_recep_Y('SinkBasin', obj_state_dict) #It's always Sink not SinkBasin in TEACH history_subgoals
            for clean_obj in clean_objs:
                obj_state_dict[clean_obj]['Clean'] = True
                if clean_obj in FILLABLE_OBJS:
                    obj_state_dict[clean_obj]['Watered'] = True
            obj_state_dict[obj]['Toggled'] = True

        if obj in ['Toaster', 'Microwave', 'StoveKnob']:
            recep_obj = obj
            if obj == 'StoveKnob':
                recep_obj = 'StoveBurner'
            cooked_objs = get_currently_in_recep_Y(recep_obj, obj_state_dict)
            for cooked_obj in cooked_objs:
                obj_state_dict[cooked_obj]['Cooked'] = True
            obj_state_dict[recep_obj]['Toggled'] = True

    elif action == 'ToggleOff':
        obj_state_dict[obj]['Toggled'] = False

    elif action == 'Pour':
        obj_state_dict[obj]['Watered'] = True
        pickedup_obj = get_currently_pickedup_object(obj_state_dict)
        obj_state_dict[pickedup_obj]['Watered'] = False

    #SEE IF ANY HISTORY ENDS WITH OPEN
    #I guess it's fine not to care about Open

    return obj_state_dict

def prev_action_history_to_init_state(history_subgoals, task_type, obj_count=None, obj_target=None, parent_target=None):
    #Parse history_subgoals into entire_action of interactions only
    assert len(history_subgoals) %2 ==0
    entire_action_with_nav = [(history_subgoals[2*i], recepY_convert(history_subgoals[2*i+1])) for i in range(int(len(history_subgoals)/2))]
    #Replace Bathtub into BathtubBasin

    #Replace Sink into SinkBasin

    #Take out all the Navigate
    entire_action = [k for k in entire_action_with_nav if not (k[0] == 'Navigate')]
    obj_state_dict = {o: copy.deepcopy(DEFAULT_NONE_DICT) for o in desired_objects_by_type(task_type, obj_count=None, obj_target=None, parent_target=None)} #Put all the desired and current objects here
    for ei, e in enumerate(entire_action):
        if not (e[1] in  obj_state_dict):
            obj_state_dict[e[1]] = copy.deepcopy(DEFAULT_NONE_DICT)
            #print("Obj state dict at time ", ei, " is ", obj_state_dict)

    for i in range(len(entire_action)):
        obj_state_dict = parse_action_to_state(i, entire_action, obj_state_dict)

    return obj_state_dict


def prev_action_history_to_object_mask():


def intermediate_state_to_desired(obj_state_dict):
    list_of_actions = []
    if task_type == 'Boil X': #Just take Microwave path only
        #Is there pan, pot or something else

        #STOVE PATH
        #Is there pan or pot on the stove?

        #Else if, is there an object filled with water?

        #Else if, if there an object that contains X?

        #Else, am I holding this object?

        #MICROWAVE PATH
        

    elif task_type == 'Water Plant':
        #Has water been filled in any of Bowl, Cup, Mug, Soapbottle, Pot, Bottle?
        #Then just pour
        filled = None
        for recep in ['Bowl', 'Cup', 'Mug', 'Soapbottle', 'Pot', 'Bottle', 'Kettle']:
            if recep in obj_state_dict and obj_state_dict[recep]['Waterfilled']:
                filled = recep
        #The filled recep is the goal. 

        if obj_state_dict[recep]['Pickedup']:
            #just go pour
            list_of_actions = [('HousePlant', 'Pour')]
        else:
            #Pick up and go pour
            list_of_actions = [(filled,"PickupObject"), ('HousePlant', 'Pour')]

        if filled == None:
            #If not, then fill water first
            #Many thing can be in the goal
            #Actually just try putting the plant in the sink
            list_of_actions = [(filled,"PickupObject"), ('SinkBasin',"PutObject"), ('Faucet',"ToggleObjectOn"), ('Faucet',"ToggleObjectOff"), (filled,"PickupObject"), ('HousePlant', 'Pour')]


    elif task_type == 'Coffee': #Clean Mug is good
        #TODO: If I am holding something other than a Mug, put it down
        p = get_currently_pickedup_object(obj_state_dict)
        if p != 'Mug' and p!= None:
            list_of_actions.append((p, 'CounterTop'))

        #Has the mug been cleaned?
        if not(obj_state_dict['Mug']['Clean']):
            if not(obj_state_dict['Mug']['Pickedup']):
                list_of_actions.append(('Mug', 'PickupObject'))
                obj_state_dict['Mug']['Pickedup'] = True
            list_of_actions.append(('SinkBasin', 'PutObject'))
            obj_state_dict['Mug']['Pickedup'] = False
            list_of_actions.append(('Faucet', 'ToggleObjectOn'))
            list_of_actions.append(('Faucet', 'ToggleObjectOff'))
            obj_state_dict['Mug']['Watered'] = True

        #Now assuming that the Mug is clean...
        if obj_state_dict['Mug']['Watered']:
            #Then first pick up and pour on Sink
            if not(obj_state_dict['Mug']['Pickedup']):
                list_of_actions.append(('Mug', 'PickupObject'))
                obj_state_dict['Mug']['Pickedup'] = True
            else:
                list_of_actions.append(('Mug', 'Pour'))
                obj_state_dict['Mug']['Watered'] = False
        else:
            if not(obj_state_dict['Mug']['Pickedup']) and not(obj_state_dict['Mug']['On'] == 'CoffeeMachine'):
                list_of_actions.append(('Mug', 'PickupObject'))
        #Now just get coffee

        if not(obj_state_dict['Mug']['On'] == 'CoffeeMachine'):
            list_of_actions.append(('CoffeeMachine', 'PutObject'))
        list_of_actions.append(('CoffeeMachine', 'ToggleObjectOn'))
        list_of_actions.append(('CoffeeMachine', 'ToggleObjectOff'))

    elif task_type in ['Put All X In One Y', 'Put All X On Y']: #Ignore all
        if not(obj_state_dict[obj_target]['Pickedup']):
            list_of_actions.append((obj_target, 'PickupObject'))
        list_of_actions.append((obj_target, 'PutObject'))

    elif task_type in ['Clean All X']:  #Ignore all
        #If Object picked up 
        if not(obj_state_dict[obj_target]['Pickedup']) and not(obj_state_dict[obj_target]['On'] == "SinkBasin"): #Final receptacle
            list_of_actions.append((obj_target, 'PickupObject'))

        if not(obj_state_dict[obj_target]['On'] == "SinkBasin"):
            list_of_actions.append(("SinkBasin", 'PutObject'))
        list_of_actions.append(('CoffeeMachine', 'ToggleObjectOn'))
        list_of_actions.append(('CoffeeMachine', 'ToggleObjectOff'))

    elif task_type in ['Plate Of Toast']: #Ignoring clean is better
        #What am I holding
        #If I am holding something irrelvant, drop
        #If I am holding something relvant, start from this
        #if holding 'Bread':

        if holding somethint not "Bread":
            list_of_actions.append(("CounterTop", "PutObject"))
        if not holding "Bread":
            list_of_actions.append(("Bread", "PickupObject"))
        if obj_state_dict['Bread']['Cooked'] and obj_state_dict['Bread']['Sliced']:
            #Then just put on plate
            pass
        elif obj_state_dict['Bread']['Sliced']:
            list_of_actions.append(("Toaster", 'PutObject'))
            list_of_actions.append(("Toaster", 'ToggleObjectOn'))
            list_of_actions.append(("Toaster", 'ToggleObjectOff'))
            list_of_actions.append(("Bread", "PickupObject"))
            
        elif obj_state_dict['Bread']['Cooked']:
            list_of_actions.append(("CounterTop", 'PutObject'))
            list_of_actions.append(("Knife", 'PickupObjectObject'))
            list_of_actions.append(("Bread", 'SliceObject'))
            list_of_actions.append(("CounterTop", 'PutObject'))
            list_of_actions.append(("Bread", 'PickupObject'))
        
        else:
            list_of_actions.append(("Toaster", 'PutObject'))
            list_of_actions.append(("Toaster", 'ToggleObjectOn'))
            list_of_actions.append(("Toaster", 'ToggleObjectOff'))
            list_of_actions.append(("Bread", "PickupObject"))
            list_of_actions.append(("CounterTop", 'PutObject'))
            list_of_actions.append(("Knife", 'PickupObjectObject'))
            list_of_actions.append(("Bread", 'SliceObject'))
            list_of_actions.append(("CounterTop", 'PutObject'))
            list_of_actions.append(("Bread", 'PickupObject'))

        list_of_actions.append(("Plate", 'PutObject'))

    elif task_type in ['Salad']: #Ignoring clean plate is better
        if holding somethint not in ['Potato', 'Lettuce', 'Tomato']:
            list_of_actions.append(("CounterTop", "PutObject"))
        #if holding something
        if holding something in ['Potato', 'Lettuce', 'Tomato']:
            if something 




    elif task_type in ['Sandwich']:  #Ignoring clean plate is better


    elif task_type == 'N Slices Of X In Y': #Set N to 1 
        #Is X already sliced?
        if not(obj_state_dict[obj_target]['Sliced']):
            #If I am holding something ELSE, put it down: TODO
            if not(obj_state_dict['Knife']['Pickedup'] or obj_state_dict['ButterKnife']['Pickedup']):
                list_of_actions.append(('Knife', 'PickupObject')) #DOUBLE CHECK KNIFE OR BUTTERKNIFE
            list_of_actions.append((obj_target, 'SliceObject'))
            list_of_actions.append(('CounterTop', 'PutObject'))

        #What am I holding?


        #If not sliced, pick up knife and slice

        #Put X in Y

    elif task_type == 'N Cooked Slices Of X In Y':
        #Is X already cooked and sliced?



def clean_plate(holding_plate=False):
    list_of_actions = []
    if not(holding_plate):
        list_of_actions.append(('Plate', 'PickupObject'))
    list_of_actions.append(("SinkBasin", 'PutObject'))
    list_of_actions.append(('Faucet', 'ToggleObjectOn'))
    list_of_actions.append(('Faucet', 'ToggleObjectOff'))
    list_of_actions.append(("Bread", "PickupObject"))
    list_of_actions.append(("Plate", 'PutObject'))

    return list_of_actions

def slice_x(x, holding_x = False, holding_knife=False):
    pass

def cook_x():
    pass

def boil_x():
    pass

def put_on_x():
    pass

def put_on_

def parse_init_states_from_actions(prev_actions_before_init):

    #First pair by two

    #

    pass

# def init_object_state_for_type(task_type, obj_count, obj_target, parent_target):
#     desired_objects = desired_objects_by_type(task_type, obj_count, obj_target, parent_target)
#     desired_object_states = {o: {'On': None, 'Clean': None, 'Sliced': None, 'Pickedup': None, 'Cooked': None, 'Coffee': None, 'Watered': None}}

#     if task_type == 'Boil X':
#         desired_object_states[obj_target]['Cooked']= True

#     elif task_type == 'Breakfast':
#         desired_object_states['Bread']['Cooked']= True

#     elif task_type == 'Clean All X':

#     elif task_type == 'Coffee':

#     elif task_type == 'N Cooked Slices Of X In Y':

#     elif task_type == 'N Slices Of X In Y':

#     elif task_type == 'Plate Of Toast':

#     elif task_type == 'Put All X In One Y':

#     elif task_type == 'Put All X On Y':

#     elif task_type == 'Salad':

#     elif task_type == 'Sandwich':

#     elif task_type == 'Water Plant':

#     else:
#         raise Exception("Task type not defined!")






def get_list_of_highlevel_actions(task_type, obj_count, obj_target, parent_target, args_nonsliced=False):
    if parent_target == "Sink":
        parent_target = "SinkBasin"
    if parent_target == "Bathtub":
        parent_target = "BathtubBasin"
    
    
    categories_in_inst = []
    list_of_highlevel_actions = []
    second_object = []
    caution_pointers = []

    sliced = 0
    if task_type in ['Breakfast', 'N Cooked Slices Of X In Y', 'N Slices Of X In Y', 'Plate Of Toast', 'Salad', 'Sandwich']:
        sliced = 1

    if sliced == 1:
       list_of_highlevel_actions.append(("Knife", "PickupObject"))
       list_of_highlevel_actions.append((obj_target, "SliceObject"))
       caution_pointers.append(len(list_of_highlevel_actions))
       list_of_highlevel_actions.append(("SinkBasin", "PutObject"))
       categories_in_inst.append(obj_target)
       
    if sliced:
        obj_target = obj_target +'Sliced'

    #######################################################################
    if task_type == 'Boil X':
        list_of_highlevel_actions.append((obj_target, "Pickup"))
        list_of_highlevel_actions.append(("Pot", "Place"))
        list_of_highlevel_actions.append(("Pot", "Pickup"))
        list_of_highlevel_actions.append(("Sink", "Place"))
        list_of_highlevel_actions.append(("Faucet", "ToggleOn"))
        list_of_highlevel_actions.append(("Faucet", "ToggleOff"))
        list_of_highlevel_actions.append(("Pot", "Pickup"))
        list_of_highlevel_actions.append(("StoveBurner", "Place"))
        list_of_highlevel_actions.append(("StoveKnob", "ToggleOn"))#Is this in the objects of MaskRCNN?
        list_of_highlevel_actions.append(("StoveKnob", "ToggleOff"))

    #elif task_type == 'Breakfast':
    #    pass
        
    elif task_type == 'Plate Of Toast':
        list_of_highlevel_actions.append(("Knife", "Pickup"))
        list_of_highlevel_actions.append(("Bread", "Slice"))
        list_of_highlevel_actions.append(("CounterTop", "Place"))
        list_of_highlevel_actions.append(("Bread", "Pickup"))
        list_of_highlevel_actions.append(("Toaster", "Place"))
        list_of_highlevel_actions.append(("Toaster", "ToggleOn"))
        list_of_highlevel_actions.append(("Toaster", "ToggleOff"))
        list_of_highlevel_actions.append(("Bread", "Pickup"))
        list_of_highlevel_actions.append(("Plate", "Place"))


    elif task_type == 'Water Plant':
        list_of_highlevel_actions.append(("Cup", "Pickup")) #REPLACE THIS WITH OBJ TARGET LATER
        list_of_highlevel_actions.append(("Sink", "Place"))
        list_of_highlevel_actions.append(("Faucet", "ToggleOn"))
        list_of_highlevel_actions.append(("Faucet", "ToggleOff"))
        list_of_highlevel_actions.append(("Cup", "Pickup"))
        list_of_highlevel_actions.append(("Plant", "Pour"))

    elif task_type == 'Clean All X':
        all_num = 1
        list_of_highlevel_actions.append((obj_target, "Pickup"))
        list_of_highlevel_actions.append(("Sink", "Pour"))
        list_of_highlevel_actions.append(("Sink", "PutObject"))
        list_of_highlevel_actions.append(("Faucet", "ToggleOn"))
        list_of_highlevel_actions.append(("Faucet", "ToggleOff"))
        list_of_highlevel_actions.append((obj_target, "Pickup"))
        list_of_highlevel_actions.append(("Sink", "Pour"))
        list_of_highlevel_actions.append(("Sink", "Place"))

    elif task_type == 'Coffee':
        list_of_highlevel_actions.append(("Mug", "PickupObject"))
        list_of_highlevel_actions.append(("Sink", "PutObject"))
        list_of_highlevel_actions.append(("Faucet", "ToggleObjectOn"))
        list_of_highlevel_actions.append(("Faucet", "ToggleObjectOff"))
        list_of_highlevel_actions.append(("Mug", "PickupObject"))
        list_of_highlevel_actions.append(("Sink", "Pour"))
        list_of_highlevel_actions.append(("CoffeeMachine", "PutObject"))
        list_of_highlevel_actions.append(("CoffeeMachine", "ToggleObjectOn"))
        list_of_highlevel_actions.append(("CoffeeMachine", "ToggleOff"))
        #list_of_highlevel_actions.append(("Mug", "PickupObject"))
        #list_of_highlevel_actions.append(("CounterTop", "PutObject"))

    elif task_type == 'Put All X On Y' or task_type == 'Put All X in one Y': #ignore "ALL" for now
        all_num = 1
        list_of_highlevel_actions.append((obj_target, "PickupObject"))
        caution_pointers.append(len(list_of_highlevel_actions))
        list_of_highlevel_actions = add_target(parent_target, "PutObject", list_of_highlevel_actions)
        categories_in_inst.append(obj_target)
        categories_in_inst.append(parent_target)

    else: #Just carelessly do these
        list_of_actions = [('Cloth', 'PickupObject'), ('SinkBasin', 'PutObject')]
        categories_in_inst = ['Cloth', 'SinkBasin']

    #######################################################################

    if sliced == 1:
       if not(parent_target == "SinkBasin"):
            categories_in_inst.append("SinkBasin")

    return list_of_highlevel_actions, categories_in_inst, second_object, caution_pointers


def list_of_conditions(task_type):
    return definitions.map_tasks_name2info[task_type].components

def list_of_parameters(task_type):
    return definitions.map_tasks_name2info[task_type].task_params
