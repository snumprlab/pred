#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 23:06:53 2022

@author: soyeonmin
"""
def commander_only_from_dialog_history(dialog_history):
    commander_only = []
    for d in dialog_history:
        if d[0] == "Commander":
            commander_only.append(d[1].lower())
    return commander_only

def driver_only_from_dialog_history(dialog_history):
    driver_only = []
    for d in dialog_history:
        if d[0] == "Driver":
            driver_only.append(d[1].lower())
    return driver_only


#Get last from salad

def salad_which_first(commander_d_h):
    potato = ['pota', 'poato', 'poat', 'tato', 'poat']
    lettuce = ['lettuce', 'letuce', 'leut']
    tomato = ['tomato', 'mato', 'tomaot', 'toamto']
    
    potato_idx = -1
    lettuce_idx = -1
    tomato_idx = -1
    
    for p in potato:
        new_idx =  -1
        try:
            new_idx = commander_d_h.index(p)
        except:
            pass
        potato_idx = max(new_idx, potato_idx)
        
    for t in tomato:
        new_idx =  -1
        try:
            new_idx = commander_d_h.index(t)
        except:
            pass
        tomato_idx = max(new_idx, tomato_idx)
        
    for l in lettuce:
        new_idx =  -1
        try:
            new_idx = commander_d_h.index(l)
        except:
            pass
        lettuce_idx = max(new_idx, lettuce_idx)
    
    if tomato_idx == max(potato_idx, lettuce_idx, tomato_idx):
        return "Tomato"
    elif potato_idx == max(potato_idx, lettuce_idx, tomato_idx):
        return "Potato"
    elif lettuce_idx == max(potato_idx, lettuce_idx, tomato_idx):
        return "Lettuce"
    else:
        return None
        

#Boil which path

def boil_microwave_or_stove(commander_d_h, driver_d_h):
    return_pot = None
    commander_d_h = ' '.join(commander_d_h)
    driver_d_h = ' '.join(commander_d_h)
    if ('stove' in commander_d_h) or ('burner' in commander_d_h):
        return True
    elif (('pot' in commander_d_h) or ('pan' in commander_d_h)) and not('bowl' in commander_d_h):
        if 'bowl' in driver_d_h:
            return False
        return True
    elif ('bowl' in commander_d_h) and not(('pot' in commander_d_h) or ('pan' in commander_d_h)):
        return False
    else:
        return False #Just do anything
        