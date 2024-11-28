#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 20:27:52 2022

@author: soyeonmin
"""
#import ipdb
edh_instance = task_type2_edh_instance_ids['Water Plant']['8d187c2be57b026b_97bd'][3]['edh_instance']
history_subgoals = edh_instance['history_subgoals']
prev_actions = edh_instance['driver_action_history']
success_prev_actions = [True]*len(prev_actions)

def prev_actions_2_history_subgoals(prev_actions, success_prev_actions):
    interactions = []
    for i, prev_action in enumerate(prev_actions):
        #print("prev action is ", prev_action)
        if prev_action['obj_interaction_action'] and ('oid' in prev_action):
            #ipdb.set_trace()
            if success_prev_actions[i]:
                #interactions.append(prev_action['action_name'])
                #interactions.append(prev_action['oid'].split('|')[0])
                interactions.append((prev_action['action_name'], prev_action['oid'].split('|')[0]))
    return interactions


h_produced = prev_actions_2_history_subgoals(prev_actions, success_prev_actions)
entire_action_with_nav = [(history_subgoals[2*i], history_subgoals[2*i+1]) for i in range(int(len(history_subgoals)/2))]
entire_action = [k for k in entire_action_with_nav if not (k[0] == 'Navigate')]

print(h_produced)
print(entire_action)
valid_seen_edh_files = (Path(DATA_FOLDER)/"edh_instances/valid_seen").glob("*.json")

all_valid_edh_files = list(chain(
    (Path(DATA_FOLDER)/"edh_instances/valid_seen").glob("*.json"),
    (Path(DATA_FOLDER)/"edh_instances/valid_unseen").glob("*.json")
))

for i, edh_instance_file in enumerate(all_valid_edh_files):
    edh_instance = json.load(edh_files.open("r"))
    history_subgoals = edh_instance['history_subgoals']
    prev_actions = edh_instance['driver_action_history']
    success_prev_actions = [True]*len(prev_actions)
    
    h_produced = prev_actions_2_history_subgoals(prev_actions, success_prev_actions)
    entire_action_with_nav = [(history_subgoals[2*i], history_subgoals[2*i+1]) for i in range(int(len(history_subgoals)/2))]
    entire_action = [k for k in entire_action_with_nav if not (k[0] == 'Navigate')]
    
    assert h_produced == entire_action
    