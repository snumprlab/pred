#Test cases for object_state.py

task_type = 'Water Plant'
history_subgoals = task_type2_subgoals['Water Plant'][10]['history_subgoals']
history_subgoals = task_type2_subgoals['Water Plant'][11]['history_subgoals']
history_subgoals =task_type2_subgoals['Water Plant'][12]['history_subgoals']
history_subgoals =task_type2_subgoals['Water Plant'][13]['history_subgoals']
history_subgoals =task_type2_subgoals['Water Plant'][17]['history_subgoals']

history_subgoals = ['Navigate', 'Mug', 'Pickup', 'Mug', 'ToggleOn', 'Faucet', 'Place', 'Sink', 'Navigate', 'Faucet', 'ToggleOff', 'Faucet']
obj_count = None
obj_target = None
parent_target = None

os1 = ObjectState(history_subgoals, task_type, obj_count, obj_target, parent_target)
init_obj_state_dict = os1.init_obj_state_dict
desired_obj_state_dict = os1.desired_object_states
future_list_of_highlevel_actions = os1.future_list_of_highlevel_actions



task_type = 'Boil X'
history_subgoals = task_type2_subgoals['Boil X'][0]['history_subgoals']
history_subgoals = task_type2_subgoals['Boil X'][1]['history_subgoals'] #[]
history_subgoals = task_type2_subgoals['Boil X'][7]['future_subgoals']
history_subgoals = task_type2_subgoals['Boil X'][9]['history_subgoals']

os1 = ObjectState(history_subgoals, task_type, obj_count, obj_target, parent_target)
init_obj_state_dict = os1.init_obj_state_dict
desired_obj_state_dict = os1.desired_object_states
future_list_of_highlevel_actions = os1.future_list_of_highlevel_actions

task_type = 'Coffee'
history_subgoals = task_type2_subgoals[task_type][0]['history_subgoals']
history_subgoals = task_type2_subgoals[task_type][1]['history_subgoals'] #[]
history_subgoals = task_type2_subgoals[task_type][2]['history_subgoals']
history_subgoals = task_type2_subgoals[task_type][4]['history_subgoals']
history_subgoals = task_type2_subgoals[task_type][7]['history_subgoals']


os1 = ObjectState(history_subgoals, task_type, obj_count, obj_target, parent_target)
init_obj_state_dict = os1.init_obj_state_dict
desired_obj_state_dict = os1.desired_object_states
future_list_of_highlevel_actions = os1.future_list_of_highlevel_actions
print("history subgoals is ")
history_subgoals
print("future_list_of_highlevel_actions is ")
future_list_of_highlevel_actions

task_type = 'Plate Of Toast'
history_subgoals = task_type2_subgoals[task_type][0]['history_subgoals']
history_subgoals = task_type2_subgoals[task_type][1]['history_subgoals']
history_subgoals = task_type2_subgoals[task_type][2]['history_subgoals']
history_subgoals = task_type2_subgoals[task_type][3]['history_subgoals']
history_subgoals = task_type2_subgoals[task_type][4]['history_subgoals']
history_subgoals = task_type2_subgoals[task_type][5]['history_subgoals']


os1 = ObjectState(history_subgoals, task_type, obj_count, obj_target, parent_target)
init_obj_state_dict = os1.init_obj_state_dict
desired_obj_state_dict = os1.desired_object_states
future_list_of_highlevel_actions = os1.future_list_of_highlevel_actions
print("history subgoals is ")
history_subgoals
print("future_list_of_highlevel_actions is ")
future_list_of_highlevel_actions


task_type = 'Salad'
history_subgoals = task_type2_subgoals[task_type][0]['history_subgoals']

os1 = ObjectState(history_subgoals, task_type, obj_count, obj_target, parent_target)
init_obj_state_dict = os1.init_obj_state_dict
desired_obj_state_dict = os1.desired_object_states
future_list_of_highlevel_actions = os1.future_list_of_highlevel_actions
print("history_subgoals is ")
history_subgoals
print("future_list_of_highlevel_actions is ")
future_list_of_highlevel_actions

#Checked Sandwich too

#Do things that require obj_type
task_type = 'Clean All X' #Check this later #HAS TABLEWARE, ETC
cur = task_type2_subgoals[task_type][0]
cur = task_type2_subgoals[task_type][1]

history_subgoals = cur['history_subgoals']; obj_count =cur['params']['obj_count']; obj_target=cur['params']['obj_target']; parent_target = cur['params']['parent_target']

os1 = ObjectState(history_subgoals, task_type, obj_count, obj_target, parent_target)
init_obj_state_dict = os1.init_obj_state_dict
desired_obj_state_dict = os1.desired_object_states
future_list_of_highlevel_actions = os1.future_list_of_highlevel_actions
print("history_subgoals is ")
history_subgoals
print("future_list_of_highlevel_actions is ")
future_list_of_highlevel_actions

#Get frequency of params
all_clean_params = {}
for i, t in enumerate(task_type2_subgoals[task_type]):
	if not(t['params']['obj_target'] in all_clean_params):
		all_clean_params[t['params']['obj_target']] = 0
	all_clean_params[t['params']['obj_target']] +=1

#Get all the drinkware indexes
drinkware_indices = []
for i, t in enumerate(task_type2_subgoals[task_type]):
	if t['params']['obj_target'] == "Drinkware":
		drinkware_indices.append(i)

tableware_indices = []
for i, t in enumerate(task_type2_subgoals[task_type]):
	if t['params']['obj_target'] == "Tableware":
		tableware_indices.append(i)

#Look at https://github.com/alexa/teach/blob/fec0ff511841d7ad9d872054ea7ebf407aae745f/src/teach/meta_data_files/ai2thor_resources/custom_object_classes.json

task_type = 'Put All X On Y' #Check this first #HAS 'Tables', 'Silverware'
cur = task_type2_subgoals[task_type][0]

history_subgoals = cur['history_subgoals']; obj_count =cur['params']['obj_count']; obj_target=cur['params']['obj_target']; parent_target = cur['params']['parent_target']

os1 = ObjectState(history_subgoals, task_type, obj_count, obj_target, parent_target)
init_obj_state_dict = os1.init_obj_state_dict
desired_obj_state_dict = os1.desired_object_states
future_list_of_highlevel_actions = os1.future_list_of_highlevel_actions
print("history_subgoals is ")
history_subgoals
print("obj target is ", obj_target, "Parent target is ", parent_target)
print("future_list_of_highlevel_actions is ")
future_list_of_highlevel_actions

all_clean_params = {'obj': {}, 'parent': {}}
for i, t in enumerate(task_type2_subgoals[task_type]):
	if not(t['params']['obj_target'] in all_clean_params['obj']):
		all_clean_params['obj'][t['params']['obj_target']] = 0
	all_clean_params['obj'][t['params']['obj_target']] +=1
	if not(t['params']['parent_target'] in all_clean_params['parent']):
		all_clean_params['parent'][t['params']['parent_target']] = 0
	all_clean_params['parent'][t['params']['parent_target']] +=1

task_type = 'Put All X In One Y' #HAS 'Tables'

cur = task_type2_subgoals[task_type][0]

history_subgoals = cur['history_subgoals']; obj_count =cur['params']['obj_count']; obj_target=cur['params']['obj_target']; parent_target = cur['params']['parent_target']

os1 = ObjectState(history_subgoals, task_type, obj_count, obj_target, parent_target)
init_obj_state_dict = os1.init_obj_state_dict
desired_obj_state_dict = os1.desired_object_states
future_list_of_highlevel_actions = os1.future_list_of_highlevel_actions
print("history_subgoals is ")
history_subgoals
print("obj target is ", obj_target, "Parent target is ", parent_target)
print("future_list_of_highlevel_actions is ")
future_list_of_highlevel_actions

all_clean_params = {'obj': {}, 'parent': {}}
for i, t in enumerate(task_type2_subgoals[task_type]):
	if not(t['params']['obj_target'] in all_clean_params['obj']):
		all_clean_params['obj'][t['params']['obj_target']] = 0
	all_clean_params['obj'][t['params']['obj_target']] +=1
	if not(t['params']['parent_target'] in all_clean_params['parent']):
		all_clean_params['parent'][t['params']['parent_target']] = 0
	all_clean_params['parent'][t['params']['parent_target']] +=1



task_type = 'N Cooked Slices Of X In Y'

cur = task_type2_subgoals[task_type][0]

history_subgoals = cur['history_subgoals']; obj_count =cur['params']['obj_count']; obj_target=cur['params']['obj_target']; parent_target = cur['params']['parent_target']

os1 = ObjectState(history_subgoals, task_type, obj_count, obj_target, parent_target)
init_obj_state_dict = os1.init_obj_state_dict
desired_obj_state_dict = os1.desired_object_states
future_list_of_highlevel_actions = os1.future_list_of_highlevel_actions
print("history_subgoals is ")
history_subgoals
print("obj target is ", obj_target, "Parent target is ", parent_target)
print("future_list_of_highlevel_actions is ")
future_list_of_highlevel_actions





