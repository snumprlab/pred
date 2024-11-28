

class Add_List_of_Actions_Util:
	def __init__(self):
		self.future_list_of_highlevel_actions = []
		self.categories_in_inst = []
	    self.second_object = []
	    self.caution_pointers = []	

	def add_target(target, target_action, list_of_actions):
	    if target in [a for a in constants.OPENABLE_CLASS_LIST if not(a == 'Box')]:
	        list_of_actions.append((target, "OpenObject"))
	    list_of_actions.append((target, target_action))
	    if target in [a for a in constants.OPENABLE_CLASS_LIST if not(a == 'Box')]:
	        list_of_actions.append((target, "CloseObject"))
	    return list_of_actions