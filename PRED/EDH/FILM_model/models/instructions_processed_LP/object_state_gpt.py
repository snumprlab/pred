#Let's write a class again
from collections import OrderedDict
import copy
import pickle

DEFAULT_NONE_DICT = {'On': None, 'Clean': None, 'Sliced': None, 'Pickedup': None, 'Cooked': None, 'Boiled': None, 'Coffee': None, 'FILLED_WITH': None, 'Toggled': None, "Open": None}
# FILLABLE_OBJS = ['Kettle', 'Glassbottle', "WineBottle", "WateringCan", 'Mug', 'Cup', 'Bowl']#  "HousePlant", 'Pot'
FILLABLE_OBJS = [ 'Mug', 'Cup', 'Bowl']#  "HousePlant", 'Pot''Kettle', 'Glassbottle', "WineBottle", "WateringCan",
BOILABLE_OBJS = ["Pot", "Bowl", "Mug", "Cup"]
OPENABLE_CLASS_LIST = ['Fridge', 'Cabinet', 'Microwave', 'Drawer', 'Safe']#, 'Box']
Silverware = ["Fork", "Spoon", "ButterKnife"]
Tables = ["Shelf", "DiningTable", "CoffeeTable", "Desk", "SideTable"]
Tableware = ["Plate", "Bowl", 'Glassbottle', "Cup", "Mug", "PepperShaker", "SaltShaker", "WineBottle"]
SmallHandheldObjects = ["Watch", "RemoteControl", "Pencil", "Pen", "KeyChain", "CreditCard", "CD", "CellPhone", "Book"]
Drinkware = ["Mug", "Cup"]
Dishware = ['Bowl', 'Plate']
SportsEquipment = ['BaseballBat', 'BasketBall', 'TennisRacket']
Condiments = ['PepperShaker', 'SaltShaker']
Chairs = ['ArmChair', 'Sofa']
Furniture = ['ArmChair', 'Bed', 'CoffeeTable', 'Desk', 'DiningTable', 'Dresser', 'Ottoman', 'Shelf', 'SideTable', 'Sofa', 'TVStand']
Cookware = ['Kettle', 'Knife', 'Ladle', 'Pan', 'Pot', 'Spatula']
Fruit = ['Apple', 'Tomato']

#TEACH actions to my actions
FILM_actions_2_teach_actions_dict = {
                        "PickupObject":'Pickup',
                        "PutObject": "Place",
                        "OpenObject": 'Open',
                        "CloseObject": 'Close',
                        'SliceObject': 'Slice',
                        "ToggleObjectOn": 'ToggleOn',
                        "ToggleObjectOff": 'ToggleOff',
                        "PourObject": "Pour"
                        }

teach_actions_2_FILM_actions_dict = {v:k for k, v in FILM_actions_2_teach_actions_dict.items()}

default_put_place = "CounterTop" #Can be SinkBasin too

class ObjectState:
    #def __init__(self, history_subgoals, task_type, obj_count=None, obj_target=None, parent_target=None):
    def __init__(self, dialog_history, prev_actions, task_type, obj_count=None, obj_target=None, parent_target=None, intermediate_obj_gpt = None):
        self.FILLABLE_OBJS = FILLABLE_OBJS
        self.Tableware = Tableware
        self.Drinkware = Drinkware
        self.SmallHandheldObjects = SmallHandheldObjects
        self.Silverware = Silverware
        self.Tables = Tables
        self.BOILABLE_OBJS = BOILABLE_OBJS
        self.OPENABLE_CLASS_LIST = OPENABLE_CLASS_LIST
        self.Chairs = Chairs
        self.Furniture = Furniture
        self.Cookware = Cookware
        self.Fruit = Fruit
        self.Drinkware  = Drinkware 
        self.SportsEquipment = SportsEquipment
        self.Condiments = Condiments
        self.Dishware = Dishware

        #self.sem_exp = sem_exp
        self.prev_actions = prev_actions #edh_instance['driver_action_history']
        self.task_type = task_type
        self.obj_count = obj_count
        self.obj_target =obj_target
        self.parent_target = parent_target
        #pickle.dump(dialog_history, open('/results/dialog_history.p', 'wb'))
        #pickle.dump(prev_actions, open('/results/prev_actions.p', 'wb'))
        #pickle.dump(task_type, open('/results/task_type.p', 'wb'))
        #pickle.dump(obj_target, open('/results/obj_target.p', 'wb'))
        #pickle.dump(parent_target, open('/results/parent_target.p', 'wb'))

        #def setup(self, dialog_history, prev_actions, task_type, obj_count=None, obj_target=None, parent_target=None):
        self.c_dh = self.commander_only_from_dialog_history(dialog_history)
        self.d_dh = self.driver_only_from_dialog_history(dialog_history)
        
        if self.task_type == "Boil X":
            self.obj_target = "Potato"

        self.intermediate_obj_gpt = intermediate_obj_gpt
        
        # Set Objeect list depend on Task type 
        self.desired_objects_by_type()
        # Set Object state depend on obj or Task type 
        self.desired_object_state_for_type()
        

        self.picked_up_obj = None
        
        ########위치 변경 ######
        self.future_list_of_highlevel_actions = []
        self.categories_in_inst = []
        self.second_object = [] #Meaningless
        self.caution_pointers = []   
        #########################
        
        #INIT obj state dict
        self.obj_state_dict = self.prev_action_history_to_init_state()
        #print("self.obj_state_dict is ", self.obj_state_dict)
        self.init_obj_state_dict = copy.deepcopy(self.obj_state_dict)
        #Initialize Picked Up mask
        #self.picked_up_mask =  #TODO
# =============================================================================
#         self.action_2_function = FILM_action2_function = {"PickupObject": self._pickup,
#                                                             "PutObject": self._put,
#                                                             "OpenObject": self._open_x,
#                                                             "CloseObject": self._close_x,
#                                                             'SliceObject': self._slice_x, 
#                                                             "ToggleObjectOn": self._toggle_on_x,
#                                                             "ToggleObjectOff": self._toggle_off_x} 
# =============================================================================


        self.actions_until_desired_state()
        
        
        
        
        ######################### Alread completed action ################################3  TMP
        if len( self.future_list_of_highlevel_actions ) == 0:
            self.future_list_of_highlevel_actions.append((default_put_place,"PutObject"))
        ################################################################################### TMP
        

    def commander_only_from_dialog_history(self, dialog_history):
        commander_only = []
        for d in dialog_history:
            if d[0] == "Commander":
                commander_only.append(d[1].lower())
        return commander_only

    def driver_only_from_dialog_history(self, dialog_history):
        driver_only = []
        for d in dialog_history:
            if d[0] == "Driver":
                driver_only.append(d[1].lower())
        return driver_only

    def boil_microwave_or_stove(self, commander_d_h, driver_d_h):
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


    def prev_actions_2_history_subgoals(self, prev_actions):
        success_prev_actions = [True]*len(prev_actions)
        interactions = []
        for i, prev_action in enumerate(prev_actions):
            #print("prev action is ", prev_action)
            if prev_action['obj_interaction_action'] and ('oid' in prev_action):
                #ipdb.set_trace()
                if success_prev_actions[i]:
                    #interactions.append(prev_action['action_name'])
                    #interactions.append(prev_action['oid'].split('|')[0])
                    interactions.append((prev_action['action_name'], self._recepY_convert(prev_action['oid'].split('|')[0])))
        return interactions


    def get_currently_in_recep_Y(self, recep_Y):
        in_recep_Y = OrderedDict()
        #if recep_Y== 'SinkBasin':
        #    recep_Y== 'Sink'
        #if recep_Y == 'BathtubBasin':
        #    recepY == 'Bathtub'
        for o in self.obj_state_dict:
            if self.obj_state_dict[o]['On'] == recep_Y:
                in_recep_Y[o] = 1
    
        return list(in_recep_Y.keys())
        
    def salad_which_first(self, commander_d_h):
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


    def desired_objects_by_type(self):
        obj_count=self.obj_count; obj_target=self.obj_target; parent_target=self.parent_target
        if self.task_type == 'Coffee':
            self.desired_objects = ['Mug', "Faucet", "CoffeeMachine"]
        elif self.task_type == 'Boil X':
            #assert obj_target !=None
            self.desired_objects = [obj_target, "Faucet"] #simbotIsBoiled (cooked with water)
        elif self.task_type == 'Breakfast':
            #assert parent_target !=None
            self.desired_objects = ['Bread', 'Plate', "Faucet"] #Forget about this for now. It's too complicated
        elif self.task_type == 'Clean All X':
            #assert obj_target !=None
            self.desired_objects = [obj_target] #All obj target must be clean 
        elif self.task_type == 'Coffee':
            self.desired_objects = ['Mug', "Faucet"] #Filled with coffee and clean
        elif self.task_type == 'N Cooked Slices Of X In Y':
            #assert obj_target !=None
            #assert parent_target !=None
            self.desired_objects =  [obj_target, parent_target] #obj_target cooked and sliced, parent_target clean
        elif self.task_type == 'N Slices Of X In Y':
            #assert obj_target !=None
            #assert parent_target !=None
            self.desired_objects =  [obj_target, parent_target] #obj_target sliced, parent_target clean
        elif self.task_type == 'Plate Of Toast':
            self.desired_objects =  ['Bread', 'Plate', "Faucet"] #Bread needs to be sliced, cooked and Plate needs to be clean
        elif self.task_type == 'Put All X In One Y':
            #assert obj_target !=None
            #assert parent_target !=None
            self.desired_objects =  [obj_target, parent_target] #all obj_target in Y
        elif self.task_type == 'Put All X On Y':
            #assert obj_target !=None
            #assert parent_target !=None
            self.desired_objects = [obj_target, parent_target] #all obj_target in Y
        elif self.task_type == 'Salad':
            self.desired_objects =  ['Lettuce', 'Tomato', 'Potato', 'Plate', "Faucet"] #Lettuce sliced, Tomato sliced, Potato sliced & cooked, Plate clean
        elif self.task_type == 'Sandwich':
            self.desired_objects =  ['Bread', 'Lettuce', 'Tomato', 'Plate', "Faucet"] #Lettuce sliced, Tomato sliced, Bread sliced & cooked, Plate clean
        elif self.task_type == 'Water Plant':
            self.desired_objects = ['HousePlant'] #This needs to be filled with liquid
        else:
            raise Exception("Task type not defined!")


    def desired_object_state_for_type(self):
        obj_count=self.obj_count; obj_target=self.obj_target; parent_target=self.parent_target
        task_type = self.task_type
        #self.desired_objects = desired_objects_by_type(task_type, obj_count, obj_target, parent_target)
        self.desired_object_states = {o: copy.deepcopy(DEFAULT_NONE_DICT) for o in self.desired_objects}
        if "Faucet" in self.desired_object_states:
            self.desired_object_states["Faucet"]['Toggled'] = False

        if task_type == "Coffee":
            self.desired_object_states["Mug"]['Clean']= True
            self.desired_object_states["Mug"]['FILLED_WITH']= "Coffee"
            self.desired_object_states["CoffeeMachine"]['Toggled']= True

        elif task_type == 'Boil X':
            self.desired_object_states[obj_target]['Boiled']= True
        elif task_type == 'Breakfast':
            self.desired_object_states['Bread']['Sliced'] = True #Give up
        elif task_type == 'Clean All X':
            self.desired_object_states[obj_target]['Clean'] = True
        elif task_type == 'Coffee':
            self.desired_object_states['Mug']['Clean'] = True
            self.desired_object_states['Mug']['Coffee'] = True
        elif task_type == 'N Cooked Slices Of X In Y': #NEED to address N
            self.desired_object_states[obj_target]['Cooked'] = True
            self.desired_object_states[obj_target]['Sliced'] = True
            self.desired_object_states[obj_target]['On'] = parent_target
        elif task_type == 'N Slices Of X In Y':        #NEED to address N
            self.desired_object_states[obj_target]['Sliced'] = True
            self.desired_object_states[obj_target]['On'] = parent_target
            self.desired_object_states[parent_target]['Clean'] = True  #####################3# Ignore  #####
        elif task_type == 'Plate Of Toast':
            self.desired_object_states['Bread']['Sliced'] = True
            self.desired_object_states['Bread']['Cooked'] = True
            self.desired_object_states['Bread']['On'] = 'Plate'
            self.desired_object_states['Plate']['Clean'] = True# Ignore  #####
        elif task_type == 'Put All X In One Y':
            self.desired_object_states[obj_target]['On'] = parent_target
        elif task_type == 'Put All X On Y':
            self.desired_object_states[obj_target]['On'] = parent_target
        elif task_type == 'Salad':
             self.desired_object_states['Lettuce']['Sliced'] = True
             self.desired_object_states['Tomato']['Sliced'] = True
             self.desired_object_states['Potato']['Sliced'] = True
             self.desired_object_states['Potato']['Cooked'] = True
             self.desired_object_states['Lettuce']['On'] = "Plate"
             self.desired_object_states['Tomato']['On'] = "Plate"
             self.desired_object_states['Potato']['On'] = "Plate"
             #desired_object_states['Plate']['Clean'] = True Ignore
        elif task_type == 'Sandwich':
             self.desired_object_states['Bread']['Sliced'] = True
             self.desired_object_states['Bread']['Cooked'] = True
             self.desired_object_states['Bread']['On'] = 'Plate'
             self.desired_object_states['Lettuce']['Sliced'] = True
             self.desired_object_states['Tomato']['Sliced'] = True
             self.desired_object_states['Lettuce']['On'] = "Plate"
             self.desired_object_states['Tomato']['On'] = "Plate"
             #desired_object_states['Plate']['Clean'] = True Ignore
        elif task_type == 'Water Plant':
            self.desired_object_states['HousePlant']['FILLED_WITH'] = "Water"
        else:
            raise Exception("Task type not defined!")
            
    

    def prev_action_history_to_init_state(self, prev_actions=True):
        #Parse history_subgoals into entire_action of interactions only
        entire_action = self.prev_actions_2_history_subgoals(self.prev_actions)


        self.obj_state_dict = {o: copy.deepcopy(DEFAULT_NONE_DICT) for o in self.desired_objects} #Put all the desired and current objects here
        for ei, e in enumerate(entire_action):
            if not (e[1] in  self.obj_state_dict):
                self.obj_state_dict[e[1]] = copy.deepcopy(DEFAULT_NONE_DICT)

        for i in range(len(entire_action)):
            self.parse_action_to_state(i, entire_action)

        if ("Faucet" in self.obj_state_dict) and (self.obj_state_dict["Faucet"]["Toggled"] == None):
            self.obj_state_dict["Faucet"]['Toggled'] = False

        return self.obj_state_dict

    def parse_action_to_state(self, single_action_pointer, entire_action):
        #self.obj_state_dict = {'On': None, 'Clean': None, 'Sliced': None, 'Pickedup': None, 'Cooked': None, 'Boiled': None, 'Coffee': None, 'Watered': None, 'Toggled': None}
        single_action = entire_action[single_action_pointer]
        teach_action = single_action[0]
        obj = single_action[1]

        film_action = teach_actions_2_FILM_actions_dict[teach_action]
        #call_fn = self.action_2_function[film_action]

        if teach_action == 'Pickup':
            self._pickup(obj)

        elif teach_action == 'Place':
            self._put_history(obj)

        elif teach_action == 'Slice':
            self._slice_x(obj)

        elif teach_action == 'ToggleOn': #Facuet toggled on and off #Previous action was faucet on
            self._toggle_on_x(obj)

        elif teach_action == 'ToggleOff':
            self._toggle_off_x(obj)

        elif teach_action == 'Pour':
            self._pour_x(obj)
        
        #return self.obj_state_dict

    def actions_until_desired_state(self):
        #First get states that don't match
        #self.sem_exp.print_log("self type is ", self.task_type)
        self.unmatched_states, self.slice_needs_to_happen = self._all_unmatched_states()
        #self.sem_exp.print_log("object state dict is ", self.obj_state_dict)
        #self.sem_exp.print_log("Unmatched states are ", self.unmatched_states)
        if len(self.unmatched_states) == 0  and (self.task_type in ['Clean All X', 'Put All X On Y', 'Put All X In One Y']):
            #print("here 2!")
            self.obj_state_dict = {k: copy.deepcopy(DEFAULT_NONE_DICT)  for k in self.obj_state_dict}
            self.init_obj_state_dict = copy.deepcopy(self.obj_state_dict)
            self.unmatched_states, self.slice_needs_to_happen = self._all_unmatched_states()
        
        #print("nope!")
        self.unmatched_objs = list(self.unmatched_states.keys())
        
        if not(self.task_type in ['Boil X', 'Water Plant']):
        #If holding relevant object that is not the final receptacle, first deal with this #With the exception of SLICED
            self._initialize_for_task_types_without_intermediate()

        else:
            ###################
            ## For now, only consider the cases in which intermediate object is already specified (I do not have to find a new intermediate object)
            ####################
            # self._get_intermediate_obj()
            # if self.intermediate_obj == None:
            #     #JUST RETURN SOMETHING MEANINGLESS
            #     if self.task_type == "Water Plant":
            #         if self.picked_up_obj != None:
            #             self._put_future(default_put_place, write_action=True)
            #         self.obj_state_dict["Bowl"] = copy.deepcopy(DEFAULT_NONE_DICT)
            #         self._pickup("Bowl", write_action=True)
            #         self.intermediate_obj = "Bowl"
            #     elif self.task_type == "Boil X":
            #         if self.boil_microwave_or_stove(self.c_dh, self.d_dh):
            #             self.intermediate_obj = "Pot"
            #             self.obj_state_dict["Pot"] = copy.deepcopy(DEFAULT_NONE_DICT)
            #         else:
            #             self.intermediate_obj = "Bowl"
            #         self.obj_state_dict["Bowl"] = copy.deepcopy(DEFAULT_NONE_DICT)
            self._get_intermediate_obj()
            if self.intermediate_obj == None:
                #JUST RETURN SOMETHING MEANINGLESS
                if self.task_type == "Water Plant":
                    if self.picked_up_obj != None:
                        self._put_future(default_put_place, write_action=True)
                    self.obj_state_dict["Bowl"] = copy.deepcopy(DEFAULT_NONE_DICT)
                    self._pickup("Bowl", write_action=True)
                    self.intermediate_obj = "Bowl"
                    if self.intermediate_obj_gpt != None:
                        self.intermediate_obj = self.intermediate_obj_gpt
                    # else:
                    #     self.intermediate_obj = "Bowl"
                    
                elif self.task_type == "Boil X":
                    if self.boil_microwave_or_stove(self.c_dh, self.d_dh):
                        self.intermediate_obj = "Pot"
                        self.obj_state_dict["Pot"] = copy.deepcopy(DEFAULT_NONE_DICT)
                    else:
                        self.intermediate_obj = "Bowl"
                    self.obj_state_dict["Bowl"] = copy.deepcopy(DEFAULT_NONE_DICT)

        #Now go through unmatched objs and do what's needed
        #Among the desired states, go in the order of 
        #Clean, FILLED_WITH, Sliced, Cooked, Boiled, On
        if not(self.task_type in ['Boil X', 'Water Plant']):
            for obj in self.unmatched_objs:
                #WATER PLANT IS AN EXCEPTION
                #If already holding this object, pass
                #if self.picked_up_obj != obj: #EXCEPTION: GOAL is "Clean" and already in SinkBasin, 
                #Get first unmatched state
                first_unmatched_state = self._get_first_unmatched_state(obj)
                
                #print("picked obj here: ", self.picked_up_obj)
                if self._decide_if_pickup(obj, first_unmatched_state):
                    self._pickup(obj, write_action=True)

                #Now go through the states of this obj in the order of Clean, FILLED_WITH, Sliced, Cooked, Boiled, On
                if self.desired_object_states[obj]["Clean"] and not(self.obj_state_dict[obj]["Clean"]):
                    self._clean_x(obj, write_action=True)

                if self.desired_object_states[obj]["FILLED_WITH"] != None and not(self.obj_state_dict[obj]["FILLED_WITH"] == self.desired_object_states[obj]["FILLED_WITH"]):
                    #print("Here 1")
                    if self.obj_state_dict[obj]["FILLED_WITH"] == "Water" and self.desired_object_states[obj]["FILLED_WITH"] == "Coffee":
                        #if not(self.picked_up_obj == obj): #ALREADY DONE IN DECIDE IF PICKUP
                        #    self._pickup(obj) 
                        #If holding something other than obj, drop it
                        if self.picked_up_obj != None and self.picked_up_obj!= obj:
                            self._put_future(default_put_place, write_action=True)
                        if self._decide_if_pickup(obj, "FILLED_WITH"):
                            self._pickup(obj, write_action=True)
                        #assert self.picked_up_obj == obj

                        self._pour_x("SinkBasin", write_action=True)
                        self._put_future("CoffeeMachine", write_action=True)
                        self._toggle_on_x("CoffeeMachine", write_action=True)
                        #self._toggle_off_x("CoffeeMachine", write_action=True)

                    elif self.obj_state_dict[obj]["FILLED_WITH"] == None and self.desired_object_states[obj]["FILLED_WITH"] == "Coffee":
                        if not(self.obj_state_dict[obj]["On"] == "CoffeeMachine"):
                            if self.picked_up_obj != None and self.picked_up_obj!= obj:
                                self._put_future(default_put_place, write_action=True)
                            if self._decide_if_pickup(obj, "FILLED_WITH"):
                                self._pickup(obj, write_action=True)
                            #assert self.picked_up_obj == obj
                            self._put_future("CoffeeMachine", write_action=True)
                        self._toggle_on_x("CoffeeMachine", write_action=True)
                        #self._toggle_off_x("CoffeeMachine", write_action=True)
                        
                    elif self.obj_state_dict[obj]["FILLED_WITH"] == None and self.desired_object_states[obj]["FILLED_WITH"] == "Water":
                        if not(self.obj_state_dict[obj]["On"] == "SinkBasin"):
                            if self.picked_up_obj != None and self.picked_up_obj!= obj:
                                self._put_future(default_put_place, write_action=True)
                            if self._decide_if_pickup(obj, "FILLED_WITH"):
                                self._pickup(obj, write_action=True)
                            #assert self.picked_up_obj == obj
                            
                            ########### #toggleoff_sinkbasin
                            self._toggle_off_x("SinkBasin", write_action=True)  
                            ###########
                            self._put_future("SinkBasin", write_action=True)
                        self._toggle_on_x("Faucet", write_action=True)
                        self._toggle_off_x("Faucet", write_action=True)
                        
                    else:
                        raise Exception("Not Implemented!")
                    

                if self.desired_object_states[obj]["Sliced"] and not(self.obj_state_dict[obj]["Sliced"]):
                    #print("Here 2")
                    #Pick up knife if we need to
                    if self.picked_up_obj in ["Knife", "ButterKnife"]:
                        pass
                    else:
                        if self.picked_up_obj != None:
                            self._put_future(default_put_place, write_action=True)
                        self._pickup("Knife", write_action=True)

                    #assert self.picked_up_obj in ["Knife", "ButterKnife"]

                    #Slice
                    self._slice_x(obj, write_action=True)

                    #Ditch Knife
                    self._put_future(default_put_place, write_action=True)

                if self.desired_object_states[obj]["Cooked"] and not(self.obj_state_dict[obj]["Cooked"]):
                    #print("Here 3")
                    if self.picked_up_obj != None and self.picked_up_obj!= obj:
                        self._put_future(default_put_place, write_action=True)
                    if self._decide_if_pickup(obj, "Cooked"):
                        self._pickup(obj, write_action=True)
                    #assert self.picked_up_obj == obj

                    
                    if obj == "Bread":
                        #Then use toaster
                        if not(self.obj_state_dict[obj]["On"] == "Toaster"):
                            ######### DO Not use toggle #############
                            self._toggle_on_x("Toaster", write_action=True)    
                            #################################
                            self._put_future("Toaster", write_action=True)


                    #Use Microwave
                    else:
                        if not(self.obj_state_dict[obj]["On"] == "Microwave"):
                            self._put_future("Microwave", write_action=True)
                        self._toggle_on_x("Microwave", write_action=True)
                        #self._toggle_off_x("Microwave", write_action=True)

                        #If on is remaining, open microave
                        if not(self.desired_object_states[obj]["On"] in ["Microwave", None]):
                            self._open_x("Microwave", write_action=True)

                #if self.desired_object_states[obj]["Boiled"] and not(self.obj_state_dict[obj]["Boiled"]): Don't do this for now

                if self.desired_object_states[obj]["On"]!=None and not(self.obj_state_dict[obj]["On"]==self.desired_object_states[obj]["On"]):
                    #print("Here 4")
                    if self._decide_if_pickup(obj, "On"):
                        self._pickup(obj, write_action=True)
                    #assert self.picked_up_obj == obj
                    self._put_future(self.desired_object_states[obj]["On"], write_action=True)

        else: #if (self.task_type in ['Boil X', 'Water Plant']):
            if self.task_type == "Water Plant":
                #If holding something other than intermediate obj, drop it
                if (self.picked_up_obj != None) and (self.picked_up_obj != self.intermediate_obj):
                    self._put_future(default_put_place, write_action=True) #CHeck if SINKBASIN is better
                if self.picked_up_obj != self.intermediate_obj:
                    self._pickup(self.intermediate_obj, write_action=True)   
                #########################################################           
                # #Pour once
                # self._pour_x("HousePlant", write_action=True)
                ########################################################
                #Now if not filled with water, go to sink and fill with water
                if self.obj_state_dict[self.intermediate_obj]['FILLED_WITH'] != "Water":
                    ########### #toggleoff_sinkbasin
                    self._toggle_off_x("SinkBasin", write_action=True)  
                    ###########
                    self._put_future("SinkBasin", write_action=True)
                    self._toggle_on_x("Faucet", write_action=True)
                    self._toggle_off_x("Faucet", write_action=True)
                    self._pickup(self.intermediate_obj, write_action=True)
                #Now go pour to houseplant
                self._pour_x("HousePlant", write_action=True)

            elif self.task_type == "Boil X":
                #First if holding something other than intermediate obj or X, drop it
                if (self.picked_up_obj != None) and not(self.picked_up_obj in [self.intermediate_obj, self.obj_target]):
                    self._put_future(default_put_place, write_action=True)

                #If already holding obj target
                if self.picked_up_obj == self.obj_target: #There might be another object in the intermediate object, but ignore this
                    if not(self.obj_state_dict[self.obj_target]["On"] == self.intermediate_obj):
                        self._put_future(self.intermediate_obj, write_action=True)
                    else:
                        self._put_future(default_put_place, write_action=True)
                    
                #If already holding intermediate obj and not filled with water
                elif (self.picked_up_obj == "Bowl") and self.obj_state_dict[self.intermediate_obj]['FILLED_WITH'] != "Water":
                    self._pickup("Bowl", write_action=True)
                    ###########
                    self._toggle_off_x("SinkBasin", write_action=True)  
                    ###########
                    self._put_future("SinkBasin", write_action=True)
                    self._toggle_on_x("Faucet", write_action=True)
                    self._toggle_off_x("Faucet", write_action=True)
                    self._pickup("Bowl", write_action=True)
                    self._pour_x(self.intermediate_obj, write_action=True)
                    self._put_future("CounterTop", write_action=True)
                    
                elif (self.picked_up_obj == self.intermediate_obj) and self.obj_state_dict[self.intermediate_obj]['FILLED_WITH'] == "Water":
                    if self.intermediate_obj in ["Pot", "Pan"]: #Stove path
                        if not(self.obj_state_dict[self.intermediate_obj]["On"] == "StoveBurner"):
                            if self.picked_up_obj == None:
                                self._pickup(self.intermediate_obj, write_action=True)
                            self._put_future("StoveBurner", write_action=True)
                        self._toggle_on_x("StoveKnob", write_action=True)
                        #self._toggle_off_x("StoveKnob", write_action=True)


                    else: #Microwave Path
                        if not(self.obj_state_dict[self.intermediate_obj]["On"] == "Microwave"):
                            if self.picked_up_obj == None:
                                self._pickup(self.intermediate_obj, write_action=True)
                            self._put_future("Microwave", write_action=True)
                        self._toggle_on_x("Microwave", write_action=True)
                        #self._toggle_off_x("Microwave", write_action=True)
                    

                #Put obj on intermediate obj
                if self.obj_state_dict[self.obj_target]["On"] != self.intermediate_obj:
                    self._pickup(self.obj_target, write_action=True)
                    self._put_future(self.intermediate_obj, write_action=True)

                #Pick up intermediate_obj if not holding in
                if not(self.obj_state_dict[self.intermediate_obj]["On"] in ["Microwave", "StoveBurner"] and self.obj_state_dict[self.intermediate_obj]["FILLED_WITH"] == "Water"):
                    if self.picked_up_obj != self.intermediate_obj:
                        self._pickup(self.intermediate_obj, write_action=True)


                #Now heat the obj
                if self.intermediate_obj in ["Pot", "Pan"]: #Stove path
                    if not(self.obj_state_dict[self.intermediate_obj]["On"] == "StoveBurner"):
                        if self.picked_up_obj == None:
                            self._pickup(self.intermediate_obj, write_action=True)
                        self._put_future("StoveBurner", write_action=True)
                    self._toggle_on_x("StoveKnob", write_action=True)
                    #self._toggle_off_x("StoveKnob", write_action=True)


                else: #Microwave Path
                    if not(self.obj_state_dict[self.intermediate_obj]["On"] == "Microwave"):
                        if self.picked_up_obj == None:
                            self._pickup(self.intermediate_obj, write_action=True)
                        self._put_future("Microwave", write_action=True)
                    self._toggle_on_x("Microwave", write_action=True)
                    #self._toggle_off_x("Microwave", write_action=True)

                #If intermediate obj does not have water, fil lit with water
                if self.obj_state_dict[self.intermediate_obj]['FILLED_WITH'] != "Water":
                    self._pickup("Bowl", write_action=True)
                    # self._pour_x(self.intermediate_obj, write_action=True)
                    ###########
                    self._toggle_off_x("SinkBasin", write_action=True)  
                    ###########
                    self._put_future("SinkBasin", write_action=True)
                    self._toggle_on_x("Faucet", write_action=True)
                    self._toggle_off_x("Faucet", write_action=True)
                    self._pickup("Bowl", write_action=True)
                    self._pour_x(self.intermediate_obj, write_action=True)
                    self._put_future("CounterTop", write_action=True)                    



            # elif self.task_type == "Boil X":
            #     #First if holding something other than intermediate obj or X, drop it
            #     if (self.picked_up_obj != None) and not(self.picked_up_obj in [self.intermediate_obj, self.obj_target]):
            #         self._put_future(default_put_place, write_action=True)

            #     #If already holding obj target
            #     if self.picked_up_obj == self.obj_target: #There might be another object in the intermediate object, but ignore this
            #         if not(self.obj_state_dict[self.obj_target]["On"] == self.intermediate_obj):
            #             self._put_future(self.intermediate_obj, write_action=True)
            #         else:
            #             self._put_future(default_put_place, write_action=True)
                    
            #     #If already holding intermediate obj and not filled with water
            #     elif (self.picked_up_obj == self.intermediate_obj) and self.obj_state_dict[self.intermediate_obj]['FILLED_WITH'] != "Water":
            #         self._put_future("SinkBasin", write_action=True)
            #         self._toggle_on_x("Faucet", write_action=True)
            #         self._toggle_off_x("Faucet", write_action=True)
            #         self._pickup(self.intermediate_obj, write_action=True)
            #         self._put_future("CounterTop", write_action=True)
                    
            #     elif (self.picked_up_obj == self.intermediate_obj) and self.obj_state_dict[self.intermediate_obj]['FILLED_WITH'] == "Water":
            #         self._put_future("CounterTop", write_action=True)

            #     #Put obj on intermediate obj
            #     if self.obj_state_dict[self.obj_target]["On"] != self.intermediate_obj:
            #         self._pickup(self.obj_target, write_action=True)
            #         self._put_future(self.intermediate_obj, write_action=True)

            #     #Pick up intermediate_obj if not holding in
            #     if not(self.obj_state_dict[self.intermediate_obj]["On"] in ["Microwave", "StoveBurner"] and self.obj_state_dict[self.intermediate_obj]["FILLED_WITH"] == "Water"):
            #         if self.picked_up_obj != self.intermediate_obj:
            #             self._pickup(self.intermediate_obj, write_action=True)

            #     #If intermediate obj does not have water, fil lit with water
            #     if self.obj_state_dict[self.intermediate_obj]['FILLED_WITH'] != "Water":
            #         self._put_future("SinkBasin", write_action=True)
            #         self._toggle_on_x("Faucet", write_action=True)
            #         self._toggle_off_x("Faucet", write_action=True)
            #         self._pickup(self.intermediate_obj, write_action=True)

            #     #Now heat the obj
            #     if self.intermediate_obj in ["Pot", "Pan"]: #Stove path
            #         if not(self.obj_state_dict[self.intermediate_obj]["On"] == "StoveBurner"):
            #             if self.picked_up_obj == None:
            #                 self._pickup(self.intermediate_obj, write_action=True)
            #             self._put_future("StoveBurner", write_action=True)
            #         self._toggle_on_x("StoveKnob", write_action=True)
            #         #self._toggle_off_x("StoveKnob", write_action=True)


            #     else: #Microwave Path
            #         if not(self.obj_state_dict[self.intermediate_obj]["On"] == "Microwave"):
            #             if self.picked_up_obj == None:
            #                 self._pickup(self.intermediate_obj, write_action=True)
            #             self._put_future("Microwave", write_action=True)
            #         self._toggle_on_x("Microwave", write_action=True)
            #         #self._toggle_off_x("Microwave", write_action=True)





        #Set categories_in_inst
        self.categories_in_inst = list(OrderedDict({k[0]: 1 for k in self.future_list_of_highlevel_actions}).keys())


    def _initialize_for_task_types_without_intermediate(self):
        if self.picked_up_obj in self.unmatched_objs:
            #If first unmatched state is Sliced, exception
            if not(self._get_first_unmatched_state(self.picked_up_obj) == 'Sliced'):
                self.unmatched_objs = self._put_first(self.unmatched_objs, self.picked_up_obj)
                return
                
        #If holding knife and there is an obj that needs to be sliced, 
        elif self.picked_up_obj in ["Knife", "ButterKnife"]:
            if self.slice_needs_to_happen:
                #Get the object whose first unmatched is sliced
                for obj in self.unmatched_objs:
                    if self._get_first_unmatched_state(obj) == 'Sliced':
                        self.unmatched_objs = self._put_first(self.unmatched_objs, obj)
                        return

        #If not returned yet (holding object not relevant or the final receptacle), put it down. #e.g. Plate
        if self.picked_up_obj !=None:
            #Just drop picked_up_obj
            self._put_future(self._drop_place(), write_action=True)
        
        return 

    def _get_intermediate_obj(self):
        self.intermediate_obj = None
        if self.task_type == 'Boil X':
            for obj in BOILABLE_OBJS:
                if (obj in self.obj_state_dict):
                    self.intermediate_obj = obj

            #If there is an object that is picked up or has water filled, make these intermediate obj
            for obj in BOILABLE_OBJS:
                if (obj in self.obj_state_dict):
                    if self.obj_state_dict[obj]["Pickedup"]:
                        self.intermediate_obj = obj

            for obj in BOILABLE_OBJS:
                if (obj in self.obj_state_dict):
                    if self.obj_state_dict[obj]["FILLED_WITH"] == "Water":
                        self.intermediate_obj = obj
            
            for obj in BOILABLE_OBJS:
                if (obj in self.obj_state_dict):
                    if self.obj_state_dict["Potato"]["On"] == obj:
                        self.intermediate_obj = obj


        elif self.task_type == 'Water Plant':
            FILLABLE_OBJS_WITHOUT_HOUSEPLANT = [f for f in FILLABLE_OBJS if not(f == "HousePlant")]
            for obj in FILLABLE_OBJS_WITHOUT_HOUSEPLANT:
                if (obj in self.obj_state_dict):
                    self.intermediate_obj = obj

            #If there is an object that is picked up or has water filled, make these intermediate obj
            for obj in FILLABLE_OBJS_WITHOUT_HOUSEPLANT:
                if (obj in self.obj_state_dict):
                    if self.obj_state_dict[obj]["Pickedup"]:
                        self.intermediate_obj = obj

            for obj in FILLABLE_OBJS_WITHOUT_HOUSEPLANT:
                if (obj in self.obj_state_dict):
                    if self.obj_state_dict[obj]["FILLED_WITH"] == "Water":
                        self.intermediate_obj = obj

    #def _initialize_for_task_types_with_intermediate(self):



    def _drop_place(self):
        if self.task_type in ['Water Plant', 'Boil X', 'Breakfast', 'Coffee', 'N Cooked Slices Of X In Y', 'N Slices Of X In Y', 'Plate Of Toast', "Salad", "Sandwich", "Clean All X"]:
            return default_put_place
        else:
            #print("Drop here!")
            return self.parent_target

    def _get_first_unmatched_state(self, obj):
        #Clean, FILLED_WITH, Sliced, Cooked, Boiled, On
        for state in ["Clean", "FILLED_WITH", "Sliced", "Cooked", "Boiled", "On"]:
            if self._unmatched(obj, state):
                return state


    def _all_unmatched_states(self):
        self.unmatched_states = {}
        self.slice_needs_to_happen = False
        for obj in self.desired_object_states:
            for state in self.desired_object_states[obj]:
                if self._unmatched(obj, state):
                    if not (obj in self.unmatched_states):
                        self.unmatched_states[obj] = []
                    self.unmatched_states[obj].append(state)
                    if state == "Sliced":
                        self.slice_needs_to_happen = True
        return self.unmatched_states, self.slice_needs_to_happen

    #Really for the final state
    def _unmatched(self, obj, state):
        if self.desired_object_states[obj][state] !=None and (self.obj_state_dict[obj][state] != self.desired_object_states[obj][state]):
            return True
        #elif self.desired_object_states[obj][state] == "Coffee" and 
        else:
            return False

    #For task types that use intermediate receptacles: Boil X, Water Plant
    # def _unmatched_intermediate(self):
    #     self.unmatched_states_intermediate = {}
    #     if self.task_type == 'Boil X':
    #         #See if there exists a FILLABLE_Obj that is filled with water
    #         for obj in BOILABLE_OBJS:
    #             if obj in 

    #         #See if there exists a FILLABLE obj that is 
    #     elif self.task_type == 'Water Plant':
    #         #See if there exists a FILLABLE_Obj that is filled with water
    #         for obj in FILLABLE_OBJS:
    #             if (obj in self.obj_state_dict) and (self.obj_state_dict[obj]['FILLED_WITH'] == "Water"):



    
    def _decide_if_pickup(self, obj, first_unmatched_state):
        return self.picked_up_obj == None and self._decide_if_pickup_inner(obj, first_unmatched_state)


    def _decide_if_pickup_inner(self, obj, first_unmatched_state):
        if first_unmatched_state == "Clean":
            if self._unmatched(obj, "Clean") and self.obj_state_dict[obj]["On"] == "SinkBasin":
                return False
            else:
                return True

        elif first_unmatched_state == "FILLED_WITH":
            if self._unmatched(obj, "FILLED_WITH"):
                if self.desired_object_states[obj]["FILLED_WITH"] == "Coffee" and self.obj_state_dict[obj]["On"] == "CoffeeMachine":
                    return False
                elif self.desired_object_states[obj]["FILLED_WITH"] == "Water" and self.obj_state_dict[obj]["On"] == "SinkBasin":
                    return False
                else:
                    return True
            #elif self.desired_object_states[obj]["FILLED_WITH"] == None and self.desired_object_states[obj]["FILLED_WITH"] == "Water":
                # if self.task_type == "Water Plant":
                #     return True
                # elif self.task_type == "Coffee":
                #     return True
                # else:
                #     raise Exception()

        elif first_unmatched_state == "Sliced":
            return False
            #if self.picked_up_obj == "Knife":
            #    return False #Need to pick up knife
            #else:
            #    return False

        elif first_unmatched_state == "Cooked": #No need to care about STOVE_BURNER For cooked
            # if (self.obj_state_dict[obj]["On"] == "Microwave") or ((self.obj_state_dict[obj]["On"] in ["Pot", "Pan"]) and  (self.obj_state_dict[obj][self.obj_state_dict[obj]["On"]]["On"] == "StoveBurner")):
            #     return False
            # elif ((self.obj_state_dict[obj]["On"] in ["Pot", "Pan"]) and  not(self.obj_state_dict[obj][self.obj_state_dict[obj]["On"]]["On"] == "StoveBurner")):
            #     return 
            # else:
            #     return True
            if (self.obj_state_dict[obj]["On"] == "Microwave") or (self.obj_state_dict[obj]["On"] == "Toaster"):
                return False
            else:
                return True

        # elif first_unmatched_state == "Boiled":
        #     if self.obj_state_dict[obj]["On"] in BOILABLE_OBJS:
        #         return False
        #     else:
        #         return True

        elif first_unmatched_state == "On":
            #if self._unmatched(obj, "On"):
            return True

        
    def _put_first(self, obj_list, obj):
        obj_idx = obj_list.index(obj)
        return [obj] + obj_list[:obj_idx] + obj_list[obj_idx+1:]

    def _recepY_convert(self, recepY):
        if recepY == 'Sink':
            return 'SinkBasin'
        elif recepY == 'Bathtub':
            return 'BathtubBasin'
        return recepY

    #Actions (Past)
    def _pickup(self, obj , write_action=False):
        
        #assert self.picked_up_obj ==  None
        if not(obj in self.obj_state_dict):
            self.obj_state_dict[obj] = copy.deepcopy(DEFAULT_NONE_DICT)
        self.picked_up_obj = obj
        self.obj_state_dict[obj]['Pickedup'] = True
        self.obj_state_dict[obj]['On'] = None

        if write_action:
            self.future_list_of_highlevel_actions.append((obj, "PickupObject"))
            

    def _put_history(self, obj, write_action=False):
        #assert self.picked_up_obj != None , obj 

        if not(obj in self.obj_state_dict):
            self.obj_state_dict[obj] = copy.deepcopy(DEFAULT_NONE_DICT)
        if not(self.picked_up_obj in self.obj_state_dict):
            self.obj_state_dict[self.picked_up_obj] = copy.deepcopy(DEFAULT_NONE_DICT)
        
        prev_picked_up_obj = self.picked_up_obj 
        self.obj_state_dict[self.picked_up_obj]['Pickedup'] = False
        self.obj_state_dict[self.picked_up_obj]['On'] = obj
        self.picked_up_obj = None

        #If Putting on SinkBasin and Faucet on, make Filled and clean
        if obj == "SinkBasin" and ("Faucet" in self.obj_state_dict and self.obj_state_dict["Faucet"]['Toggled']):
            for sink_obj in self.get_currently_in_recep_Y("SinkBasin"):
                self.obj_state_dict[sink_obj]["Clean"] = True
                if sink_obj in FILLABLE_OBJS:
                    self.obj_state_dict[sink_obj]["FILLED_WITH"] = "Water"


        #If Putting on StoveBurner, Toaster
        elif obj == "Toaster" and ("Toaster" in self.obj_state_dict and self.obj_state_dict[obj]["Toggled"]):
            #For toaster, get object on stove burner
            self.obj_state_dict[prev_picked_up_obj]["Cooked"] = True

            #For Stoveburner, get object on prev_picked_up_obj
            

        elif obj == "StoveBurner" and ("StoveKnob" in self.obj_state_dict and self.obj_state_dict["StoveKnob"]["Toggled"]):
            #Obj on the stoveburner
            objs_on_pot_pan = self.get_currently_in_recep_Y(prev_picked_up_obj)
            for obj in objs_on_pot_pan:
                self.obj_state_dict[obj]["Cooked"] = "True"

        #If Putting on CoffeeMachine
        elif obj == "CoffeeMachine" and ("CofeeMachine" in self.obj_state_dict and self.obj_state_dict[obj]["Toggled"]):
            self.obj_state_dict[prev_picked_up_obj]["FILLED_WITH"] = "Coffee"

        #print("Put obj is ", obj)
        #print("Defulat put place is ", default_put_place)
        if write_action:
            self.future_list_of_highlevel_actions.append((obj, "PutObject"))


    def _put_future(self, obj, write_action=True):
        self.caution_pointers.append(len(self.future_list_of_highlevel_actions))
        write_action=True
        if obj in OPENABLE_CLASS_LIST:
            self._open_x(obj, write_action=True)
        self._put_history(obj, write_action=True)
        #if obj in OPENABLE_CLASS_LIST:
        #    self._close_x(obj,write_action=True)


    #def _cook_x(self):
    #    pass

    #def _boil_x(self):
    #    pass

    def _slice_x(self, obj, write_action=False):
        if not(obj in self.obj_state_dict):
            self.obj_state_dict[obj] = copy.deepcopy(DEFAULT_NONE_DICT)
        #Pick up knife if not already pickedup  #Skip for past
        self.obj_state_dict[obj]['Sliced'] = True
        if write_action:
            self.future_list_of_highlevel_actions.append((obj, "SliceObject"))

    def _toggle_on_x(self, obj, write_action=False):
        self.caution_pointers.append(len(self.future_list_of_highlevel_actions))  ###########################333
        if not(obj in self.obj_state_dict):
            self.obj_state_dict[obj] = copy.deepcopy(DEFAULT_NONE_DICT)
        if obj =='Faucet':
            clean_objs = self.get_currently_in_recep_Y('SinkBasin') #It's always Sink not SinkBasin in TEACH history_subgoals
            #print("CLEAN objs is ", clean_objs)
            for clean_obj in clean_objs:
                self.obj_state_dict[clean_obj]['Clean'] = True
                if clean_obj in FILLABLE_OBJS:
                    self.obj_state_dict[clean_obj]['FILLED_WITH'] = "Water"
            self.obj_state_dict[obj]['Toggled'] = True

        elif obj in ['Toaster', 'Microwave', 'StoveKnob']:
            self.caution_pointers.append(len(self.future_list_of_highlevel_actions))     #####################JY
            recep_obj = obj
            if obj == 'StoveKnob':
                recep_obj = 'StoveBurner'
            if not (recep_obj == 'StoveBurner'):
                cooked_objs = self.get_currently_in_recep_Y(recep_obj) 
            else:
                on_stove_objs = self.get_currently_in_recep_Y(recep_obj) 
                cooked_objs = []
                for on_stove_obj in on_stove_objs:
                    cooked_objs += self.get_currently_in_recep_Y(on_stove_obj)
            for cooked_obj in cooked_objs:
                self.obj_state_dict[cooked_obj]['Cooked'] = True
        
            self.obj_state_dict[obj]['Toggled'] = True

        elif obj == "CoffeeMachine":
            
            coffee_objs = self.get_currently_in_recep_Y(obj)
            #assert len(coffee_objs) <= 1
            for coffee_obj in coffee_objs:
                if (coffee_obj in FILLABLE_OBJS) and self.obj_state_dict[coffee_obj]['FILLED_WITH'] == None:
                    self.obj_state_dict[coffee_obj]['FILLED_WITH'] = 'Coffee'

        #else:
        #if    raise Exception("Toggled") #No need to care about other objects
        if write_action:
            self.future_list_of_highlevel_actions.append((obj, "ToggleObjectOn"))


    def _toggle_off_x(self, obj, write_action=False):
        self.caution_pointers.append(len(self.future_list_of_highlevel_actions))
        if not(obj in self.obj_state_dict):
            self.obj_state_dict[obj] = copy.deepcopy(DEFAULT_NONE_DICT)
        self.obj_state_dict[obj]['Toggled'] = False
        if write_action:
            self.future_list_of_highlevel_actions.append((obj, "ToggleObjectOff"))

    def _pour_x(self, obj, write_action=False):
        #############################3
        # self.caution_pointers.append(len(self.future_list_of_highlevel_actions))
        ############################3
        if not(obj in self.obj_state_dict):
            self.obj_state_dict[obj] = copy.deepcopy(DEFAULT_NONE_DICT)
        #assert self.picked_up_obj in FILLABLE_OBJS
        if obj in FILLABLE_OBJS:
            self.obj_state_dict[obj]['FILLED_WITH'] = copy.deepcopy(self.obj_state_dict[self.picked_up_obj]['FILLED_WITH'])
        self.obj_state_dict[self.picked_up_obj]['FILLED_WITH'] = None
        if write_action:
            self.future_list_of_highlevel_actions.append((obj, "PourObject"))

    def _clean_x(self, obj, write_action=False):
        #Put on SinkBasin, Toggle on and off
        #Assume that obj already picked up
        #assert self.picked_up_obj == obj
        if not (self.obj_state_dict[obj]["On"] == "SinkBasin"):
#################################################################################
            self._toggle_off_x("SinkBasin", write_action=write_action) #toggleoff_sinkbasin
##################################################################################            
            self._put_future("SinkBasin", write_action=write_action)
        self._toggle_on_x("Faucet", write_action=write_action)
        self._toggle_off_x("Faucet", write_action=write_action)
    ########################################################################
        if not (self.task_type == 'Plate Of Toast')  :
            self._pickup(obj, write_action=write_action)
######################################################################3
        if self.task_type in ["Coffee", "Breakfast"]:
            self._pour_x("SinkBasin",  write_action=write_action)


    #def _water_plant(self):

    #def _water_x(self):

    def _open_x(self, obj, write_action=False):
        if not(obj in self.obj_state_dict):
            self.obj_state_dict[obj] = copy.deepcopy(DEFAULT_NONE_DICT)
        self.obj_state_dict[obj]['Open'] = True
        if write_action:
            self.future_list_of_highlevel_actions.append((obj, "OpenObject"))

    def _close_x(self, obj, write_action=False):
        if not(obj in self.obj_state_dict):
            self.obj_state_dict[obj] = copy.deepcopy(DEFAULT_NONE_DICT)
        self.obj_state_dict[obj]['Open'] = False
        if write_action:
            self.future_list_of_highlevel_actions.append((obj, "CloseObject"))

    #Actions Future



#This is off the shelf
#Just added Pour from Teach
def determine_consecutive_interx(list_of_actions, previous_pointer, sliced=False):
    returned, target_instance = False, None
    if previous_pointer <= len(list_of_actions)-1:
        if list_of_actions[previous_pointer][0] == list_of_actions[previous_pointer+1][0]:
            returned = True
            #target_instance = list_of_target_instance[-1] #previous target
            target_instance = list_of_actions[previous_pointer][0]
        #Micorwave or Fridge
        elif list_of_actions[previous_pointer][1] == "OpenObject" and list_of_actions[previous_pointer+1][1] == "PickupObject":
            returned = True
            #target_instance = list_of_target_instance[0]
            target_instance = list_of_actions[previous_pointer+1][0]
            #if sliced: #THESE DON'T HAPPEN IN TEACH
            #    #target_instance = list_of_target_instance[3]
            #    target_instance = list_of_actions[3][0]
        #Micorwave or Fridge
        elif list_of_actions[previous_pointer][1] == "PickupObject" and list_of_actions[previous_pointer+1][1] == "CloseObject":
            returned = True
            #target_instance = list_of_target_instance[-2] #e.g. Fridge
            target_instance = list_of_actions[previous_pointer-1][0]
        #Faucet
        elif list_of_actions[previous_pointer+1][0] == "Faucet" and list_of_actions[previous_pointer+1][1] in ["ToggleObjectOn", "ToggleObjectOff"]:
            returned = True
            target_instance = "Faucet"
        #Pick up after faucet 
        elif list_of_actions[previous_pointer][0] == "Faucet" and list_of_actions[previous_pointer+1][1] == "PickupObject":
            returned = True
            #target_instance = list_of_target_instance[0]
            target_instance = list_of_actions[previous_pointer+1][0]
            #if sliced: #THESE DON'T HAPPEN IN TEACH
            #    #target_instance = list_of_target_instance[3]
            #    target_instance = list_of_actions[3][0]
        #NEWLY ADDED: Picked up from the sink and now Pouring in Sink
        elif list_of_actions[previous_pointer][1] == "PickupObject" and (list_of_actions[previous_pointer+1][1] == "PourObject" and list_of_actions[previous_pointer+1][0] == "SinkBasin") : # and self.task_type  != 'Boil X':
            returned = True
            target_instance = "SinkBasin"
            
            
        ################################################################################################
        # elif list_of_actions[previous_pointer][0] == "Toaster" and list_of_actions[previous_pointer][1] == "ToggleObjectOn" \
        #     and (list_of_actions[previous_pointer+1][0] == "Bread" or list_of_actions[previous_pointer+1][0] == "BreadSliced"):
        #     print("###############Toast Task###########")
        #     print("###############Toast Task###########")
        #     print("###############Toast Task###########")
        #     print("###############Toast Task###########")
        #     print("###############Toast Task###########")
        #     print("###############Toast Task###########")
        #     print("###############Toast Task###########")
        #     print("###############Toast Task###########")
        #     returned = True
        #     target_instance = list_of_actions[previous_pointer+1][0]
        elif list_of_actions[previous_pointer][0] == "Toaster" and list_of_actions[previous_pointer][1] == "PutObject" \
            and (list_of_actions[previous_pointer+1][0] == "Bread" or list_of_actions[previous_pointer+1][0] == "BreadSliced"):
            print("###############Toast Task###########")
            print("###############Toast Task###########")
            print("###############Toast Task###########")
            print("###############Toast Task###########")
            print("###############Toast Task###########")
            print("###############Toast Task###########")
            print("###############Toast Task###########")
            print("###############Toast Task###########")
            returned = True
            target_instance = list_of_actions[previous_pointer+1][0]
        ###############################################################################################            

            
    return returned, target_instance




