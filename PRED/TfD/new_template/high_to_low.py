new_tem = dict()

# Low
new_tem['Coffee'] = [
    # Clean Mug
    ['Mug', 'PickupObject'],
    ['SinkBasin', 'PutObject'],
    ['Faucet', 'ToggleObjectOn'],
    ['Faucet', 'ToggleObjectOff'],
    
    # Pour Mug to SinkBasin
    ['Mug', 'PickupObject'],
    ['SinkBasin', 'PourObject'],
    
    # Put Mug to CoffeeMachine
    ['CoffeeMachine', 'PutObject'],
    
    # ToggleOn CoffeeMachine
    ['CoffeeMachine', 'ToggleObjectOn']
]
# High
new_tem['Coffee'] = [
    ['Clean', 'Mug', 'None'],
    ['Pickup', 'Mug', 'None'],
    ['Pour', 'Mug', 'SinkBasin'],
    ['Put', 'Mug', 'CoffeeMachine'],
    ['ToggleOn', 'CoffeeMachine', 'None']
]





# Low
new_tem['Water Plant'] = [
    # Fill Target (=Clean Target)
    ['Target', 'PickupObject'],
    ['SinkBasin', 'PutObject'],
    ['Faucet', 'ToggleObjectOn'],
    ['Faucet', 'ToggleObjectOff'],
    
    ['Target', 'PickupObject'],
    ['HousePlant', 'PourObject']
]
#High
new_tem['Water Plant'] = [
    ['Fill', 'Target', 'None'],
    ['Pickup', 'Target', 'None'],
    ['Pour', 'Target', 'HousePlant']
]




# Low
new_tem['Boil X'] = [
    # Move Potato to Pot
    ['Potato', 'PickupObject'],
    ['Pot', 'PutObject'],
    
    # Move Pot to StoveBurner
    ['Pot', 'PickupObject'],
    ['StoveBurner', 'PutObject'],
    
    # Fill Bowl
    ['Bowl', 'PickupObject'],
    ['SinkBasin', 'PutObject'],
    ['Faucet', 'ToggleObjectOn'],
    ['Faucet', 'ToggleObjectOff'],
    
    # Move Bowl to Pot
    ['Bowl', 'PickupObject'],
    ['Pot', 'PourObject'],
    
    ['StoveKnob', 'ToggleObjectOn']
]
# High
new_tem['Boil X'] = [
    ['Move', 'Potato', 'Pot'],
    ['Move', 'Pot', 'StoveBurner'],
    ['Fill', 'Target', 'None'],
    ['Move', 'Target', 'Pot'],
    ['ToggleOn', 'StoveKnob', 'None']
]





# Low   
new_tem['Clean All X'] = [
    # Clean Target
    ['Target', 'PickupObject'],
    ['SinkBasin', 'PutObject'],
    ['Faucet', 'ToggleObjectOn'],
    ['Faucet', 'ToggleObjectOff']
]
# High
new_tem['Clean All X'] = [
    ['Move', 'Target', 'SinkBasin'],
    ['ToggleOn', 'Faucet', 'None'],
    ['ToggleOff', 'Faucet', 'None']
]




# Low
new_tem['N Slices Of X In Y'] = [
    # Clean Parent
    ['Parent', 'PickupObject'],
    ['SinkBasin', 'PutObject'],
    ['Faucet', 'ToggleObjectOn'],
    ['Faucet', 'ToggleObjectOff'],
    
    # Move Target to Parent
    ['Target', 'PickupObject'],
    ['Parent', 'PutObject'],
    
    # Cut Target
    ['Knife', 'PickupObject'],
    ['Target', 'SliceObject'],
    ['CounterTop', 'PutObject']     # CounterTop can be changed to the most nearest and large receptacle
]
# High
new_tem['N Slices Of X In Y'] = [
    ['Clean', 'Parent', 'None'],
    ['Move', 'Target', 'Parent'],
    ['Cut', 'Target', 'CounterTop']
]



# Low
new_tem['N Cooked Slices Of X In Y'] = [
    # Clean Parent
    ['Parent', 'PickupObject'],
    ['SinkBasin', 'PutObject'],
    ['Faucet', 'ToggleObjectOn'],
    ['Faucet', 'ToggleObjectOff'],
    
    # (Move Target to Parent)
    ['Target', 'PickupObject'],
    ['Parent', 'PutObject'],
    
    # Cut Target
    ['Knife', 'PickupObject'],
    ['Target', 'SliceObject'],
    ['CounterTop', 'PutObject'],
    
    # Heat Parent
    ['Parent', 'PickupObject'],
    ['Microwave', 'OpenObject'],
    ['Microwave', 'PutObject'],
    ['Microwave', 'CloseObject'],
    ['Microwave', 'ToggleObjectOn'],
    ['Microwave', 'ToggleObjectOff'],
    ['Microwave', 'OpenObject']
]
# High
new_tem['N Cooked Slices Of X In Y'] = [
    ['Clean', 'Parent', 'None'],
    ['Move', 'Target', 'Parent'],
    ['Cut', 'Target', 'CounterTop'],
    ['Heat', 'Parent', 'None']
]





# Low
new_tem['Plate Of Toast'] = [
    # (Cut, Lettuce)
    ['Knife', 'PickupObject'],
    ['Bread', 'SliceObject'],
    ['CounterTop', 'PutObject'],
    
    # Move Bread to Toaster
    ['Bread', 'PickupObject'],
    ['Toaster', 'PutObject'],
    
    ['Toaster', 'ToggleObjectOn'],
    
    # Clean Plate
    ['Plate', 'PickupObject'],
    ['SinkBasin', 'PutObject'],    
    ['Faucet', 'ToggleObjectOn'],
    ['Faucet', 'ToggleObjectOff'],
    
    # (Move Bread to Plate)
    ['Bread', 'PickupObject'],
    ['Plate', 'PutObject']
]
# High
new_tem['Plate Of Toast'] = [
    ['Cut', 'Lettuce', 'CounterTop'],
    ['Move', 'Bread', 'Toaster'],
    ['ToggleOn', 'Toaster', 'None'],
    ['Clean', 'Plate', 'None'],
    ['Move', 'Bread', 'Plate']
]






# Low
new_tem['Put All X In One Y'] = [
    # Move Target to Parent
    ['Target', 'PickupObject'],
    ['Parent', 'PutObject']
]
# High
new_tem['Put All X In One Y'] = [
    ['Move', 'Target', 'Parent']
]





# Low
new_tem['Put All X On Y'] = [
    # Move Target to Parent
    ['Target', 'PickupObject'],
    ['Parent', 'PutObject']
]
# High
new_tem['Put All X On Y'] = [
    ['Move', 'Target', 'Parent']
]



# Salad Low
new_tem['Salad'] = [
    # (Clean, Plate)
    ['Plate', 'PickupObject'],
    ['SinkBasin', 'PutObject'],
    ['Faucet', 'ToggleObjectOn'],
    ['Faucet', 'ToggleObjectOff'],

    # (Cut, Lettuce)
    ['Knife', 'PickupObject'],
    ['Lettuce', 'SliceObject'],
    ['CounterTop', 'PutObject'],

    # Move, Lettuce, Plate
    ['Lettuce', 'PickupObject'],
    ['Plate', 'PutObject'],

    # Cut, Tomato
    ['Knife', 'PickupObject'],
    ['Tomato', 'SliceObject'],
    ['CounterTop', 'PutObject'],

    # Move, Tomato, Plate
    ['Tomato', 'PickObject'],
    ['Plate', 'PutObject'],

    # Heat, Tomato
    ['Potato', 'PickObject'],
    ['Microwave', 'OpenObject'],
    ['Microwave', 'PutObject'],
    ['Microwave', 'CloseObject'],
    ['Microwave', 'ToggleObjectOn'],
    ['Microwave', 'ToggleObjectOff'],
    ['Microwave', 'OpenObject'],

    # Move Potato, CounterTop
    ['Potato', 'PickupObject'],
    ['CounterTop', 'PutObject'],

    # cut, Potato
    ['Knife', 'PickupObject'],
    ['Potato', 'SliceObject'],
    ['CounterTop', 'PutObject'],

    # Move Potato, Plate
    ['Potato', 'PickObject'],
    ['Plate', 'PutObject']
]


new_tem['Salad'] = [
    # (Clean, Plate)
    ['Plate', 'PickupObject', ['Clean', 'Plate'], 0],
    ['SinkBasin', 'PutObject', ['Clean', 'Plate'], 0],
    ['Faucet', 'ToggleObjectOn', ['Clean', 'Plate'], 0],
    ['Faucet', 'ToggleObjectOff', ['Clean', 'Plate'], 0],

    # (Cut, Lettuce)
    ['Knife', 'PickupObject', ['Cut', 'Lettuce'], 1],
    ['Lettuce', 'SliceObject', ['Cut', 'Lettuce'], 1],
    ['CounterTop', 'PutObject', ['Cut', 'Lettuce'], 1],

    # Move, Lettuce, Plate
    ['Lettuce', 'PickupObject', ['Move', 'Lettuce', 'Plate'], 2],
    ['Plate', 'PutObject', ['Move', 'Lettuce', 'Plate'], 2],

    # Cut, Tomato
    ['Knife', 'PickupObject', ['Cut', 'Tomato'], 3],
    ['Tomato', 'SliceObject', ['Cut', 'Tomato'], 3],
    ['CounterTop', 'PutObject', ['Cut', 'Tomato'], 3],

    # Move, Tomato, Plate
    ['Tomato', 'PickObject', ['Move', 'Tomato', 'Plate'], 4],
    ['Plate', 'PutObject', ['Move', 'Tomato', 'Plate'], 4],

    # Heat, Potato
    ['Potato', 'PickObject', ['Heat', 'Potato'], 5],
    ['Microwave', 'OpenObject', ['Heat', 'Potato'], 5],
    ['Microwave', 'PutObject', ['Heat', 'Potato'], 5],
    ['Microwave', 'CloseObject', ['Heat', 'Potato'], 5],
    ['Microwave', 'ToggleObjectOn', ['Heat', 'Potato'], 5],
    ['Microwave', 'ToggleObjectOff', ['Heat', 'Potato'], 5],
    ['Microwave', 'OpenObject', ['Heat', 'Potato'], 5],

    # Move Potato, CounterTop
    ['Potato', 'PickupObject', ['Move', 'Potato', 'CounterTop'], 6],
    ['CounterTop', 'PutObject', ['Move', 'Potato', 'CounterTop'], 6],

    # Cut, Potato
    ['Knife', 'PickupObject', ['Cut', 'Potato'], 7],
    ['Potato', 'SliceObject', ['Cut', 'Potato'], 7],
    ['CounterTop', 'PutObject', ['Cut', 'Potato'], 7],

    # Move Potato, Plate
    ['Potato', 'PickObject', ['Move', 'Potato', 'Plate'], 8],
    ['Plate', 'PutObject', ['Move', 'Potato', 'Plate'], 8]
]

# Salad High
new_tem['Salad'] = [
    ['Clean', 'Plate', 'None'],
    ['Cut', 'Lettuce', 'CounterTop'],
    ['Move', 'Lettuce', 'Plate'],
    ['Cut', 'Tomato', 'CounterTop'],
    ['Move', 'Tomato', 'Plate'],
    ['Heat', 'Potato', 'None'],
    ['Move', 'Potato', 'CounterTop'],
    ['Cut', 'Potato', 'CounterTop'],
    ['Move', 'Potato', 'Plate']
]



# Sandwich Low
new_tem['Sandwich'] = [
    # Cut Bread
    ['Knife', 'PickupObject'],
    ['Bread', 'SliceObject'],
    ['CounterTop', 'PutObject'],
    
    # Move Bread to Toaster
    ['Bread', 'PickupObject'],
    ['Toaster', 'PutObject'],
    
    ['Toaster', 'ToggleObjectOn'],

    # (Clean, Plate)
    ['Plate', 'PickupObject'],
    ['SinkBasin', 'PutObject'],
    ['Faucet', 'ToggleObjectOn'],
    ['Faucet', 'ToggleObjectOff'],

    # Move Bread to Plate
    ['Bread', 'PickupObject'],
    ['Plate', 'PutObject'],

    # (Cut, Lettuce)
    ['Knife', 'PickupObject'],
    ['Lettuce', 'SliceObject'],
    ['CounterTop', 'PutObject'],

    # Move, Lettuce, Plate
    ['Lettuce', 'PickupObject'],
    ['Plate', 'PutObject'],

    # Cut, Tomato
    ['Knife', 'PickupObject'],
    ['Tomato', 'SliceObject'],
    ['CounterTop', 'PutObject'],

    # Move, Tomato, Plate
    ['Tomato', 'PickObject'],
    ['Plate', 'PutObject']
]



# Sandwich High
new_tem['Sandwich'] = [
    # Cut Bread
    ['Cut', 'Bread', 'CounterTop'],

    # Move Bread to Toaster
    ['Move', 'Bread', 'Toaster'],
    
    ['ToggleOn', 'Toaster', 'None'],

    # (Clean, Plate)
    ['Clean', 'Plate', 'None'],

    # Move Bread to Plate
    ['Move', 'Bread', 'Plate'],

    # (Cut, Lettuce)
    ['Cut', 'Lettuce', 'CounterTop'],

    # Move, Lettuce, Plate
    ['Move', 'Lettuce', 'Plate'],

    # Cut, Tomato
    ['Cut', 'Tomato', 'CounterTop'],

    # Move, Tomato, Plate
    ['Move', 'Tomato', 'Plate']
]





# Breakfast Low
new_tem['Breakfast'] = [
    # Clean Mug
    ['Mug', 'PickupObject'],
    ['SinkBasin', 'PutObject'],
    ['Faucet', 'ToggleObjectOn'],
    ['Faucet', 'ToggleObjectOff'],
    
    ['Mug', 'PickupObject'],
    
    ['SinkBasin', 'PourObject'],
    
    ['CoffeeMachine', 'PutObject'],
    
    ['CoffeeMachine', 'ToggleObjectOn'],
    
    # Cut Bread
    ['Knife', 'PickupObject'],
    ['Bread', 'SliceObject'],
    ['CounterTop', 'PutObject'],
    
    # Move Bread to Toaster
    ['Bread', 'PickupObject'],
    ['Toaster', 'PutObject'],
    
    ['Toaster', 'ToggleObjectOn'],

    # (Clean, Plate)
    ['Plate', 'PickupObject'],
    ['SinkBasin', 'PutObject'],
    ['Faucet', 'ToggleObjectOn'],
    ['Faucet', 'ToggleObjectOff'],

    # Move Bread to Plate
    ['Bread', 'PickupObject'],
    ['Plate', 'PutObject'],

    # (Cut, Lettuce)
    ['Knife', 'PickupObject'],
    ['Lettuce', 'SliceObject'],
    ['CounterTop', 'PutObject'],

    # Move, Lettuce, Plate
    ['Lettuce', 'PickupObject'],
    ['Plate', 'PutObject'],

    # Cut, Tomato
    ['Knife', 'PickupObject'],
    ['Tomato', 'SliceObject'],
    ['CounterTop', 'PutObject'],

    # Move, Tomato, Plate
    ['Tomato', 'PickObject'],
    ['Plate', 'PutObject']
]
# Breakfast High
new_tem['Breakfast'] = [
    # Clean Mug
    ['Clean', 'Mug', 'None'],

    ['Pickup', 'Mug', 'None'],
    
    ['Pour', 'Mug', 'SinkBasin'],
    
    ['Put', 'Mug', 'CoffeeMachine'],    

    ['ToggleOn', 'CoffeeMachine', 'None'],

    # Cut Bread
    ['Cut', 'Bread', 'CounterTop'],
    
    # Move Bread to Toaster
    ['Move', 'Bread', 'Toaster'],
    
    ['ToggleOn', 'Toaster', 'None'],

    # (Clean, Plate)
    ['Clean', 'Plate', 'None'],

    # Move Bread to Plate
    ['Move', 'Bread', 'Plate'],

    # (Cut, Lettuce)
    ['Cut', 'Lettuce', 'CounterTop'],

    # Move, Lettuce, Plate
    ['Move', 'Lettuce', 'Plate'],

    # Cut, Tomato
    ['Cut', 'Tomato', 'CounterTop'],

    # Move, Tomato, Plate
    ['Move', 'Tomato', 'Plate']
]

sliced = []
def high_to_low(act, highlevel_idx, ext_info):
    Targets = list(ext_info.keys())
    new_actions = []
    if act[0] == "Clean":
        new_actions.append([act[1], 'PickupObject', act, highlevel_idx])
        new_actions.append(['SinkBasin', 'PutObject', act, highlevel_idx])
        new_actions.append(['Faucet', 'ToggleObjectOn', act, highlevel_idx])
        new_actions.append(['Faucet', 'ToggleObjectOff', act, highlevel_idx])

    elif act[0] == "Fill":
        new_actions.append([act[1], 'PickupObject', act, highlevel_idx])
        new_actions.append(['SinkBasin', 'PutObject', act, highlevel_idx])
        new_actions.append(['Faucet', 'ToggleObjectOn', act, highlevel_idx])
        new_actions.append(['Faucet', 'ToggleObjectOff', act, highlevel_idx])

    elif act[0] == "Heat":
        new_actions.append([act[1], 'PickupObject', act, highlevel_idx])
        new_actions.append(['Microwave', 'OpenObject', act, highlevel_idx])
        new_actions.append(['Microwave', 'PutObject', act, highlevel_idx])
        new_actions.append(['Microwave', 'CloseObject', act, highlevel_idx])
        new_actions.append(['Microwave', 'ToggleObjectOn', act, highlevel_idx])
        new_actions.append(['Microwave', 'ToggleObjectOff', act, highlevel_idx])
        new_actions.append(['Microwave', 'OpenObject', act, highlevel_idx])
    
    elif act[0] == "Cut":
        new_actions.append(['Knife', 'PickupObject', act, highlevel_idx])
        new_actions.append([act[1], 'SliceObject', act, highlevel_idx])
        new_actions.append(['CounterTop', 'PutObject', act, highlevel_idx])
        sliced.append(act[1])
    
    elif act[0] == "Move":
        target = act[1]
        quantity = 1
        if target in Targets:
            if target in sliced:
                if "quantity_of_slices" in ext_info[target]:
                    if ext_info[target]["quantity_of_slices"] != "X":
                        if ext_info[target]["quantity_of_slices"] in ["all", "All"]: quantity = 3
                        else:
                            try:
                                quantity = int(ext_info[target]["quantity_of_slices"])
                            except:
                                quantity = 3
            else:
                if "quantity" in ext_info[target]:
                    if ext_info[target]["quantity"] != "X":
                        if ext_info[target]["quantity"] in ["all", "All"]: quantity = 3
                        else: quantity = int(ext_info[target]["quantity"])            
        for _ in range(quantity):
            new_actions.append([act[1], 'PickupObject', act, highlevel_idx])
            new_actions.append([act[2], 'PutObject', act, highlevel_idx])
        
    elif act[0] == "Pickup":
        new_actions.append([act[1], 'PickupObject', act, highlevel_idx])
    
    elif act[0] == "Put":
        new_actions.append([act[2], 'PutObject', act, highlevel_idx])
        
    elif act[0] == "Pour":
        new_actions.append([act[2], 'PourObject', act, highlevel_idx])
        
    elif act[0] == "ToggleOn":
        new_actions.append([act[1], 'ToggleObjectOn', act, highlevel_idx])

    elif act[0] == "ToggleOff":
        new_actions.append([act[1], 'ToggleObjectOff', act, highlevel_idx])
        
    return new_actions