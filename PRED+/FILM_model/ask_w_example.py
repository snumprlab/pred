import openai
import os
import json
import random
import ast

# openai.api_key = 

base_path = os.path.join(os.path.dirname(__file__),"unique_lists")

##########################################################################################################
instruction_Common = f'''
You need to determine whether to revise the action sequence to solve the task considering 'GUIDE' and then write down the final sequence of actions if needed.
This action sequence is made for solving a household task. Each action's format is a tuple.
The possible action spaces are as follows:
['Target', 'PickupObject'], ['Target', 'PutObject'], ['Target', 'OpenObject'], ['Target', 'CloseObject'], ['Target', 'ToggleObjectOn'], ['Target', 'ToggleObjectOff'], ['Target', 'PourObject'], ['Target', 'SliceObject']
For example, ['Target', 'PickupObject'] is a command to pick up the target.
Modify the action sequence by only adding or subtracting an action at the very beginning if needed.
Do not remove or alter any actions in the sequence that follow the initial action(s).
Write down the final action sequence exactly as provided after making any necessary changes.
Do not add additional explanation. Just write the final action sequence in the correct format (tuples in list).
'''

instruction_Common = f'''
You need to determine whether to revise the action sequence to solve the task considering 'GUIDE' and then write down the final sequence of actions if needed.
This action sequence is made for solving household task. Each action's format is tuple.
The possible action spaces are as follows.
['Target', 'PickupObject'], ['Target', 'PutObject'], ['Target', 'OpenObject'], ['Target', 'CloseObject'], ['Target', 'ToggleObjectOn'], ['Target', 'ToggleObjectOff'], ['Target', 'PourObject'], ['Target', 'SliceObject']
For example, ['Target', 'PickupObject'] is a command to pick up the target.
Modify the action sequence by adding or subtracting action to suit the situation if needed. If it is not needed to be revised, just write down given action sequence.
In this context, 'action' refers to each element in tuple format in the list(action sequence).
You can add or subtract an action at the very first part of the action sequence if needed. Do not modify the actions that follow, and write them down as originally provided.
Do not add additional explannation. Just write the final action sequence in the right format (tuples in list).
'''

instruction_Common_triplet = f'''
You need to determine whether to revise the action sequence to solve the task considering 'GUIDE' and then write down the final sequence of actions if needed.
This action sequence is made for solving household task. Each action's format is triplet.
There are two options.
First, if you think that the first action is not needed, remove that action(triplet) and write down actions after that.
You can only revise the first action; do not modify the actions (from the second actions) that follow, and write them down as originally provided.
In this context, 'action' refers to each element in triplet format in the list(action sequence).
Second, if you still think the first action is necessary, write it down exactly as received.
Do not add additional explanation. Just write the final action sequence in the right format (triplets in list).
'''

##########################################################################################################
instruction_GUIDE_Clean_Fill = f'''
GUIDE: Our purpose is to solve the task efficiently.
Sometimes, there are already cleaned or filled object.
In this case, skipping redundant actions to clean or fill the object makes task solving more efficient.

Current action plan:
'''

instruction_feedback_prompt_Clean_Fill = f'''

The target object in the test image has been either filled or cleaned, as determined by the similarity check.
'''
##########################################################################################################




##########################################################################################################
instruction_GUIDE_RecepObservation = f'''
GUIDE: When you want to pick up an object, it sometimes is located in receptacles that should be opened to pick up the object.
In this case, action sequence contains the sequence of open, pickup(or slice), close.
Your objective is just picking up(or slicing) the object.
If the target object is detected outside, you can pick it up instead of opening receptacles for efficient planning.

'''

instruction_feedback_prompt_RecepObservation = f'''

The target is either directly visible to the agent or its information is already stored on the map.
'''
##########################################################################################################



##########################################################################################################
instruction_GUIDE_PutSink = f'''
GUIDE: If you are about to put something in the Sink or SinkBasin, it is needed to check whether the Faucet is toggled on or not.
If the Faucet is toggled on, it is hard to put the object in the Sink or SinkBasin.

Current action plan:
'''

instruction_feedback_prompt_PutSink = f'''

The faucet is turned on and the sink basin is filled with water, as indicated by the similarity between the test image and the "ToggleOn" reference image.
'''
##########################################################################################################

instruction_GUIDE_OpenMicrowave = f'''
GUIDE: Microwave cannot be opened if the microwave is toggled on.
As a result of checking at this point, the microwave oven is already turned on.
Given the action sequence is as follows.

Current action plan:
'''

instruction_feedback_prompt_OpenMicrowave = f'''
After checking, the microwave's image is confirmed to be in the "ToggleOn" state.
'''


##########################################################################################################
instruction_GUIDE_Repicking = f'''
GUIDE: When picking up an object, it may not always be the intended object.
But you can only handle one object in your hand. Thus, you have to put the object and pick the correct object if you want to pick another object.
Let's assume that you have picked up an object in the first action.
In this case, do not revise the first action, but if you think some actions should be added, add them right after the first action.
Use 'Parent' as the place you put it on. The agent should pick up the 'Target' again if it put it on 'Parent'.

Current action plan:
'''

instruction_feedback_prompt_Repicking = f'''

After checking, the picked up object's mask is detected, but the object is not the desired one as its center coordinates do not fall within the specified range (width between 100 and 200, height between 100 and 230).
'''
##########################################################################################################
instruction_GUIDE_Object_Relationship = f'''
GUIDE: There can be some objects that are already located in the desired destination.
If you think executing the following action should be avoided as it is no longer needed, add a Pass action in triplet form (same as given action) with Pass for action, None for Target and Parent, before the given action (including given action) without further explannation. If you think the following actions are still needed, repeat the given actions.
'''

instruction_feedback_prompt_Object_Relationship = f'''
After checking, it is found that the second object has already been placed in the desired location as indicated by the interaction mask, so no further interaction is needed.
'''


def reask_gpt_w_updated_prompt(dn, example, highactions) :
    done = True
    
    if dn == 'Clean_Fill':
        prompt = instruction_Common_triplet
        prompt += instruction_GUIDE_Clean_Fill
        prompt += example
        prompt += "Current action plan: " + str(highactions)
        prompt += "\n\nFeedback Information: "
        prompt += instruction_feedback_prompt_Clean_Fill

    if dn == 'PutSink':
        prompt = instruction_Common
        prompt += instruction_GUIDE_PutSink
        prompt += example
        prompt += "Current action plan: " +str(highactions)
        prompt += "\n\nFeedback Information: "
        prompt += instruction_feedback_prompt_PutSink

    if dn == 'OpenMicrowave':
        prompt = instruction_Common
        prompt += instruction_GUIDE_OpenMicrowave
        prompt += str(highactions)
        prompt += "\n\nFeedback Information: "
        prompt += instruction_feedback_prompt_OpenMicrowave

    if dn == 'Recep_Observation':
        prompt = instruction_Common
        prompt += instruction_GUIDE_RecepObservation
        prompt += example
        prompt += "Current action plan: " + str(highactions)
        prompt += "\n\nFeedback Information: "
        prompt += instruction_feedback_prompt_RecepObservation
        
    if dn == 'Repicking':
        prompt = instruction_Common
        prompt += instruction_GUIDE_Repicking
        prompt += example
        prompt += "Current action plan: " + str(highactions)
        prompt += "\n\nFeedback Information: "
        prompt += instruction_feedback_prompt_Repicking
        
    if dn == 'Object_Relationship':
        prompt = instruction_Common_triplet
        prompt += instruction_GUIDE_Object_Relationship
        prompt += str(highactions)
        prompt += "\n\nFeedback Information: "
        prompt += instruction_feedback_prompt_Object_Relationship

        
    # prompt = prompt.replace('\n','')
    prompt = prompt.replace("\\", "")
    prompt = prompt.replace('\\', "")
    print(prompt)
    
    first_respond= False
    while(not (first_respond)) :
        try:
            ##### prompt 선택 #####
            # gpt-4
            first_respond = openai.ChatCompletion.create( model="gpt-4o-mini", temperature=0, messages=[{"role": "user", "content": prompt}])
            # gpt-3.5-turbo
            # first_respond = openai.ChatCompletion.create( model="gpt-3.5-turbo", temperature=0, messages=[{"role": "user", "content": prompt}])
            fr = first_respond['choices'][0]['message']['content']
            before = highactions
            try:
                after = ast.literal_eval(fr)
            except:
    
                    print('Before:', before)
                    print('After :', fr)
                    print()

            print('Before:', before)
            print('After :', after)
                
            print("===================================================================================================")
            print()
            
        except AssertionError as e:
            print(e)

    return str(after)
