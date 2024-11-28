# import glob
# import pickle

# templates = ['=== Total + toast_upgrade + put knife change + Clean_plate_Bowl + Put Sink mask === ']


# Task_names = ['Coffee','Water Plant','Boil X', 'Plate Of Toast','Breakfast', 'Clean All X',
#                  'N Cooked Slices Of X In Y', 'N Slices Of X In Y',  'Put All X In One Y', 'Put All X On Y','Salad','Sandwich']

# def f(template, split=None):
#     splits = ['edh_vs_','edh_vus_','tfd_vs_', 'tfd_vus_'] if split is None else [split]
#     for split in splits:
#         #########  parameter initialize  #######
#         success, total = 0, 0
#         task_success =dict()
#         task_tatoal =dict()
#         for task_name in Task_names :            
#             task_success[task_name] = 0
#             task_tatoal[task_name] = 0
#         #########################################
#         game_ids = []
#         for p in glob.glob(f'results/analyze_recs/*{split}*'):
#             result_files = pickle.load(open(p, 'rb'))
#             # print(p ,str(" :    ") ,str(len(result_files)))
#             for result_file in  result_files:
#                 success += result_file['success']
#                 task_success[result_file['task_type']]+=result_file['success']
#                 task_tatoal[result_file['task_type']] += 1
#                 game_ids.append(result_file['edh_instance_true'])
#                 # if result_file['edh_instance_true'] in ['0196f21923a77e88_288f']:
#                 #     print("same")
#             total += len(result_files)
#         ################## print results ##################################
#         print("")
#         print ("----------- total score -----------------")
#         print(split, success, total, (success / total)*100)
#         # print(len(game_ids))
#         # print(len(list(set(game_ids))))
#         visited = set()
#         dup = [x for x in game_ids if x in visited or (visited.add(x) or False)]
#         print(dup)
        
        
#         # print ("----------- task score -----------------")
#         # for task_name in Task_names :  
#         #     if task_tatoal[task_name] !=0 :
#         #         print(task_name, task_success[task_name] , task_tatoal[task_name], (task_success[task_name] / task_tatoal[task_name])*100)
#         #     else : 
#         #         print(task_name,  '0' , '0', '0' , 0)
#         # print()


# for template in templates :
#     print(template)
#     for split in['edh_vs_','edh_vus_','tfd_vs_', 'tfd_vus_']:
#         try:
#             f(template, split)
#         except:
#             pass


import glob
import pickle
import json
with open('new_template/task_type_game_id.json') as f:
    task_type_game_id = json.load(f)

game_id_task_type = dict()
for task in list(task_type_game_id.keys()):
    for id in task_type_game_id[task]:
        game_id_task_type[id] = task

templates = ['======= RED + EDH =======']


Task_names = ['Coffee','Water Plant','Boil X', 'Plate Of Toast','Breakfast', 'Clean All X',
                 'N Cooked Slices Of X In Y', 'N Slices Of X In Y',  'Put All X In One Y', 'Put All X On Y','Salad','Sandwich']

def f(template, split=None):
    splits = ['edh_vs_','edh_vus_','tfd_vs_', 'tfd_vus_','_test'] if split is None else [split]
    for split in splits:
        #########  parameter initialize  #######
        success, total,total_path_len_weight, completed_goal_conditions,total_goal_conditions, path_len_weighted_success_spl,path_len_weighted_goal_condition_spl= 0, 0,0,0,0,0,0
        task_success =dict()
        task_tatoal =dict()
        task_total_path_len_weight = dict()
        task_completed_goal_conditions = dict()
        task_total_goal_conditions = dict()
        task_path_len_weighted_success_spl =dict()
        task_path_len_weighted_goal_condition_spl =dict()
        SR_success_total_path_len_weight=0
        
        for task_name in Task_names :            
            task_success[task_name] = 0
            task_tatoal[task_name] = 0
            task_total_path_len_weight[task_name] = 0.0
            task_completed_goal_conditions[task_name] = 0
            task_total_goal_conditions[task_name] = 0
            task_path_len_weighted_success_spl[task_name] = 0
            task_path_len_weighted_goal_condition_spl[task_name] = 0
        #########################################
        
        for p in glob.glob(f'Results_1/results/analyze_recs/*{split}*'):
            result_files = pickle.load(open(p, 'rb'))
            # print(p ,str(" :    ") ,str(len(result_files)))
            for result_file in  result_files:
                game_id = result_file['game_id']
                success += result_file['success']
                
                task_success[game_id_task_type[game_id]] += result_file['success']
                task_tatoal[game_id_task_type[game_id]] += 1
                task_total_path_len_weight[game_id_task_type[game_id]] += result_file['gt_path_len']
                task_completed_goal_conditions[game_id_task_type[game_id]] +=result_file['completed_goal_conditions']
                task_total_goal_conditions[game_id_task_type[game_id]] += result_file['total_goal_conditions']
                task_path_len_weighted_success_spl[game_id_task_type[game_id]] +=result_file['path_len_weighted_success_spl']
                task_path_len_weighted_goal_condition_spl[game_id_task_type[game_id]] +=result_file['path_len_weighted_goal_condition_spl']
                
                
                
                total_path_len_weight += result_file['gt_path_len']
                SR_success_total_path_len_weight += (result_file['gt_path_len']*result_file['success'])
                completed_goal_conditions += result_file['completed_goal_conditions']
                total_goal_conditions +=result_file['total_goal_conditions']
                path_len_weighted_success_spl +=result_file['path_len_weighted_success_spl']
                path_len_weighted_goal_condition_spl +=result_file['path_len_weighted_goal_condition_spl']

            total += len(result_files)
        ################## print results ##################################
        if total > 0 :
            print ("----------- total score -----------------")
            print(split, success, total)
            print(split, "SR     : ", round((success / total)*100,2), '(',success,'/' ,total,')')
            print(split, "PLW_SR : ", round((path_len_weighted_success_spl / float(total_path_len_weight))*100,2))
            # print(split, "PLW_SR : ", round((path_len_weighted_success_spl / float(SR_success_total_path_len_weight))*100,2))
            print(split, "GC     : ", round((completed_goal_conditions / float(total_goal_conditions))*100,2), '(',completed_goal_conditions,'/' ,total_goal_conditions,')')
            print(split, "PLW_GC : ", round((path_len_weighted_goal_condition_spl / float(total_path_len_weight))*100,2))
            # print ("----------- task score -----------------")
            # for task_name in Task_names :  
            #     if task_tatoal[task_name] !=0 :
            #         print(task_name, "SR     : ", (task_success[task_name] / task_tatoal[task_name])*100)
            #         print(task_name, "PLW_SR : ", (task_path_len_weighted_success_spl[task_name] / float(task_total_path_len_weight[task_name]))*100)
            #         print(task_name, "GC     : ", (task_completed_goal_conditions[task_name] / float(task_total_goal_conditions[task_name]))*100)
            #         print(task_name, "PLW_GC : ", (task_path_len_weighted_goal_condition_spl[task_name] / float(task_total_path_len_weight[task_name]))*100)
            #         # print(task_name, task_success[task_name] , task_tatoal[task_name], (task_success[task_name] / task_tatoal[task_name])*100)
            #     else : 
            #         print(task_name,  '0' , '0', '0' , 0)
            # print()


for template in templates :
    print(template)
    for split in ['edh_vs_','edh_vus_','tfd_vs_', 'tfd_vus_','_test']:
        try:
            f(template, split)
        except:
            pass
