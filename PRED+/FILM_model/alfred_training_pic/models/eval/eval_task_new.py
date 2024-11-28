import os
import json
import numpy as np
from PIL import Image
from datetime import datetime
from eval import Eval
from env.thor_env import ThorEnv
import pickle
import torch
import cv2
from datetime import datetime
import time
from get_annotation_function_eff import get_annotation_instances
import json

class EvalTask(Eval):
    '''
    evaluate overall task performance
    '''

    @classmethod
    def run(cls, model, resnet, task_queue, args, lock, successes, failures, results, process_no, rgb_picture_folder_name, ss_picture_folder_name):
        '''
        evaluation loop
        '''
        # start THOR
        global episode_no
        env = ThorEnv()
        
        episode_no = 0
        ai2thor_objects2idx = pickle.load(open("/home/soyeonm/projects/devendra/segmentation/ai2thor_objects2idx.p", "rb"))
        #if cls.args.small:
        #training_pic_save_task_ids = pickle.load(open('alfred_training_pic_idxes_small.p', 'rb'))
        #else:
        #training_pic_save_task_ids = pickle.load(open('alfred_training_pic_idxes_10.p', 'rb'))
        prev_t_returned = process_no * 10**70
        while True:
            if task_queue.qsize() == 0:
                break

            task = task_queue.get()

            try:                
                #traj = model.load_task_json(task)
                traj = json.load(open('data/json_2.1.0/' + task['task'] + '/pp/ann_' + str(task['repeat_idx']) + '.json'))
                r_idx = task['repeat_idx']
                #r_idx = 1
                print("Evaluating: %s" % (traj['root']))
                print("No. of trajectories left: %d" % (task_queue.qsize()))
                np.random.seed((process_no*10 + episode_no)% 10000 )
                if np.random.choice(10) ==0: #save about 700
                    t_returned = cls.evaluate(env, model, r_idx, resnet, traj, args, lock, successes, failures, results, episode_no, ai2thor_objects2idx, rgb_picture_folder_name, ss_picture_folder_name, prev_t_returned)
                    prev_t_returned += t_returned
                    print("prev t returned is now ", prev_t_returned)
                episode_no +=1
            except Exception as e:
                import traceback
                traceback.print_exc()
                print("Error: " + repr(e))

        # stop THOR
        env.stop()


    

    @classmethod
    def evaluate(cls, env, model, r_idx, resnet, traj_data, args, lock, successes, failures, results, episode_no, ai2thor_objects2idx, rgb_picture_folder_name, ss_picture_folder_name, prev_t_returned):
        
        # reset model
        #model.reset()

        # setup scene
        reward_type = 'dense'
        cls.setup_scene(env, traj_data, r_idx, args, reward_type=reward_type)

        # extract language features
        #feat = model.featurize([traj_data], load_mask=False)

        # goal instr
        goal_instr = traj_data['turk_annotations']['anns'][r_idx]['task_desc']

        done, success = False, False
        fails = 0
        t = prev_t_returned
        print("starting t is ", t)
        t_actions = 0
        #t1 = 0
        reward = 0
        pickle_count= 0
        #print(env.last_event.metadata.keys())
        #episode_no = cls.episode_no
        print("episode no is ", episode_no)
        
        expert_actions = [a['discrete_action'] for a in traj_data['plan']['low_actions']]
        while not done and t_actions < len(expert_actions)-1:
            # break if max_steps reached
            first_pickup = False
            if t - prev_t_returned>= args.max_steps:
                break
            #f t ==0:
            
            
            def write(env, t, interaction = False):
                np.random.seed((t+episode_no*1000) % 10000)
                
                rgb = torch.tensor(env.last_event.frame.copy()).numpy() #shape (h, w, 3)                
                
                if (np.random.choice(4) == 0 or interaction):
                    #flip rgb and do imwrite
                    rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(rgb_picture_folder_name + "rgb_" + str(t) + ".png" , rgb)
                    seg = env.last_event.instance_segmentation_frame

                    #########Parts for input into get_annotation_instances#############
                    masks = []
                    label_ids = {}
                    counter = 0
                    color2seg = env.last_event.color_to_object_id
                    for k, v in color2seg.items():
                        obj = v.split('|')[0]
                        if obj in ai2thor_objects2idx:
                            obj_cat = ai2thor_objects2idx[obj]
                            wheres_obj = np.where(seg[:,:] == np.array(list(k)))
                            if len(wheres_obj[0]) >0:
                                mask = np.zeros((300,300))
                                mask[(wheres_obj[0], wheres_obj[1])] = counter
                                masks.append(mask)
                            
                                label_ids[counter] = obj_cat
                                counter +=1
                        else:
                            pass

                    for k in label_ids:
                        assert k<= len(label_ids)
                    #########Parts for input into get_annotation_instances#############

                    json_dict = get_annotation_instances(masks, t, label_ids)

                    json.dump(json_dict, open(ss_picture_folder_name+ '/anno_'+ str(t) + '.json' ,'w'))

                t += 1
                #time.sleep(1)
                return t
            
            if t-prev_t_returned ==0:
                horizon = env.last_event.metadata['agent']['cameraHorizon']
                pose = env.last_event.metadata['agent']['position']
                #rotate and lookdown, up
                #time.sleep(1)
                env.set_horizon(0)
                for r_i in range(4):
                    angle = 0
                    for l_i in range(6):
                        angle =15
                        env.look_angle(angle, render_settings=None)
                        t = write(env, t)
                        #print("updated t is ", t)
                    for l_i in range(3):
                        env.look_angle(-30, render_settings=None)
                        #t = write(env, t)
                        #print("updated t is ", t)
                    _, _, _, _ , _ = env.va_interact("RotateLeft_90", interact_mask=None, smooth_nav=args.smooth_nav, debug=args.debug)
                #time.sleep(1)
                env.set_horizon(horizon)
                env.set_pose(pose)
                #pickle.dump(env.last_event.color_to_object_id, open(ss_picture_folder_name+ '/color_to_object_id.p', 'wb'))
                            

                env.set_horizon(0)
                for r_i in range(4):
                    s= True
                    while s:
                        s, _, _, _ , _ = env.va_interact("MoveAhead_25", interact_mask=None, smooth_nav=args.smooth_nav, debug=args.debug)
                    t = write(env, t, interaction=True)
                    for l_i in range(6):
                        angle =15
                        env.look_angle(angle, render_settings=None)
                        t = write(env, t)
                    for l_i in range(3):
                         env.look_angle(-30, render_settings=None)
                         #t = write(env, t, True)
                    _, _, _, _ , _ = env.va_interact("RotateLeft_90", interact_mask=None, smooth_nav=args.smooth_nav, debug=args.debug)
                env.set_horizon(horizon)
                env.set_pose(pose)
            
            
            # extract visual features
            curr_image = Image.fromarray(np.uint8(env.last_event.frame))
            action = expert_actions[t_actions]
            compressed_mask = action['args']['mask'] if 'mask' in action['args'] else None
            mask = env.decompress_mask(compressed_mask) if compressed_mask is not None else None
            action = action['action'] 

            # print action
            if args.debug:
                print(action)

            # use predicted action and mask (if available) to interact with the env
            #pickle.dump(env.last_event.instance_segmentation_frame, open("pickle/segmentation_frame" + str(pickle_count) + ".p", "wb"))
            #pickle.dump(env.last_event.instance_masks, open("pickle/instance_mask" + str(pickle_count) + ".p", "wb"))
            #pickle.dump(env.last_event.frame, open("pickle/rgb" + str(pickle_count) + ".p", "wb"))
            pickle_count +=1
            t_success, _, _, err, _ = env.va_interact(action, interact_mask=mask, smooth_nav=args.smooth_nav, debug=args.debug)
            if not t_success:
                fails += 1
                if fails >= args.max_fails:
                    print("Interact API failed %d times" % fails + "; latest error '%s'" % err)
                    break
            
            interaction = not('Rotate' in action or 'Move' in action or 'Look' in action) and t_success
            t = write(env, t, interaction)

            if not(first_pickup) and action == 'PickupObject' and t_success:
                #save horizon, pose
                horizon = env.last_event.metadata['agent']['cameraHorizon']
                pose = env.last_event.metadata['agent']['position']
                first_pickup = True
                #rotate and lookdown, up
                env.set_horizon(0)
                for r_i in range(4):
                    angle = 0
                    for l_i in range(6):
                        angle =15
                        env.look_angle(angle, render_settings=None)
                        t = write(env, t)
                    for l_i in range(3):
                        env.look_angle(-30, render_settings=None)
                        #t = write(env, t)
                    _, _, _, _ , _ = env.va_interact("RotateLeft_90", interact_mask=None, smooth_nav=args.smooth_nav, debug=args.debug)
                env.set_horizon(horizon)
                env.set_pose(pose)
                
                
                env.set_horizon(0)
                for r_i in range(4):
                    s= True
                    while s:
                        s, _, _, _ , _ = env.va_interact("MoveAhead_25", interact_mask=None, smooth_nav=args.smooth_nav, debug=args.debug)
                    t = write(env, t, interaction = True)
                    for l_i in range(6):
                        angle =15
                        env.look_angle(angle, render_settings=None)
                        t = write(env, t)
                    for l_i in range(3):
                         env.look_angle(-30, render_settings=None)
                         #t = write(env, t, True)
                    _, _, _, _ , _ = env.va_interact("RotateLeft_90", interact_mask=None, smooth_nav=args.smooth_nav, debug=args.debug)
                env.set_horizon(horizon)
                env.set_pose(pose)
                
                
                    
            # next time-step
            t_reward, t_done = env.get_transition_reward()
            #print("reward is ", t_reward, " and " , t_done)
            reward += t_reward
            t_actions +=1
            

        # check if goal was satisfied
        goal_satisfied = env.get_goal_satisfied()
        if goal_satisfied:
            print("Goal Reached")
            success = True


        # goal_conditions
        pcs = env.get_goal_conditions_met()
        goal_condition_success_rate = pcs[0] / float(pcs[1])

        # SPL
        path_len_weight = len(traj_data['plan']['low_actions'])
        s_spl = (1 if goal_satisfied else 0) * min(1., path_len_weight / float(t))
        pc_spl = goal_condition_success_rate * min(1., path_len_weight / float(t))

        # path length weighted SPL
        plw_s_spl = s_spl * path_len_weight
        plw_pc_spl = pc_spl * path_len_weight

        # log success/fails
        lock.acquire()
        log_entry = {'trial': traj_data['task_id'],
                     'type': traj_data['task_type'],
                     'repeat_idx': int(r_idx),
                     'goal_instr': goal_instr,
                     'completed_goal_conditions': int(pcs[0]),
                     'total_goal_conditions': int(pcs[1]),
                     'goal_condition_success': float(goal_condition_success_rate),
                     'success_spl': float(s_spl),
                     'path_len_weighted_success_spl': float(plw_s_spl),
                     'goal_condition_spl': float(pc_spl),
                     'path_len_weighted_goal_condition_spl': float(plw_pc_spl),
                     'path_len_weight': int(path_len_weight),
                     'reward': float(reward)}
        if success:
            successes.append(log_entry)
        else:
            failures.append(log_entry)

        # overall results
        results['all'] = cls.get_metrics(successes, failures)

        print("-------------")
        print("SR: %d/%d = %.3f" % (results['all']['success']['num_successes'],
                                    results['all']['success']['num_evals'],
                                    results['all']['success']['success_rate']))
        print("GC: %d/%d = %.3f" % (results['all']['goal_condition_success']['completed_goal_conditions'],
                                    results['all']['goal_condition_success']['total_goal_conditions'],
                                    results['all']['goal_condition_success']['goal_condition_success_rate']))
        print("PLW SR: %.3f" % (results['all']['path_length_weighted_success_rate']))
        print("PLW GC: %.3f" % (results['all']['path_length_weighted_goal_condition_success_rate']))
        print("-------------")

        # task type specific results
        task_types = ['pick_and_place_simple', 'pick_clean_then_place_in_recep', 'pick_heat_then_place_in_recep',
                      'pick_cool_then_place_in_recep', 'pick_two_obj_and_place', 'look_at_obj_in_light',
                      'pick_and_place_with_movable_recep']
        for task_type in task_types:
            task_successes = [s for s in (list(successes)) if s['type'] == task_type]
            task_failures = [f for f in (list(failures)) if f['type'] == task_type]
            if len(task_successes) > 0 or len(task_failures) > 0:
                results[task_type] = cls.get_metrics(task_successes, task_failures)
            else:
                results[task_type] = {}

        lock.release()
        return t

    @classmethod
    def get_metrics(cls, successes, failures):
        '''
        compute overall succcess and goal_condition success rates along with path-weighted metrics
        '''
        # stats
        num_successes, num_failures = len(successes), len(failures)
        num_evals = len(successes) + len(failures)
        total_path_len_weight = sum([entry['path_len_weight'] for entry in successes]) + \
                                sum([entry['path_len_weight'] for entry in failures])
        completed_goal_conditions = sum([entry['completed_goal_conditions'] for entry in successes]) + \
                                   sum([entry['completed_goal_conditions'] for entry in failures])
        total_goal_conditions = sum([entry['total_goal_conditions'] for entry in successes]) + \
                               sum([entry['total_goal_conditions'] for entry in failures])

        # metrics
        sr = float(num_successes) / num_evals
        pc = completed_goal_conditions / float(total_goal_conditions)
        plw_sr = (float(sum([entry['path_len_weighted_success_spl'] for entry in successes]) +
                        sum([entry['path_len_weighted_success_spl'] for entry in failures])) /
                  total_path_len_weight)
        plw_pc = (float(sum([entry['path_len_weighted_goal_condition_spl'] for entry in successes]) +
                        sum([entry['path_len_weighted_goal_condition_spl'] for entry in failures])) /
                  total_path_len_weight)

        # result table
        res = dict()
        res['success'] = {'num_successes': num_successes,
                          'num_evals': num_evals,
                          'success_rate': sr}
        res['goal_condition_success'] = {'completed_goal_conditions': completed_goal_conditions,
                                        'total_goal_conditions': total_goal_conditions,
                                        'goal_condition_success_rate': pc}
        res['path_length_weighted_success_rate'] = plw_sr
        res['path_length_weighted_goal_condition_success_rate'] = plw_pc

        return res

    def create_stats(self):
            '''
            storage for success, failure, and results info
            '''
            self.successes, self.failures = self.manager.list(), self.manager.list()
            self.results = self.manager.dict()

    def save_results(self):
        results = {'successes': list(self.successes),
                   'failures': list(self.failures),
                   'results': dict(self.results)}

        save_path = os.path.dirname('results')
        if not(os.path.exists(save_path)):
            os.makedirs(save_path)
        save_path = os.path.join(save_path, 'task_results_' + self.args.eval_split + '_' + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + '.json')
        with open(save_path, 'w') as r:
            json.dump(results, r, indent=4, sort_keys=True)

