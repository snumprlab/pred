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


class EvalTask(Eval):
    '''
    evaluate overall task performance
    '''

    @classmethod
    def run(cls, model, resnet, task_queue, args, lock, successes, failures, results):
        '''
        evaluation loop
        '''
        # start THOR
        env = ThorEnv()
        cls.dn = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        episode_no = 0
# =============================================================================
#         training_pic_save_task_ids = 
# =============================================================================
        while True:
            if task_queue.qsize() == 0:
                break

            task = task_queue.get()

            try:
                #task = {'repeat_idx': 0,'task': 'pick_and_place_simple-Vase-None-CoffeeTable-207/trial_T20190909_091246_807206'}
                #task={'repeat_idx': 1,'task': 'pick_and_place_simple-Tomato-None-Microwave-13/trial_T20190908_124829_632660'}
                #task ={'repeat_idx': 0, 'task': 'pick_cool_then_place_in_recep-TomatoSliced-None-Microwave-12/trial_T20190908_165525_911839'}
                #task = {'repeat_idx': 0, 'task': 'pick_and_place_with_movable_recep-Ladle-Pan-SinkBasin-4/trial_T20190908_192636_561572'}
                #task = {'repeat_idx': 0, 'task': 'pick_clean_then_place_in_recep-ButterKnife-None-Drawer-30/trial_T20190908_052007_212776'}
                #task = {'repeat_idx': 1, 'task': 'pick_clean_then_place_in_recep-AppleSliced-None-DiningTable-27/trial_T20190907_151802_277016'}
                #task = {'repeat_idx': 0, 'task': 'pick_heat_then_place_in_recep-PotatoSliced-None-SinkBasin-13/trial_T20190909_115736_122556'}

                
                traj = model.load_task_json(task)
                r_idx = task['repeat_idx']
                #r_idx = 1
                print("Evaluating: %s" % (traj['root']))
                print("No. of trajectories left: %d" % (task_queue.qsize()))
# =============================================================================
#                 if traj['task_id'] in training_pic_save_task_ids:
# =============================================================================
                cls.evaluate(env, model, r_idx, resnet, traj, args, lock, successes, failures, results)
            except Exception as e:
                import traceback
                traceback.print_exc()
                print("Error: " + repr(e))

        # stop THOR
        env.stop()


    @classmethod
    def evaluate(cls, env, model, r_idx, resnet, traj_data, args, lock, successes, failures, results):
        # reset model
        model.reset()

        # setup scene
        reward_type = 'dense'
        cls.setup_scene(env, traj_data, r_idx, args, reward_type=reward_type)

        # extract language features
        feat = model.featurize([traj_data], load_mask=False)

        # goal instr
        goal_instr = traj_data['turk_annotations']['anns'][r_idx]['task_desc']

        done, success = False, False
        fails = 0
        t = 0
        reward = 0
        pickle_count= 0
        print(env.last_event.metadata.keys())
        
        expert_actions = [a['discrete_action'] for a in traj_data['plan']['low_actions']]
        #expert_actions = expert_actions[:39] + [{'action': 'MoveAhead_25', 'args': {}}]  + [{'action': 'LookDown', 'args': {}}] + expert_actions[39:]
        #expert_actions = expert_actions[:21] + expert_actions[23:]
        while not done:
            # break if max_steps reached
            first_pickup = False
            if t >= args.max_steps:
                break
            if t ==0:
                depth_picture_folder_name = "trainig_pictures/" +  cls.dn + "/depth/"+ traj_data['trial_T20190906_214648_973919'] + "/"
                rgb_picture_folder_name = "trainig_pictures/" + cls.dn + "/rgb/" + traj_data['trial_T20190906_214648_973919']  + "/"
                ss_picture_folder_name = "trainig_pictures/" + cls.dn + "/segmentation/" + traj_data['trial_T20190906_214648_973919']  + "/"
                os.makedirs(depth_picture_folder_name)
                os.makedirs(rgb_picture_folder_name)
                os.makedirs(ss_picture_folder_name)
            
            if t == 1:
                pickle.dump(env.last_event.color_to_object_id, open(ss_picture_folder_name+ '/color_to_object_id.p', 'wb'))

            # extract visual features
            curr_image = Image.fromarray(np.uint8(env.last_event.frame))
            feat['frames'] = resnet.featurize([curr_image], batch=1).unsqueeze(0)

            # forward model
            #m_out = model.step(feat)
            #m_pred = model.extract_preds(m_out, [traj_data], feat, clean_special_tokens=False)
            #m_pred = list(m_pred.values())[0]
            #print("m pred is ", m_pred)
            #print("action low is ", m_pred['action_low'])
            #print("mask shape ", m_pred['action_low_mask'][0].shape)


            # check if <<stop>> was predicted
            #if m_pred['action_low'] == cls.STOP_TOKEN:
            #    print("\tpredicted STOP")
            #    break

            # get action and mask
            #action, mask = m_pred['action_low'], m_pred['action_low_mask'][0]
            #mask = np.squeeze(mask, axis=0) if model.has_interaction(action) else None
            
# =============================================================================
#             if t in [0,1]:
#                 mask = None
#                 action = "LookUp"
# =============================================================================
            #else:
            action = expert_actions[t]
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
            
            rgb = torch.tensor(env.last_event.frame.copy()).numpy() #shape (h, w, 3)
            depth = torch.tensor(env.last_event.depth_frame.copy()).numpy() #shape (h, w)
            depth /= 1000.0
            depth = np.expand_dims(depth, 2)
            
            state = np.concatenate((rgb, depth), axis = 2).transpose(2, 0, 1).transpose(1, 2, 0)
            
            rgb = state[:, :, :3]
            depth = state[:, :, 3:4]
            
            
# =============================================================================
#             cv2.imwrite(rgb_picture_folder_name + "rgb_" + str(t) + ".png" , rgb)
#             cv2.imwrite(depth_picture_folder_name + "depth_" + str(t) + ".png" , depth[:, :, 0]*100.0)
#             pickle.dump(cls.last_event.instance_segmentation_frame, open(ss_picture_folder_name + "segmentation_" + str(t) + ".p", 'wb'))
# 
# =============================================================================
            if not(first_pickup) and action == 'PickupObject':
                #save horizon, pose
                horizon = env.last_event.metadata['agent']['cameraHorizon']
                pose = env.last_event.metadata['agent']['position']
                first_pickup = True
                
                    
            # next time-step
            t_reward, t_done = env.get_transition_reward()
            #print("reward is ", t_reward, " and " , t_done)
            reward += t_reward
            t += 1

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

        save_path = os.path.dirname(self.args.model_path)
        save_path = os.path.join(save_path, 'task_results_' + self.args.eval_split + '_' + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + '.json')
        with open(save_path, 'w') as r:
            json.dump(results, r, indent=4, sort_keys=True)

