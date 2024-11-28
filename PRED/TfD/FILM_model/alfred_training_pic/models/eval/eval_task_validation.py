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

import sys

sys.path.append(os.path.join(os.environ['BTS_ROOT'], 'pytorch'))
sys.path.append(os.path.join(os.environ['BTS_ROOT'], 'pytorch/models/bts_nyu_v2_pytorch_densenet161/'))


from bts import *
from bts_dataloader import preprocessing_transforms

import torch
from torch.autograd import Variable
from types import SimpleNamespace
from torchvision import transforms



depth_gpu =  torch.device("cuda")


bts_args = SimpleNamespace(model_name='bts_nyu_v2_pytorch_densenet161' ,
            encoder='densenet161_bts',
            dataset='alfred',
            input_height=300,
            input_width=300,
            max_depth=5,
            mode = 'test',
            device = depth_gpu,
            set_view_angle= False,
            load_decoder=False,
            load_encoder=False,
            bts_size=512)

depth_pred_model = BtsModel(params=bts_args).to(device=depth_gpu)

depth_checkpoint_path  = 'alfred_depth_focal_259_batchsize_24_no_angle_lr_5e-05_ld_0.01/model-11250-best_rms_0.32901'
ckpt_path = os.environ['BTS_ROOT'] + '/pytorch/models/' + depth_checkpoint_path 
            
checkpoint = torch.load(ckpt_path, map_location=depth_gpu)['model']

new_checkpoint = OrderedDict()
for k, v in checkpoint.items():
    name = k[7:] # remove `module.`
    new_checkpoint[name] = v
del checkpoint
# load params
depth_pred_model.load_state_dict(new_checkpoint)
depth_pred_model.eval()
depth_pred_model.to(device=depth_gpu)



class EvalTask(Eval):
    '''
    evaluate overall task performance
    '''

    @classmethod
    def run(cls, model, resnet, task_queue, args, lock, successes, failures, results, process_no):
        '''
        evaluation loop
        '''
        # start THOR
        global episode_no
        env = ThorEnv()
        cls.dn = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        
        episode_no = -1 + 1000* process_no
        #epno = 0
        #if cls.args.small:
        #training_pic_save_task_ids = pickle.load(open('alfred_training_pic_idxes_small.p', 'rb'))
        #else:
        #training_pic_save_task_ids = pickle.load(open('alfred_training_pic_idxes_10.p', 'rb'))

        #depth_picture_folder_name = "validation_pictures_10_depth_only/" +  "/depth/"
        rgb_picture_folder_name = "validation_pictures_10_segmentation/" + "/rgb/" 
        ss_picture_folder_name = "validation_pictures_10_segmentation/"  "/segmentation/" 
        #if not os.path.exists(depth_picture_folder_name):
        #    os.makedirs(depth_picture_folder_name)
        if not os.path.exists(rgb_picture_folder_name):
            os.makedirs(rgb_picture_folder_name)
        if not os.path.exists(ss_picture_folder_name):
            os.makedirs(ss_picture_folder_name)

        prev_t_returned = 0
        while True:
            #prev_t_returned = 0

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

                
                #traj = model.load_task_json(task)
                
                #np.random.seed(episode_no)
                episode_no +=1
                np.random.seed(episode_no)
                if  np.random.choice(100) == 0:
                    traj = json.load(open('data/json_2.1.0/' + task['task'] + '/pp/ann_' + str(task['repeat_idx']) + '.json'))
                    r_idx = task['repeat_idx']
                    #r_idx = 1
                    print("Evaluating: %s" % (traj['root']))
                    print("No. of trajectories left: %d" % (task_queue.qsize()))

                    t_returned = cls.evaluate(env, model, r_idx, resnet, traj, args, lock, successes, failures, results, rgb_picture_folder_name, ss_picture_folder_name,  episode_no, prev_t_returned, process_no)
                    prev_t_returned += t_returned
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                print("Error: " + repr(e))

        # stop THOR
        env.stop()


    

    @classmethod
    def evaluate(cls, env, model, r_idx, resnet, traj_data, args, lock, successes, failures, results, rgb_picture_folder_name, ss_picture_folder_name, episode_no, prev_t_returned, process_no):
        

        def pad_rgb(rgb, ww=352, hh=352):
            ht, wd, cc= rgb.shape
            ht, wd, = 300, 300 
            #
            # create new image of desired size and color (blue) for padding
            color = (255,255,255) #black
            result = np.full((hh,ww,cc), color, dtype=np.uint8)
            #
            # compute center offset
            xx = (ww - wd) // 2
            yy = (hh - ht) // 2
            #
            # copy img image into center of result image
            result[yy:yy+ht, xx:xx+wd] = rgb
            return result

        def depth_center(depth_est):
            ht, wd, = 300, 300            
            ww = 352; hh = 352 #Change accordingly 
            xx = (ww - wd) // 2
            yy = (hh - ht) // 2
            return depth_est[:, :, yy:yy+ht, xx:xx+wd]

        def normalize_depth(value):
            value = value.cpu().numpy()[0, :, :]
            #
            vmin = value.min() 
            vmax = value.max() 
            #
            if vmin != vmax:
                value = (value - vmin) / (vmax - vmin)
            else:
                value = value * 0.
            #
            return np.expand_dims(value, 2)


        def normalize_depth_2(value):
            value = value.cpu().numpy()[0, :, :]
            value = value/5.0
            return np.expand_dims(value, 2)


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
        #np.random.seed(prev_t_returned)
        t = process_no * 10000 +  prev_t_returned

        t_actions = 0
        #t1 = 0
        reward = 0
        pickle_count= 0
        #print(env.last_event.metadata.keys())
        #episode_no = cls.episode_no
        #print("episode no is ", episode_no)
        
        expert_actions = [a['discrete_action'] for a in traj_data['plan']['low_actions']]
        #expert_actions = expert_actions[:39] + [{'action': 'MoveAhead_25', 'args': {}}]  + [{'action': 'LookDown', 'args': {}}] + expert_actions[39:]
        #expert_actions = expert_actions[:21] + expert_actions[23:]
        while not done and t_actions < len(expert_actions)-1:
            # break if max_steps reached
            first_pickup = False
            if t -(process_no * 10000 +  prev_t_returned) >= args.max_steps:
                break
        
            
            def write(env, t, interaction = False, dontwrite=False):
                np.random.seed(t)
                focal = 259.8076
                if np.random.choice(3) ==0:

                    rgb = torch.tensor(env.last_event.frame.copy()).numpy() #shape (h, w, 3)

                    rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR) #Now in BGR

                    normalize_rgb = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    rgb_padded = pad_rgb(rgb).astype(np.float32) / 255.0

                    rgb_padded =  torch.tensor(rgb_padded.transpose(2, 0, 1))
                    rgb_padded[:3, :, :] = normalize_rgb(rgb_padded[:3, :, :])
                    rgb_padded = rgb_padded.unsqueeze(0)
                    rgb_padded = rgb_padded.to(device=depth_gpu)

                    _, _, _, _,depth_est  = depth_pred_model(Variable(rgb_padded), focal)

                    depth_est = depth_center(depth_est)
                    #
                    #
                    depth = depth_est.squeeze(1).detach()
                    depth_save = normalize_depth_2(depth[0:1, :, :])
                    rgb_d = rgb_picture_folder_name + "image_" + str(t) + ".png"
                    depth_dir = rgb_d.replace('image_', 'depth_').replace('.png', '.p')
                    pickle.dump(depth_save, open(depth_dir.replace('depth_', 'depth2_'), 'wb'))
                    
                    #if (np.random.choice(5) == 0 or interaction) and not(dontwrite):
                    cv2.imwrite(rgb_picture_folder_name + "image_" + str(t) + ".png" , rgb)
                    #pickle.dump(depth[:, :, 0]*100.0, open(depth_picture_folder_name + "depth_" + str(t) + ".p", "wb"))
                    #pickle.dump(cam_hor, open(depth_picture_folder_name + "angle_" + str(t) + ".p", "wb"))
                    #if (np.random.choice(5) == 0 or interaction) and not(dontwrite):
                    pickle.dump(env.last_event.instance_segmentation_frame, open(ss_picture_folder_name + "segmentation_" + str(t) + ".p", 'wb'))
                    pickle.dump(env.last_event.color_to_object_id, open(ss_picture_folder_name+ '/color_to_object_id'+ str(t) + '.p', 'wb'))
                t += 1
                #time.sleep(1)
                return t
            
            if t -(process_no * 10000 +  prev_t_returned)  ==0:
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
            #feat['frames'] = resnet.featurize([curr_image], batch=1).unsqueeze(0)

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
            
            #else:
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

