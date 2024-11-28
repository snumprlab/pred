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
from alfworld_mrcnn import load_pretrained_model
import copy
import torchvision.transforms as T
import cv2

from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.structures import Boxes, Instances

from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2 import model_zoo
from collections import Counter, OrderedDict



#Mohit's model
sem_seg_model = load_pretrained_model('/Users/soyeonmin/Documents/OGN/maskrcnn_alfworld/mrcnn.pth', torch.device('cpu'))
sem_seg_model.eval()
transform = T.Compose([T.ToTensor()])
alfworld_objects = ['AlarmClock', 'Apple', 'AppleSliced', 'BaseballBat', 'BasketBall', 'Book', 'Bowl', 'Box', 'Bread', 'BreadSliced', 'ButterKnife', 'CD', 'Candle', 'CellPhone', 'Cloth', 'CreditCard', 'Cup', 'DeskLamp', 'DishSponge', 'Egg', 'Faucet', 'FloorLamp', 'Fork', 'Glassbottle', 'HandTowel', 'HousePlant', 'Kettle', 'KeyChain', 'Knife', 'Ladle', 'Laptop', 'LaundryHamperLid', 'Lettuce', 'LettuceSliced', 'LightSwitch', 'Mug', 'Newspaper', 'Pan', 'PaperTowel', 'PaperTowelRoll', 'Pen', 'Pencil', 'PepperShaker', 'Pillow', 'Plate', 'Plunger', 'Pot', 'Potato', 'PotatoSliced', 'RemoteControl', 'SaltShaker', 'ScrubBrush', 'ShowerDoor', 'SoapBar', 'SoapBottle', 'Spatula', 'Spoon', 'SprayBottle', 'Statue', 'StoveKnob', 'TeddyBear', 'Television', 'TennisRacket', 'TissueBox', 'ToiletPaper', 'ToiletPaperRoll', 'Tomato', 'TomatoSliced', 'Towel', 'Vase', 'Watch', 'WateringCan', 'WineBottle']
alfworld_obj2idx = {o:i for i,o in enumerate(alfworld_objects)}
alfworld_idx2obj = {v:k for k,v in alfworld_obj2idx.items()}

#My segmentation model


cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.INPUT.MASK_FORMAT = "bitmask"
cfg.INPUT.MIN_SIZE_TRAIN =  (300,300)
cfg.INPUT.MIN_SIZE_TEST = 300
cfg.INPUT.MAX_SIZE_TEST = 300
cfg.SOLVER.IMS_PER_BATCH = 1
cfg.MODEL.DEVICE = "cpu"
cfg.MODEL.PIXEL_MEAN=[103.530, 116.280, 123.675, 1500]
cfg.MODEL.PIXEL_STD=[1.0, 1.0, 1.0, 10.0]


cfg.MODEL.ROI_HEADS.NUM_CLASSES =77 # Num. small classes
sem_seg_model_small = build_model(cfg)

#state_dict = torch.load('segmentation_models/' + args.load_my_model_dir, map_location='cpu')
small_state_dict = torch.load('/Users/soyeonmin/Documents/OGN/segmentation_models/' + 'small_1e-3model_map_iter_18299ap_9.330891561735355.pth', map_location=torch.device('cpu'))
new_state_dict = OrderedDict()
for k, v in small_state_dict.items():
    name = k[7:] # remove `module.`
    new_state_dict[name] = v
del small_state_dict

sem_seg_model_small.load_state_dict(new_state_dict)
sem_seg_model_small.to(torch.device('cpu'))
sem_seg_model_small.eval()

large_objects_with_basin = ['ArmChair',
'Bathtub',
'BathtubBasin',
'Bed',
'Blinds',
'Cabinet',
'Chair',
'CoffeeMachine',
'CounterTop',
'Curtains',
'Desk',
'DeskLamp',
'Drawer',
'Dresser',
'FloorLamp',
'Footstool',
'Fridge',
'GarbageCan',
'HousePlant',
'LaundryHamper',
'LaundryHamperLid',
'Microwave',
'Mirror',
'Ottoman',
'Painting',
'Pillow',
'Shelf',
'ShowerDoor',
'ShowerGlass',
'Sink',
'SinkBasin',
'Sofa',
'DiningTable',
'CoffeeTable',
'SideTable',
'Television',
'TVStand',
'Window']
ai2thor_objects2idx = pickle.load(open('/Users/soyeonmin/Documents/OGN/ai2thor_objects2idx.p', 'rb'))
small = [k for k in ai2thor_objects2idx if not(k in large_objects_with_basin)]
small_objects2idx = {k:i for i, k in enumerate(small)}
small_idx2small_object = {v:k for k,v in small_objects2idx.items()}

class EvalTask(Eval):
    '''
    evaluate overall task performance
    '''

    @classmethod
    def run(cls, model, resnet, task_queue, args, lock, successes, failures, results, n,a,b):
        '''
        evaluation loop
        '''
        # start THOR
        global episode_no
        env = ThorEnv()
        cls.dn = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        
        cls.episode_no = 0
        #if cls.args.small:
        training_pic_save_task_ids = pickle.load(open('alfred_training_pic_idxes_small.p', 'rb'))
        #else:
        #training_pic_save_task_ids = pickle.load(open('alfred_training_pic_idxes_10.p', 'rb'))
        files = json.load(open("../OGN/alfred_data_small/splits/oct21.json"))[args.eval_split][args.from_idx:args.to_idx]
        task_count = 0
        while task_count < args.to_idx -args.from_idx:
            try:
                #task = {'repeat_idx': 0,'task': 'pick_and_place_simple-Vase-None-CoffeeTable-207/trial_T20190909_091246_807206'}
                #task={'repeat_idx': 1,'task': 'pick_and_place_simple-Tomato-None-Microwave-13/trial_T20190908_124829_632660'}
                #task ={'repeat_idx': 0, 'task': 'pick_cool_then_place_in_recep-TomatoSliced-None-Microwave-12/trial_T20190908_165525_911839'}
                #task = {'repeat_idx': 0, 'task': 'pick_and_place_with_movable_recep-Ladle-Pan-SinkBasin-4/trial_T20190908_192636_561572'}
                #task = {'repeat_idx': 0, 'task': 'pick_clean_then_place_in_recep-ButterKnife-None-Drawer-30/trial_T20190908_052007_212776'}
                #task = {'repeat_idx': 1, 'task': 'pick_clean_then_place_in_recep-AppleSliced-None-DiningTable-27/trial_T20190907_151802_277016'}
                #task = {'repeat_idx': 0, 'task': 'pick_heat_then_place_in_recep-PotatoSliced-None-SinkBasin-13/trial_T20190909_115736_122556'}

                task = files[task_count]
                #traj = model.load_task_json(task)
                traj = json.load(open('data/json_2.1.0/' + task['task'] + '/pp/ann_' + str(task['repeat_idx']) + '.json'))
                r_idx = task['repeat_idx']
                #r_idx = 1
                print("Evaluating: %s" % (traj['root']))
                print("No. of trajectories left: %d" % (task_queue.qsize()))
                cls.evaluate(env, model, r_idx, resnet, traj, args, lock, successes, failures, results)
                cls.episode_no +=1
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                print("Error: " + repr(e))
            task_count +=1
        # stop THOR
        env.stop()


    

    @classmethod
    def evaluate(cls, env, model, r_idx, resnet, traj_data, args, lock, successes, failures, results):
        
        # reset model
        #model.reset()

        # setup scene
        reward_type = 'dense'
        cls.setup_scene(env, traj_data, r_idx, args, reward_type=reward_type)
        print("starting camera horizon is ", env.last_event.metadata['agent']['cameraHorizon'])

        # extract language features
        #feat = model.featurize([traj_data], load_mask=False)

        # goal instr
        goal_instr = traj_data['turk_annotations']['anns'][r_idx]['task_desc']

        done, success = False, False
        fails = 0
        t = 0
        #t1 = 0
        reward = 0
        pickle_count= 0
        #print(env.last_event.metadata.keys())
        episode_no = cls.episode_no
        print("episode no is ", episode_no)
        
        expert_actions = [a['discrete_action'] for a in traj_data['plan']['low_actions']]
        pickle.dump(expert_actions, open('e_actions.p', 'wb'))
        #expert_actions = expert_actions[:39] + [{'action': 'MoveAhead_25', 'args': {}}]  + [{'action': 'LookDown', 'args': {}}] + expert_actions[39:]
        #expert_actions = expert_actions[:21] + expert_actions[23:]
        while not done:
            # break if max_steps reached
            if t >= args.max_steps:
                break
            
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
            cv2.imwrite('33rgb/rgb_' + str(t) + '.png',cv2.cvtColor(env.last_event.frame, cv2.COLOR_BGR2RGB))
            cv2.imwrite('33depth/depth_' + str(t) + '.png',env.last_event.depth_frame/ np.max(env.last_event.depth_frame))

                
            #Get segmented result
            rgb = copy.deepcopy(env.last_event.frame)
            rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR) #now in BGR
            # depth = env.last_event.depth_frame
            # depth = torch.tensor(np.expand_dims(depth, axis=0)).type(torch.FloatTensor)
            # #Add depth
            # im = torch.tensor(rgb.transpose(2,0,1)).type(torch.FloatTensor)
            # im = torch.cat([im, depth], dim=0)
            # #im = im.to(device=self.sem_seg_gpu)
            # del rgb; del depth
            # input_dict = {"image": im}

            small_ims = [rgb]
            small_im_tensors = [transform(i) for i in small_ims]
            results_small = sem_seg_model(small_im_tensors)[0]  

            #outputs_small = sem_seg_model_small([input_dict])[0]
            #out_inst_small = outputs_small['instances']            

            indices = []
            for i  in range(len(results_small['scores'])):
                score = results_small['scores'][i]
                if score > 0.5:
                    indices.append(i)

            #for i  in range(len(out_inst_small.scores)):
            #    score = out_inst_small.scores[i]
            #    if score > 0.3:
            #        indices.append(i)

            classes =[alfworld_idx2obj[l.item()] for l in results_small['labels'][indices]]
            #classes =[small_idx2small_object[l.item()] for l in out_inst_small.pred_classes[indices]]
            print("classes seen is ", classes)

            # next time-step
            t_reward, t_done = env.get_transition_reward()
            #print("reward is ", t_reward, " and " , t_done)
            reward += t_reward
            t +=1
            #time.sleep(1)
            

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

        save_path = os.path.dirname('results')
        if not(os.path.exists(save_path)):
            os.makedirs(save_path)
        save_path = os.path.join(save_path, 'task_results_' + self.args.eval_split + '_' + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + '.json')
        with open(save_path, 'w') as r:
            json.dump(results, r, indent=4, sort_keys=True)

