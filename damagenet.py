import os
from copy import deepcopy
import numpy as np
import argparse
import innvestigate.utils as iutils
from keras.models import Model
from keras.utils import to_categorical
import keras
import tensorflow as tf

from utils import *
from lrp import *
from transfers import *


def AoA(net_name, start_id, end_id, gpu_id):
    # acquire basic information
    sess = tf.InteractiveSession()
    inputs, analysis, loss, direction = {}, {}, {}, {}
    sample, prob, labels, result_dir, record, rmsds = {}, {}, {}, {}, [], []
    size, pre_pro = load_net(net_name, return_net=False)

    # hyper parameters
    config   = {'TI': False, 'DI': False, 'SI': True}
    rmsd_thes= 7
    bound    = 25.5
    epsilon  = 2

    # record directory
    result_dir['adv'] = 'DAmageNet'
    os.makedirs(result_dir['adv'], exist_ok=True)
    print('\n' + result_dir['adv'], '\n')
    
    # build networks and LRP
    inputs['image']    = tf.placeholder(tf.float32, [1, size, size, 3], name='input')
    net, _             = load_net(net_name, inp=pre_pro(inputs['image'], backend=keras.backend, layers=keras.layers, models=keras.models, utils=keras.utils))
    prob_tensor        = net.output
    inputs['label']    = tf.placeholder(tf.float32, [1, 1000], name='label')
    
    # define loss
    partial_model      = Model(inputs=net.inputs, outputs=iutils.keras.graph.pre_softmax_tensors(net.outputs), name=net.name + '_partial')
    analysis['first']  = build_lrp(partial_model, out=inputs['label'])
    analysis['second'] = build_lrp(partial_model, out=prob_tensor, out_ori=inputs['label'])
    loss['aoa'] = tf.log(tf.reduce_mean(analysis['first']) / tf.reduce_mean(analysis['second'])) + 1000 * tf.reduce_mean(inputs['label'] * tf.log(prob_tensor + 1e-25))
    direction['aoa'] = build_direction(loss['aoa'], inputs['image'], TI=config['TI'])[0]
    initialize_uninitialized(sess)
    
    # load label
    with open(paths['Label'], 'r') as f: label_list = f.read().split('\n')
    for item in label_list: 
        if item == '': continue
        item = item.split(' ')
        labels[item[0]] = int(item[1])

    # begin attack
    file_lists = sorted(os.listdir(paths['Data']))
    num_attack = end_id - start_id
    prt_detail = num_attack <= 100
    start = time.time()
    for sam_id, file in enumerate(file_lists):
        # stop condition
        if sam_id < start_id - 1: continue
        if sam_id >= end_id: break

        # load sample
        class_id = labels[file]
        sample['ori'] = process_sample(paths['Data'] + '/' + file, size)
        sample['adv'] = deepcopy(sample['ori'])
        
        # get original variables
        prob['pred']  = sess.run(prob_tensor, {inputs['image']: [sample['ori']]})[0]
        pred_ori = np.argmax(prob['pred'])
        prob['label'] = to_categorical(class_id, 1000)
            
        # attack iter for each sample
        success_condition = False
        iter_done = 0
        direction_value = 0
        if prt_detail: print()
        while 1:
            # stop condition
            feed_dict = {inputs['image']: [sample['adv']], inputs['label']: [prob['label']]}
            rmsd = np.sqrt(np.mean(np.square(sample['adv']-sample['ori'] + 1e-18)))
            out_dict = {'Iter': iter_done, 'Success': success_condition, 'RMSD': rmsd}
            loss_value = sess.run(loss, feed_dict)
            out_dict.update(loss_value)
            output(out_dict, prt=prt_detail)
            if rmsd > rmsd_thes: break

            # attack
            direction_value = update_sample(direction['aoa'], sess, feed_dict, sample['adv'], inputs['image'], config['DI'], config['SI'])
            if np.isnan(np.mean(direction_value)): break
            sample['adv'] = np.clip(sample['adv'] - epsilon * direction_value, 0, 255)
            sample['adv'] = np.clip(sample['adv'], sample['ori'] - bound, sample['ori'] + bound)
            prob['pred'] = sess.run(prob_tensor, {inputs['image']: [sample['adv']]})[0]
            
            # success condition
            success_condition = np.argmax(prob['pred']) != class_id
            iter_done += 1

        # visualization
        record.append(success_condition)
        rmsds.append(rmsd)
        output({'No':        '%d/%d' % (sam_id+1, end_id),
                'File':      os.path.splitext(file)[0][-8:],
                'Iter':      iter_done,
                'Succ':      success_condition,
                'Rate':      sum(record)/(len(record)+0.001),
                'RMSD':      '%.3f in %.3f' % (rmsd, sum(rmsds)/(len(rmsds)+0.001)),
                'TimeRm':    convert_second_to_time((time.time()-start)/len(rmsds)*(num_attack-len(rmsds))),
                'Prob':      '%.2f->%.2f %.2f' % (np.max(prob['label']), prob['pred'][np.argmax(prob['label'])], np.max(prob['pred'])),
                'Class':     '%d %d->%d' % (np.argmax(prob['label']), pred_ori, np.argmax(prob['pred'])),
                })
        
        # save record
        def save_imgs(img, path): PIL.Image.fromarray(img.astype(np.uint8)).save(path)
        save_imgs(sample['adv'], result_dir['adv'] + '/' + os.path.splitext(file)[0] + '.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('start_id', type=int, help='start_id')
    parser.add_argument('end_id', type=int, help='end_id')
    parser.add_argument('gpu_id', help='GPU(s) used')
    args, _ = parser.parse_known_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    AoA(net_name='VGG19', start_id=args.start_id, end_id=args.end_id, gpu_id=args.gpu_id)