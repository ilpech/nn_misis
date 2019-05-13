import argparse
import mxnet as mx
import numpy as np
import os
from mxnet import gluon, nd
from mxnet.gluon.data.vision import transforms
from tools import *
import sys

def get_data_raw_collect(dataset_path, batch_size, num_workers, input_size):
    val_path = os.path.join(dataset_path, 'val')
    test_path = os.path.join(dataset_path, 'test')
    transform_test = transforms.Compose([
        transforms.Resize(input_size, keep_ratio = True),
        transforms.ToTensor(),
    ])
    val_data = gluon.data.DataLoader(
        gluon.data.vision.ImageFolderDataset(val_path).transform_first(transform_test),
        batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_data = gluon.data.DataLoader(
        gluon.data.vision.ImageFolderDataset(test_path).transform_first(transform_test),
        batch_size=batch_size, shuffle=False, num_workers = num_workers)
    return val_data, test_data

def get_tp_fp_fn(net, val_data, ctx, classes_len):
    tp = np.zeros(shape=(1,classes_len))
    fp = np.zeros(shape=(1,classes_len))
    fn = np.zeros(shape=(1,classes_len))
    error_matrix = np.zeros(shape=(classes_len,classes_len))
    for i, batch in enumerate(val_data):
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
        label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)
        outputs = [net(X) for X in data]
        for data_ind in range(len(outputs[0])):
            pred = nd.argmax(outputs[0][data_ind], axis=0).astype('int')
            pred = pred.asnumpy()[0]
            ground_truth = label[0][data_ind]
            ground_truth = ground_truth.asnumpy()[0]

            if pred == ground_truth:
                tp[0][ground_truth] += 1
                error_matrix[ground_truth][ground_truth] += 1
            elif pred != ground_truth:
                fn[0][ground_truth] += 1
                fp[0][pred] += 1
                error_matrix[ground_truth][pred] += 1

    return tp, fp, fn, error_matrix

parser = argparse.ArgumentParser(
                                description=('Train a mxnet model'
                                             'for image classification.')
                                )
parser.add_argument(
                    '--net-name', type=str,
                    help='set network name'
                    )
parser.add_argument(
                    '--params-dir', type=str,
                    help='set path to dir with params & struct'
                    )
parser.add_argument(
                    '--dataset-path', type=str,
                    help=('path to folder with sorted data:'
                          'into test, train, val with classes inside'
                          )
                    )
parser.add_argument(
                    '--epoch', type=int,
                    help='set number of epochs for training, def = 240'
                    )
opt = parser.parse_args()
net_name = opt.net_name
params_dir = opt.params_dir
dataset_path = opt.dataset_path
epoch = opt.epoch

num_gpus = 1
num_workers = os.cpu_count()
ctx = [mx.cpu()]
params_path = os.path.join(params_dir, '{}-{:04d}.params'.format(
                                                            net_name, epoch))
sym_path = os.path.join(params_dir, '{}-symbol.json'.format(net_name))
net = gluon.nn.SymbolBlock.imports(sym_path, ['data'], params_path, ctx=ctx)
input_size = (64, 64)
val_data, test_data = get_data_raw_collect(dataset_path, 16, num_workers, input_size)
num_batch = len(val_data)
params_path = os.path.join(params_dir, net_name)
dict_path = os.path.join(params_dir, net_name + '_classes.txt')
with open(dict_path) as f:
    content = f.readlines()
class_names = [x.strip() for x in content]
classes_len = len(class_names)

testing_dir = '/datasets/testing'
ensure_folder(testing_dir)

test_tp, test_fp, test_fn, test_error_matrix = get_tp_fp_fn(net, test_data, ctx, classes_len)
np.savetxt(os.path.join(testing_dir, 'test_error_matrix.csv'), 
                        test_error_matrix, fmt='%d',delimiter=',')
np.savetxt(os.path.join(testing_dir, 'test_tp.csv'), 
                        test_tp, fmt='%d',delimiter=',')
np.savetxt(os.path.join(testing_dir, 'test_fp.csv'), 
                        test_fp, fmt='%d',delimiter=',')
np.savetxt(os.path.join(testing_dir, 'test_fn.csv'), 
                        test_fn, fmt='%d',delimiter=',')

print('test')
print('test_class matrix')
print(test_error_matrix)
print('test_tp: ', test_tp)
print('test_fp: ', test_fp)
print('test_fn: ', test_fn)

val_tp, val_fp, val_fn, val_error_matrix = get_tp_fp_fn(net, val_data, ctx, classes_len)
np.savetxt(os.path.join(testing_dir, 'val_error_matrix.csv'), 
                        val_error_matrix, fmt='%d',delimiter=',')
np.savetxt(os.path.join(testing_dir, 'val_tp.csv'), 
                        val_tp, fmt='%d',delimiter=',')
np.savetxt(os.path.join(testing_dir, 'val_fp.csv'), 
                        val_fp, fmt='%d',delimiter=',')
np.savetxt(os.path.join(testing_dir, 'val_fn.csv'), 
                        val_fn, fmt='%d',delimiter=',')
print('val')
print('val matrix')
print(val_error_matrix)
print('val tp: ', val_tp)
print('val fp: ', val_fp)
print('val fn: ', val_fn)

name, val_acc = test(net, val_data, ctx)

print(val_acc)

name, test_acc = test(net, test_data, ctx)

print(test_acc)
