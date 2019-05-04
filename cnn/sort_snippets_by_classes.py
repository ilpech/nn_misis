#brief: script for sorting snippets by network classes, creates classes dirs in writing directory, copies snippets there
#author: pichugin
#usage: python3 sort_snippets_by_classes.py --net-name pillars_detector.001 --epoch 140
#       --params-dir /datasets/mxnet_data/models --dataset-path /datasets/mobis/datasets_/balanced/balanced_row_dataset
#       --writing-dir /datasets/mobis/datasets_/balanced/pvsp_test --softmax-output True --with-score True
import argparse
import os, shutil, sys
import pathlib
import mxnet as mx
from mxnet import gluon, image, nd
from mxnet.gluon.data.vision import transforms

from tools import *

parser = argparse.ArgumentParser(
                                description=('script for sorting snippets by network classes')
                                )
parser.add_argument(
                    '--net-name', type=str,
                    help='set network name'
                    )
parser.add_argument(
                    '--epoch', type=int,
                    help='epoch of net to use'
                    )
parser.add_argument(
                    '--params-dir', type=str,
                    help='set path to dir with params & struct'
                    )
parser.add_argument(
                    '--dataset-path', type=str,
                    help='set path to dir with images to classify'
                    )
parser.add_argument(
                    '--writing-dir', type=str,
                    help='set path to dir where create classes and put sorted images'
                    )
parser.add_argument(
                    '--softmax-output', type=boolean_string,
                    help='set true if model has softmax output'
                    )
parser.add_argument(
                    '--with-scores', type=boolean_string,
                    help='set true if imgs will be sorted by scores'
                    )

opt = parser.parse_args()

params_dir  = opt.params_dir
dataset_path = opt.dataset_path
writing_dir  = opt.writing_dir
net_name     = opt.net_name
epoch        = opt.epoch
softmax_output = opt.softmax_output
with_score = opt.with_scores

dict_path = os.path.join(params_dir, net_name + '_classes.txt')
with open(dict_path) as f:
    content = f.readlines()
class_names = [x.strip() for x in content]

print('Creating writing dir: {}'.format(writing_dir))
ensure_folder(writing_dir)

for class_name in class_names:
    class_path = os.path.join(writing_dir, class_name)
    print('Creating class dir: {}'.format(class_path))
    ensure_folder(class_path)

ctx = [mx.cpu()]

params_path = os.path.join(params_dir, '{}-{:04d}.params'.format(
                                                            net_name, epoch))
sym_path = os.path.join(params_dir, '{}-symbol.json'.format(net_name))
net = gluon.nn.SymbolBlock.imports(sym_path, ['data'], params_path, ctx=ctx)

def transform(img):
    input_size = (220, 170)
    # h = len(img)
    # w = len(img[0])
    transform = transforms.Compose([
        # transforms.CenterCrop(w,w),
        transforms.Resize(input_size, keep_ratio = True),
        transforms.ToTensor()
    ])
    return transform(img)

def transform_trm(img):
        class TrmTransform(mx.gluon.block.Block):
            def __init__(self):
                super(TrmTransform, self).__init__()

            def forward(self, x):
                if isinstance(x, mx.ndarray.ndarray.NDArray):
                    h, w, c = x.shape
                    ratio_h_w = 1.0 * h / w
                    if ratio_h_w <= 1.5:
                        return mx.image.center_crop(x, (w , w))[0]
                    elif ratio_h_w > 1.5:
                        return mx.image.center_crop(x, (h , h))[0]
                else:
                    print("Error in img transform")
                    raise TypeError
        inp_shape = (64,64)
        transform_test = transforms.Compose([
            TrmTransform(),
            transforms.Resize(inp_shape),
            transforms.ToTensor(),
        ])
        return transform_test(img)

print('Process images')
for season in os.listdir(dataset_path):
    path_to_season = os.path.join(dataset_path,season)
    for episode in os.listdir(path_to_season):
        path_to_episode = os.path.join(path_to_season,episode)
        for snippet in os.listdir(path_to_episode):
            path_to_snippet = os.path.join(path_to_episode, snippet)
            try:
                img = image.imread(path_to_snippet)
            except:
                print('error during reading img', path_to_snippet)
                continue
            img = transform(img)
            pred = net(img.expand_dims(axis=0))
            if softmax_output:
                score = pred*255
            else:
                score = pred.softmax()*255
            ind = nd.argmax(pred, axis=1).astype('int')
            img_score = int(score[0][ind.asscalar()].asscalar())
            class_path = os.path.join(writing_dir, class_names[ind.asscalar()])
            ensure_folder(class_path)
            if with_score:
                img_score_path = os.path.join(class_path, '{:03d}'.format(img_score))
                ensure_folder(img_score_path)
                dst = os.path.join(img_score_path, snippet)
            else:
                dst = os.path.join(class_path, snippet)
            src = path_to_snippet
            shutil.copy(src, dst)
