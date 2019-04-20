import argparse
import mxnet as mx
import os
import sys
from mxnet import gluon, nd
import pathlib
import imgaug as ia
from imgaug import augmenters as iaa


class SingleClassAccuracy(mx.metric.EvalMetric):
    def __init__(self, name='single_class_accuracy',
                 output_names=None, label_names=None):
        super(SingleClassAccuracy, self).__init__(
            name,
            output_names=output_names, label_names=label_names)


    def update(self, label, preds):
        pred_label = nd.argmax(preds, axis=0)
        pred_label = pred_label.asnumpy().astype('int32')
        if pred_label[0] == label:
            self.sum_metric += 1
        self.num_inst += 1

#TODO add augmentation
def augment_tl_snippet(tensor_img):
    """
    augment input img after toTensor, changing
    shape from (C x H x W) to (H x W x C) and back
    :param tensor_img: input tensor with (C x H x W) shape and float32 type.
    :return: output augmented tensor with (C x H x W) shape and float32 type
    """
    seq = iaa.Sequential([
        iaa.Crop(percent=(0, 0.1)),
        iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            rotate=(-25, 25),
            shear=(-8, 8)
        )
    ], random_order=True)
    img = seq.augment_image(nd.transpose(tensor_img, (1,2,0)).asnumpy())
    return nd.transpose(nd.array(img), (2,0,1))


def test(net, val_data, ctx):
    metric = mx.metric.Accuracy()
    for i, batch in enumerate(val_data):
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
        label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)
        outputs = [net(X) for X in data]
        metric.update(label, outputs)
    return metric.get()


def test_on_single_class(net, val_data, ctx, class_ind):
    metric = SingleClassAccuracy()
    for i, batch in enumerate(val_data):
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
        label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)
        outputs = [net(X) for X in data]
        for label_ind in range(len(label[0])):
            if label[0][label_ind] == class_ind:
                metric.update(label[0][label_ind].asnumpy()[0], outputs[0][label_ind])
    return metric.get()


def ensure_folder(dir_fname):
    if not os.path.exists(dir_fname):
        try:
            pathlib.Path(dir_fname).mkdir(parents=True, exist_ok=True)
        except PermissionError:
            print('Unable to create {} directory. Permission denied'.format(dir_fname))

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string, use False or True')
    return s == 'True'

def get_train_argparse():
    parser = argparse.ArgumentParser(
                                    description=('Train a mxnet model'
                                                 'for image classification.')
                                    )
    parser.add_argument(
                        '--train', type=boolean_string,
                        help='set train mode (or only test)'
                        )
    parser.add_argument(
                        '--model-name', type=str,
                        help='set model name'
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
                        '--resume-epoch', type=int, default=0,
                        help='set epoch to finetune from, def=0'
                        )
    parser.add_argument(
                        '--dataset-path', type=str,
                        help=('path to folder with sorted data:'
                              'into test, train, val with classes inside'
                              )
                        )
    parser.add_argument(
                        '--ctx', type=str, default='gpu',
                        help='device to use for training, def = gpu'
                        )
    parser.add_argument(
                        '--device-index', type=int, default=0,
                        help='device index to use for training, def = 0'
                        )
    parser.add_argument(
                        '--batch-size', type=int, default='16',
                        help='set batch size per device for training, def = 16'
                        )
    parser.add_argument(
                        '--epochs', type=int, default='240',
                        help='set number of epochs for training, def = 240'
                        )
    parser.add_argument(
                        '--optimizer', type=str, default='sgd',
                        help='set optimizer type, def = sgd'
                        )
    parser.add_argument(
                        '--lr', type=float, default='0.1',
                        help='set learning rate for optimizer, def=0.001'
                        )
    parser.add_argument(
                        '--lr-decay', type=float, default='0.1',
                        help='set learning rate for optimizer, def=0.1'
                        )
    parser.add_argument(
                        '--lr-decay-interval', type=int, default=20,
                        help='interval of epochs to decay lr'
                        )
    parser.add_argument(
                        '--momentum', type=float, default='0.9',
                        help='momentum value for optimizer, def=0.9'
                        )
    parser.add_argument(
                        '--wd', type=float, default='0.0001',
                        help='set lr, def=0.0001'
                        )
    parser.add_argument(
                        '--save', type=boolean_string,
                        help='bool to save plots and params every log interval'
                        )
    parser.add_argument(
                        '--ssh', type=boolean_string, default=False,
                        help='bool for remote access (no plots saving), def=False'
                        )
    parser.add_argument(
                        '--log-interval', type=int, default=10,
                        help='number of batches to wait before each logging and model saving'
                        )
    parser.add_argument(
                        '--note', type=str,
                        help='help note to add in top of log file if necessary'
                        )

    opt = parser.parse_args()

    return opt
