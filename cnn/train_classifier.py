# brief: скрипт обучения классификатора
# author: pichugin
# usage: python3 trm_tl_train_classifier.py --train True --model-name
# cifar_wide_resnet_16_2 --net-name zf_tl_classifier.001
# --params-dir /home/pichugin/datasets/models
# --dataset-path /home/pichugin/datasets/zf_dataset_splitted  --ctx gpu
# --batch-size 100 --epochs 300 --save True --ssh False
# --lr 0.1 --lr-decay 0.1 --lr-decay-interval 10 --device-index 1

import time
from mxnet import init
from mxnet.gluon.data.vision import transforms
from mxnet import autograd as ag
import matplotlib
matplotlib.use('Agg')
from gluoncv.utils import TrainingHistory
import sys
import cifar_wide_resnet
from tools import *

def get_data_raw(dataset_path, batch_size, num_workers):
    train_path = os.path.join(dataset_path, 'train')
    val_path = os.path.join(dataset_path, 'val')
    test_path = os.path.join(dataset_path, 'test')

    transform_train = transforms.Compose([
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_data = gluon.data.DataLoader(
        gluon.data.vision.ImageFolderDataset(train_path).transform_first(transform_train),
        batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_data = gluon.data.DataLoader(
        gluon.data.vision.ImageFolderDataset(val_path).transform_first(transform_test),
        batch_size=batch_size, shuffle=False, num_workers = num_workers)
    test_data = gluon.data.DataLoader(
        gluon.data.vision.ImageFolderDataset(test_path).transform_first(transform_test),
        batch_size=batch_size, shuffle=False, num_workers = num_workers)
    return train_data, val_data, test_data


opt = get_train_argparse()
train = opt.train
model_name = opt.model_name
net_name = opt.net_name
params_dir = opt.params_dir
dataset_path = opt.dataset_path
ctx_type = opt.ctx
device_index = opt.device_index
batch_size = opt.batch_size
epochs = opt.epochs
resume_epoch = opt.resume_epoch
optimizer = opt.optimizer
lr = opt.lr
lr_decay = opt.lr_decay
lr_decay_interval = opt.lr_decay_interval
momentum = opt.momentum
wd = opt.wd
save = opt.save
log_interval = opt.log_interval
ssh = opt.ssh
num_gpus = 1
num_workers = os.cpu_count()

if ctx_type == 'cpu':
    ctx = [mx.cpu()]
elif ctx_type == 'gpu':
    ctx = [mx.gpu(device_index)]

classes_list = sorted(os.listdir(os.path.join(dataset_path, 'train')))
classes = len(classes_list)

net = cifar_wide_resnet.cifar_wideresnet16_10(classes=classes)
net.collect_params().initialize(init.Xavier(magnitude=2.24), ctx = ctx)
net.collect_params().reset_ctx(ctx)
net.hybridize()

train_data, val_data, test_data = get_data_raw(dataset_path, batch_size, num_workers)

num_batch = len(train_data)
params_path = os.path.join(params_dir, net_name)
ensure_folder(params_path)

if train and save:
    dict_file = os.path.join(params_path, net_name + '_classes.txt')
    dict = open(dict_file, 'w')
    for class_name in classes_list:
        dict.write(class_name + '\n')
    log_file = os.path.join(params_path, net_name + '_' + model_name + '_logs.txt')
    if os.path.isfile(log_file):
        rewrite = input('log file exists want to rewrite (y/n): ')
        if rewrite != 'y':
            print('Change network name')
            raise NameError
    log = open(log_file, 'w')
    for arg in range(len(sys.argv)):
        log.write(sys.argv[arg] + '\n')

if resume_epoch > 0:
    params_f = os.path.join(params_path,
                            '{}_{:03d}__{}.params'.format(
                                                net_name,resume_epoch,model_name))
    if not os.path.isfile(params_f):
        print('Check params path to finetune', params_f)
        raise FileNotFoundError
    net.load_parameters(params_f)

optimizer_params = {'wd': wd, 'momentum': momentum, 'learning_rate': lr}
trainer = gluon.Trainer(net.collect_params(), optimizer, optimizer_params)
train_metric = mx.metric.Accuracy()
L = gluon.loss.SoftmaxCrossEntropyLoss()
#TODO rewrite plot history to use remotely without gluoncv
train_history = TrainingHistory(['training-error', 'validation-error'])
print("Batch size", batch_size)
print('Workon dataset_path: ', dataset_path)
print('Model Name: ', model_name)
print('Params saving in: ', params_path)
print('Start training loop')
best_val_acc = 0
save_best_val_acc = False
lr_decay_count = 0
if train:
    assert resume_epoch < epochs, ('Error in finetune resume_epoch < epochs')
    for epoch in range(resume_epoch, epochs):
        if epoch % lr_decay_interval == 0 and epoch != 0:
            trainer.set_learning_rate(trainer.learning_rate*lr_decay)
            lr_decay_count += 1

        tic = time.time()
        train_loss = 0
        train_metric.reset()

        for i, batch in enumerate(train_data):
            data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
            label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)
            # AutoGrad
            with ag.record():
                outputs = [net(X) for X in data]
                loss = [L(yhat, y) for yhat, y in zip(outputs, label)]
            # Backpropagation
            for l in loss:
                l.backward()
            # Optimize
            trainer.step(batch_size)

            train_loss += sum([l.mean().asscalar() for l in loss]) / len(loss)
            train_metric.update(label, outputs)

        name, train_acc = train_metric.get()
        train_loss /= num_batch

        name, val_acc = test(net, val_data, ctx)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_best_val_acc = True
        else:
            save_best_val_acc = False

        train_history.update([1-train_acc, 1-val_acc])

        scores = ('[Epoch {:d}] Train-acc: {:.3f}, loss: {:.3f} | Val-acc: {:.3f} | time: {:.1f}').format(
                epoch, train_acc, train_loss, val_acc, time.time() - tic)
        print(scores)
        if save:
            log.write(scores + '\n' )
        if epoch != 0:
            if (epoch+1) % log_interval == 0 or save_best_val_acc:
                if save_best_val_acc:
                    val_acc_save_m = 'Params saved on epoch {}, new best val acc founded'.format(epoch+1)
                    print(val_acc_save_m)
                    if save:
                        log.write(val_acc_save_m)
                    for i in range(classes):
                        per_class_acc = '{}={}'.format(classes_list[i], test_on_single_class(net, val_data, ctx, i)[1])
                        print(per_class_acc)
                        if save:
                            log.write(per_class_acc)
                else:
                    print('Params saved on epoch {}'.format(epoch+1))
                net.save_parameters(os.path.join(
                        params_path,
                        '{:s}_{:03d}__{}.params'.format(net_name, epoch+1, model_name))
                        )
                if not ssh:
                    train_history.plot(save_path=(os.path.join(
                               params_path,
                               '{:s}_{:03d}__{}.png'.format(net_name, epoch+1, model_name))
                               ))
    if not ssh:
        train_history.plot()

# name, test_acc = test(net, test_data, ctx)
# test_score = '[Finished] Test-acc: {:.3f}'.format(test_acc)
# print(test_score)
# if train:
#     log.write(test_score + '\n')
#     log.close()
