# %load cnn/viz_model.py
import argparse
import os
import mxnet as mx

def showModel(params_dir, net_name):
    sym_path = os.path.join(params_dir, '{}-symbol.json'.format(net_name))
    net = mx.symbol.load(sym_path)

    mx.viz.print_summary(symbol=net)

print("Модель для работы с большими картинками")
showModel("res/big-pics", "gel_cls.002")