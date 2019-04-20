import argparse
import os
import mxnet as mx

parser = argparse.ArgumentParser(
                                description=('script for sorting snippets by network classes')
                                )
parser.add_argument(
                    '--params-dir', type=str,
                    help='set path to dir with params & struct'
                    )
parser.add_argument(
                    '--net-name', type=str,
                    help='set net name'
                    )
opt = parser.parse_args()

params_dir  = opt.params_dir
net_name = opt.net_name

ctx = [mx.cpu()]
sym_path = os.path.join(params_dir, '{}-symbol.json'.format(net_name))
net = mx.symbol.load(sym_path)

v = mx.viz.plot_network(symbol=net)
v.view()
