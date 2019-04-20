import os
from mxnet.gluon.block import HybridBlock
from mxnet.gluon import nn
from mxnet import cpu

def _conv3x3(channels, stride, in_channels):
    return nn.Conv2D(channels, kernel_size=3, strides=stride, padding=1,
                     use_bias=False, in_channels=in_channels)

class CIFARBasicBlockV2(HybridBlock):
    def __init__(self, channels, stride, downsample=False, drop_rate=0.0, in_channels=0, **kwargs):
        super(CIFARBasicBlockV2, self).__init__(**kwargs)
        self.bn1 = nn.BatchNorm()
        self.conv1 = _conv3x3(channels, stride, in_channels)
        self.bn2 = nn.BatchNorm()
        self.conv2 = _conv3x3(channels, 1, channels)
        self.droprate = drop_rate
        if downsample:
            self.downsample = nn.Conv2D(channels, 1, stride, use_bias=False,
                                        in_channels=in_channels)
        else:
            self.downsample = None

    def hybrid_forward(self, F, x):
        residual = x
        x = self.bn1(x)
        x = F.Activation(x, act_type='relu')
        if self.downsample:
            residual = self.downsample(x)
        x = self.conv1(x)

        x = self.bn2(x)
        x = F.Activation(x, act_type='relu')

        if self.droprate > 0:
            x = F.Dropout(x, self.droprate)
        x = self.conv2(x)

        return x + residual


class Softmax(HybridBlock):
    def __init__(self, **kwargs):
         super(Softmax, self).__init__(**kwargs)

    def hybrid_forward(self, F, x):
        return F.softmax(x)

class CIFARWideResNet(HybridBlock):
    def __init__(self, block, layers, channels, drop_rate, classes=10, **kwargs):
        super(CIFARWideResNet, self).__init__(**kwargs)
        assert len(layers) == len(channels) - 1
        with self.name_scope():
            self.features = nn.HybridSequential(prefix='')
            self.features.add(nn.BatchNorm(scale=False, center=False))
            self.features.add(nn.Conv2D(channels[0], 3, 1, 1, use_bias=False))
            self.features.add(nn.BatchNorm())

            in_channels = channels[0]
            for i, num_layer in enumerate(layers):
                stride = 1 if i == 0 else 2
                self.features.add(self._make_layer(block, num_layer, channels[i+1], drop_rate,
                                                   stride, i+1, in_channels=in_channels))
                in_channels = channels[i+1]
            self.features.add(nn.BatchNorm())
            self.features.add(nn.Activation('relu'))
            self.features.add(nn.GlobalAvgPool2D())
            self.features.add(nn.Flatten())
            self.features.add(nn.Dense(classes))
            self.output = Softmax()

    def _make_layer(self, block, layers, channels, drop_rate, stride, stage_index, in_channels=0):
        layer = nn.HybridSequential(prefix='stage%d_'%stage_index)
        with layer.name_scope():
            layer.add(block(channels, stride, channels != in_channels, drop_rate,
                            in_channels=in_channels, prefix=''))
            for _ in range(layers-1):
                layer.add(block(channels, 1, False, drop_rate, in_channels=channels, prefix=''))
        return layer

    def hybrid_forward(self, F, x):
        x = self.features(x)
        x = self.output(x)
        return x

# Constructor
def get_cifar_wide_resnet(num_layers, width_factor=1, drop_rate=0.0, ctx=cpu(), **kwargs):
    assert (num_layers - 4) % 6 == 0
    n = (num_layers - 4) // 6
    layers = [n] * 3
    channels = [16, 16*width_factor, 32*width_factor, 64*width_factor]
    net = CIFARWideResNet(CIFARBasicBlockV2, layers, channels, drop_rate, **kwargs)
    return net

def cifar_wideresnet16_2(**kwargs):
    return get_cifar_wide_resnet(16, 2, **kwargs)

def cifar_wideresnet16_10(**kwargs):
    return get_cifar_wide_resnet(16, 10, **kwargs)
