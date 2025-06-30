import numpy as np
import torch
import torch.nn as nn
import os
import sys
from torch.autograd import Variable
import functools
import torch.nn.functional as F
# from util.image_pool import ImagePool
# from . import networks
# import matplotlib.pyplot as plt

class BaseModel(torch.nn.Module):
    def name(self):
        return 'BaseModel'
    def initialize(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        self.save_path = opt.checkpoint_path
        print("pix2pixHD BaseModel initialize", self.save_path)
    def forward(self):
        pass
# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation=nn.ReLU(True), use_dropout=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim),
                       activation]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

class GlobalGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d,
                 upsample=False, padding_type='reflect'):
        assert (n_blocks >= 0)
        super(GlobalGenerator, self).__init__()
        activation = nn.ReLU(True)

        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        ### downsample
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2), activation]

        ### resnet blocks
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]

        ### upsample
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)

            if upsample:
                model += [nn.Upsample(scale_factor=2),
                          nn.Conv2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=1, padding=1)]
            else:
                model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1,
                                             output_padding=1)]

            model += [norm_layer(int(ngf * mult / 2)), activation]

        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


class SPADE(nn.Module):
    def __init__(self, norm_nc, label_nc, norm):
        super().__init__()

        ks = 3
        self.param_free_norm = norm(norm_nc)

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 128

        pw = ks // 2
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)

    def forward(self, x, segmap):

        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta

        return out


class SpadeDown(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, kernel_size=3, stride=1, padding=0, activation=nn.ReLU,
                 input_nc=1, spectral=False):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        # if spectral:
        #     self.conv = spectral_norm(self.conv)
        self.norm = SPADE(out_channels, input_nc, norm_layer)
        self.activation = activation

    def forward(self, x):
        x_in, segmap = x
        x1 = self.conv(x_in)
        x2 = self.norm(x1, segmap)
        x_out = self.activation(x2)
        return x_out, segmap


class ResnetBlockSpade(nn.Module):
    def __init__(self, dim, norm_layer, activation=nn.ReLU(True), use_dropout=False, input_nc=1, spectral=False):
        super().__init__()

        self.padding1 = nn.ReflectionPad2d(1)
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=0)
        # if spectral:
        #     self.conv1 = spectral_norm(self.conv1)
        self.norm1 = SPADE(dim, input_nc, norm_layer)
        self.activation = activation

        self.use_dropout = use_dropout
        if self.use_dropout:
            self.drop_out = nn.Dropout(0.5)

        self.padding2 = nn.ReflectionPad2d(1)

        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=0)
        # if spectral:
        #     self.conv2 = spectral_norm(self.conv2)
        self.norm2 = SPADE(dim, input_nc, norm_layer)

    def forward(self, x):
        x_in, segmap = x
        x1 = self.padding1(x_in)
        x2 = self.conv1(x1)
        x3 = self.norm1(x2, segmap)
        x4 = self.activation(x3)

        if self.use_dropout:
            x4 = self.drop_out(x4)

        x5 = self.padding2(x4)
        x6 = self.conv2(x5)
        x7 = self.norm2(x6, segmap)

        x_out = x_in + x7
        return x_out, segmap


class SpadeUp(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, kernel_size=3, stride=2, padding=1, activation=nn.ReLU,
                 input_nc=1, spectral=False):
        super().__init__()

        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, padding)
        # if spectral:
        #     self.conv = spectral_norm(self.conv)
        self.norm = SPADE(out_channels, input_nc, norm_layer)
        self.activation = activation

    def forward(self, x):
        x_in, segmap = x
        x1 = self.conv(x_in)
        x2 = self.norm(x1, segmap)
        x_out = self.activation(x2)
        return x_out, segmap


class GlobalGeneratorSpade(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d,
                 upsample=False, padding_type='reflect', spectral=False):
        assert (n_blocks >= 0)
        super(GlobalGeneratorSpade, self).__init__()

        activation = nn.ReLU(True)
        self.reflection_pad1 = nn.ReflectionPad2d(3)
        model = [SpadeDown(input_nc, ngf, norm_layer, kernel_size=7, padding=0, activation=activation, input_nc=input_nc, spectral=spectral)]

        for i in range(n_downsampling):
            mult = 2**i
            model.append(SpadeDown(ngf * mult, ngf * mult * 2, norm_layer, kernel_size=3, stride=2, padding=1,
                                   activation=activation, input_nc=input_nc, spectral=spectral))

        ### resnet blocks
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model.append(ResnetBlockSpade(ngf * mult, norm_layer, activation=activation, input_nc=input_nc, spectral=spectral))

        ### upsample
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model.append(SpadeUp(ngf * mult, int(ngf * mult / 2), norm_layer, kernel_size=3, stride=2, padding=1,
                                 activation=activation, input_nc=input_nc, spectral=spectral))

        self.model = nn.Sequential(*model)

        self.reflection_pad2 = nn.ReflectionPad2d(3)
        self.out_conv = nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)
        self.out_activation = nn.Tanh()

    def forward(self, x):
        x1 = self.reflection_pad1(x)

        x2 = self.model((x1, x))

        x_out = self.reflection_pad2(x2[0])
        x_out = self.out_conv(x_out)
        x_out = self.out_activation(x_out)

        return x_out


class LocalEnhancerSpade(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=32, n_downsample_global=3, n_blocks_global=9,
                 n_local_enhancers=1, n_blocks_local=3, norm_layer=nn.BatchNorm2d, upsample=False,
                 padding_type='reflect', spectral=False):
        super(LocalEnhancerSpade, self).__init__()
        self.n_local_enhancers = n_local_enhancers

        ###### global generator model #####
        ngf_global = ngf * (2 ** n_local_enhancers)
        model_global = GlobalGeneratorSpade(input_nc, output_nc, ngf_global, n_downsample_global, n_blocks_global,
                                       norm_layer, upsample, spectral=spectral)
        model_global = [m for m in model_global.children()][:-3]  # get rid of final convolution layers
        self.global_pad = model_global[0]
        self.model = model_global[-1]

        ###### local enhancer layers #####
        for n in range(1, n_local_enhancers + 1):
            ### downsample
            ngf_global = ngf * (2 ** (n_local_enhancers - n))
            model_downsample = [nn.ReflectionPad2d(3),
                                SpadeDown(input_nc, ngf, norm_layer, kernel_size=7, padding=0, activation=nn.ReLU(True),
                                          input_nc=input_nc, spectral=spectral),
                                SpadeDown(ngf_global, ngf_global * 2, norm_layer, kernel_size=3, stride=2, padding=1,
                                          activation=nn.ReLU(True), input_nc=input_nc, spectral=spectral)]

            ### residual blocks
            model_upsample = []
            for i in range(n_blocks_local):
                model_upsample += [ResnetBlockSpade(ngf_global * 2, norm_layer, activation=nn.ReLU(True),
                                                    input_nc=input_nc, spectral=spectral)]

            ### upsample
            model_upsample += [SpadeUp(ngf_global * 2, ngf_global, norm_layer, kernel_size=3, stride=2, padding=1,
                                                      activation=nn.ReLU(True), input_nc=input_nc, spectral=spectral)]

            ### final convolution
            model_final_conv = []
            if n == n_local_enhancers:
                model_final_conv += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
                                   nn.Tanh()]

            setattr(self, 'model' + str(n) + '_1', nn.ModuleList(model_downsample))
            setattr(self, 'model' + str(n) + '_2', nn.ModuleList(model_upsample))
            setattr(self, 'model' + str(n) + '_3', nn.ModuleList(model_final_conv))

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def forward(self, input):
        ### create input pyramid
        input_downsampled = [input]
        for i in range(self.n_local_enhancers):
            input_downsampled.append(self.downsample(input_downsampled[-1]))

        ### output at coarest level
        pad = self.global_pad(input_downsampled[-1])
        output_prev, _ = self.model((pad, input_downsampled[-1]))

        ### build up one layer at a time
        for n_local_enhancers in range(1, self.n_local_enhancers + 1):
            model_downsample = getattr(self, 'model' + str(n_local_enhancers) + '_1')
            model_upsample = getattr(self, 'model' + str(n_local_enhancers) + '_2')
            model_final_conv = getattr(self, 'model' + str(n_local_enhancers) + '_3')
            input_i = input_downsampled[self.n_local_enhancers - n_local_enhancers]

            for down in model_downsample:
                if isinstance(down, nn.ReflectionPad2d):
                    down_out = down(input_i)
                else:
                    down_out, _ = down((down_out, input_i))

            up_out = down_out + output_prev

            for up in model_upsample:
                up_out, _ = up((up_out, input_i))

        for final_conv in model_final_conv:
            up_out = final_conv(up_out)
        return up_out


class LocalEnhancer(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=32, n_downsample_global=3, n_blocks_global=9,
                 n_local_enhancers=1, n_blocks_local=3, norm_layer=nn.BatchNorm2d, upsample=False,
                 padding_type='reflect'):
        super(LocalEnhancer, self).__init__()
        self.n_local_enhancers = n_local_enhancers

        ###### global generator model #####
        ngf_global = ngf * (2 ** n_local_enhancers)
        model_global = GlobalGenerator(input_nc, output_nc, ngf_global, n_downsample_global, n_blocks_global,
                                       norm_layer, upsample).model
        model_global = [model_global[i] for i in
                        range(len(model_global) - 3)]  # get rid of final convolution layers
        self.model = nn.Sequential(*model_global)

        ###### local enhancer layers #####
        for n in range(1, n_local_enhancers + 1):
            ### downsample
            ngf_global = ngf * (2 ** (n_local_enhancers - n))
            model_downsample = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf_global, kernel_size=7, padding=0),
                                norm_layer(ngf_global), nn.ReLU(True),
                                nn.Conv2d(ngf_global, ngf_global * 2, kernel_size=3, stride=2, padding=1),
                                norm_layer(ngf_global * 2), nn.ReLU(True)]
            ### residual blocks
            model_upsample = []
            for i in range(n_blocks_local):
                model_upsample += [ResnetBlock(ngf_global * 2, padding_type=padding_type, norm_layer=norm_layer)]

            ### upsample
            if upsample:
                model_upsample += [nn.Upsample(scale_factor=2),
                                   nn.Conv2d(ngf_global * 2, ngf_global, kernel_size=3, stride=1, padding=1)]
            else:
                model_upsample += [nn.ConvTranspose2d(ngf_global * 2, ngf_global, kernel_size=3, stride=2, padding=1,
                                                      output_padding=1)]
            model_upsample += [norm_layer(ngf_global), nn.ReLU(True)]

            ### final convolution
            if n == n_local_enhancers:
                model_upsample += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
                                   nn.Tanh()]

            setattr(self, 'model' + str(n) + '_1', nn.Sequential(*model_downsample))
            setattr(self, 'model' + str(n) + '_2', nn.Sequential(*model_upsample))

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def forward(self, input):
        ### create input pyramid
        input_downsampled = [input]
        for i in range(self.n_local_enhancers):
            input_downsampled.append(self.downsample(input_downsampled[-1]))

        output_prev = self.model(input_downsampled[-1])

        for n_local_enhancers in range(1, self.n_local_enhancers + 1):
            model_downsample = getattr(self, 'model' + str(n_local_enhancers) + '_1')
            model_upsample = getattr(self, 'model' + str(n_local_enhancers) + '_2')
            input_i = input_downsampled[self.n_local_enhancers - n_local_enhancers]
            output_prev = model_upsample(model_downsample(input_i) + output_prev)
        return output_prev

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def define_G(input_nc, output_nc, ngf, netG, n_downsample_global=3, n_blocks_global=9, n_local_enhancers=1,
             n_blocks_local=3, norm='instance', upsample=False, gpu_ids=[]):
    norm_layer = get_norm_layer(norm_type=norm)
    if netG == 'local':
        netG = LocalEnhancer(input_nc, output_nc, ngf, n_downsample_global, n_blocks_global,
                                  n_local_enhancers, n_blocks_local, norm_layer, upsample)
    elif netG == 'local_spade':
        netG = LocalEnhancerSpade(input_nc, output_nc, ngf, n_downsample_global, n_blocks_global,
                                  n_local_enhancers, n_blocks_local, norm_layer, upsample)

    # print(netG)
    # 원래 아래 세 줄 이였음
    #if len(gpu_ids) > 0:
    #    assert(torch.cuda.is_available())
    #    netG.cuda(gpu_ids[0])
        
    num_gpus = torch.cuda.device_count()
    for gpu_id in range(num_gpus):
        gpu_name = torch.cuda.get_device_name(gpu_id)
        if 'NVIDIA' in gpu_name:
            netG.cuda(gpu_id)
            break
    
    netG.apply(weights_init)
    return netG

class Pix2PixHDModel(BaseModel):
    def name(self):
        return 'Pix2PixHDModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        print('are you running ?')
        input_nc = opt.label_nc if opt.label_nc != 0 else opt.input_nc
        self.cuda1 = opt.cuda1

        netG_input_nc = input_nc
        self.netG = define_G(netG_input_nc, opt.output_nc, opt.ngf, opt.netG,
                                      opt.n_downsample_global, opt.n_blocks_global, opt.n_local_enhancers,
                                      opt.n_blocks_local, opt.norm, opt.upsample, gpu_ids=self.gpu_ids)

    def forward(self, x):
        # Handle different device types
        if torch.cuda.is_available() and len(self.gpu_ids) > 0:
            return self.netG(Variable(x).cuda(self.cuda1))
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() and len(self.gpu_ids) > 0:
            return self.netG(Variable(x).to('mps'))
        else:
            return self.netG(Variable(x))

class BuildModel(torch.nn.Module):
    def __init__(self, opt):
        super(BuildModel, self).__init__()
        self.opt = opt
        self.model = Pix2PixHDModel()
        self.model.initialize(opt)

    def forward(self, inp):
        return self.model(inp)
