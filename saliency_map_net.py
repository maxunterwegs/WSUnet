from model import *
from model import resnet50_short
from torch.nn.modules import Upsample


import torch

class CustomNorm(Module):
    ''' essntially a softmax without exp'''
    __constants__ = ['dim']

    def __init__(self, dim=None):
        super(CustomNorm, self).__init__()
        self.dim = dim

    def __setstate__(self, state):
        self.__dict__.update(state)
        if not hasattr(self, 'dim'):
            self.dim = None

    def forward(self, input):
        sum_scores = input.sum(self.dim) + 1e-8
        output = torch.div(input, sum_scores.unsqueeze(1))
        return output


class ConvRelu(nn.Module):
    def __init__(self, in_: int, out: int, activation=torch.nn.ReLU):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_, out, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out)
        self.activation = activation()


    def forward(self, x):
        x = self.conv(x)
        x = self.bn1(x)
        x = self.activation(x)
        return x


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, *input):
        return input


class SaliencyMapNet(nn.Module):
    '''
    :param layers : determines type of resnet. [2,2,2,2] --> 18 layer, [3,4,6,3] --> 50 layer
    :param block: either resnet.BasicBlock or resnet.Bottleneck for deeper networks

    '''
    def __init__(self, num_classes, gr=32, resnet_backbone='ResNet50', dense_config='normal', pretrained=True):
        super(SaliencyMapNet, self).__init__()

        self.num_classes = num_classes
        self.resnet_backbone = resnet_backbone
        self.gr = gr
        if resnet_backbone == 'ResNet18':
            layers = [2, 2, 2, 2]
        elif resnet_backbone =='ResNet50':
            layers = [3, 4, 6, 3]
        else:
            print('given Resnet backbone does not exist')

        if dense_config == 'normal':
            block_config = [6, 12, 24, 16]

        self.pretrained = pretrained

        self.pretrained_resnet = resnet50(pretrained=self.pretrained)
        self._modify_resnet(num_classes)

        # added resnet Block to downsample to 8x 8
        # todo: fix the last layer so that it only downsamples once. done in _modifyresnet
        # too many weights in last layer, reduce planes, and use more blacks.
        # self.layer5 = self._make_layer(inplanes=2048, block=resnet.Bottleneck, planes=1024, blocks=1, stride=2,
        #                                dilate=False)

        ## normal pretrainable resnet:

        # model prior to Unet shape:

        self.d1 = DenseNet(growth_rate=self.gr, block_config=6,
                           num_init_features=256, num_output=256)

        self.d2 = DenseNet(growth_rate=self.gr, block_config=12,
                           num_init_features=512, num_output=512)

        self.d3 = DenseNet(growth_rate=self.gr, block_config=24,
                           num_init_features=1024, num_output=1024)

        # self.d4 = DenseNet(growth_rate=self.gr, block_config=12,
        #                    num_init_features=2048, num_output=2048)


        # upwards path
        # self.upsample0 = Upsample(scale_factor=2, mode='nearest')
        # # self.upsample1 = MyUpsampler(num_init_features =8*self.num_classes, num_out_features=4*self.num_classes) # from [bs, 8k, 8, 8] --> [bs, 4k, 16, 16]
        #
        # self.updense0 = DenseNet(growth_rate=self.gr, block_config=12,
        #                          num_init_features=4096+2048,
        #                          num_output=2048)
        # self.mixdense0 = DenseNet(growth_rate=self.gr, block_config=12,
        #                           num_init_features=2048, num_output=2048)
        self.upsample = Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # self.upsample1 = MyUpsampler(num_init_features =8*self.num_classes, num_out_features=4*self.num_classes) # from [bs, 8k, 8, 8] --> [bs, 4k, 16, 16]

        self.updense1 = DenseNet(growth_rate=self.gr, block_config=2,
                                 num_init_features=2048+1024,
                                 num_output=1024)
        self.mixdense1 = DenseNet(growth_rate=self.gr, block_config=2,
                                  num_init_features=1024, num_output=1024)

        self.updense2 = DenseNet(growth_rate=self.gr, block_config=2,
                                 num_init_features=1024+512,
                                 num_output=512)
        self.mixdense2 = DenseNet(growth_rate=self.gr, block_config=2,
                                  num_init_features=512, num_output=512)

        self.updense3 = DenseNet(growth_rate=self.gr, block_config=2,
                                 num_init_features=512+256,
                                 num_output=256)
        self.mixdense3 = DenseNet(growth_rate=self.gr, block_config=2,
                                  num_init_features=256, num_output=256)

        self.onexone = torch.nn.Conv2d(in_channels=256, out_channels=self.num_classes, kernel_size=1)

        # use a softmax if dataset is mutually exclusive.
        self.sigmoid = nn.Sigmoid()
        # self.softmax = nn.Softmax(dim=1)

        self.pooling = CustomPooling(beta=1, r_0=10, dim=(-1, -2), mode=None)

        self.norm = CustomNorm(1)

    def _modify_resnet(self, num_classes):

        # modification to downsample in first layer
        # change the stride in layer1 conv2 and downsampling to 2.
        self.pretrained_resnet.layer1[0].conv2.stride = (2, 2)
        self.pretrained_resnet.layer1[0].downsample[0].stride = (2, 2)
        # modifications to allow transfer learning
        num_ftrs = self.pretrained_resnet.fc.in_features
        self.pretrained_resnet.fc = torch.nn.Linear(num_ftrs, num_classes)

        # change average pooling to max pooling as in attention paper
        self.pretrained_resnet.avgpool = torch.nn.AdaptiveMaxPool2d(1)


    def _make_layer(self, inplanes, block, planes, blocks, stride=1, dilate=False):

        downsample = None

        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample, norm_layer=nn.BatchNorm2d))
        inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(inplanes, planes, norm_layer=nn.BatchNorm2d))

        return nn.Sequential(*layers)


    def forward(self, x):
        #syage 0

        x, xr1, xr2, xr3, xr4 = self.pretrained_resnet(x)

        #not needed if layer 1 is adjusted.
        # xr5 = self.layer5(xr4)


        xd1 = self.d1(xr1)

        xd2 = self.d2(xr2)

        xd3 = self.d3(xr3)

        #not needed if layer 1 is adjusted.
        # xd4 = self.d4(xr4)

        ## upward path

        # upsampling last feature map using bilinear interpolation
        # not needed if layer 1 downsampling is active
        # up0 = self.upsample0(xr5)
        # cat0 = torch.cat((up0, xd4), dim=1)
        # updense0 = self.updense0(cat0)

        # changed from updense 0 to xr4, if downsampling in layer1
        up1 = self.upsample(xr4)
        #concat the upsampled feature maps with the dense featuremaps from previous layer
        cat1 = torch.cat((up1, xd3), dim=1)
        # densenet to reduce channels to 4k --> [bs x 56 x 16 x 16]
        updense1 =self.updense1(cat1)

        # upsampling last feature map using bilinear interpolation
        up2 = self.upsample(updense1)
        # concat the upsampled feature maps with the dense featuremaps from previous layer
        cat2 = torch.cat((up2, xd2), dim=1)
        # densenet to reduce channels to 2k --> [bs x 28 x 32 x 32]
        updense2 =self.updense2(cat2)

        # upsampling last feature map using bilinear interpolation
        up3 = self.upsample(updense2)
        # concat the upsampled feature maps with the dense featuremaps from previous layer
        cat3 = torch.cat((up3, xd1), dim=1)
        # densenet to reduce channels to k --> [bs x 14 x 64 x 64]
        updense3 = self.updense3(cat3)

        reduced = self.onexone(updense3)
        # final sigmoid layer before the saliency maps
        # todo: add normalization before sigmoid, change sigmoid to alternative to prevent dying gradients, also add pooling
        saliency_map = self.sigmoid(reduced)
        #saliency_map = self.sigmoid(reduced)

        class_scores = self.pooling(saliency_map)

        norm_scores = self.norm(class_scores)

        # log scores for Nllloss on RSNA
        log_scores = torch.log(norm_scores)

        return(saliency_map, log_scores)


class Saliency_simple(nn.Module):
    '''
    Saliency_simple creates a map with weightless decoder

    :param layers : determines type of resnet. [2,2,2,2] --> 18 layer, [3,4,6,3] --> 50 layer
    :param block: either resnet.BasicBlock or resnet.Bottleneck for deeper networks

    '''
    def __init__(self, num_classes, resnet_backbone='ResNet50', pretrained=True, mode=None):
        super(Saliency_simple, self).__init__()
        self.mode = mode
        self.num_classes = num_classes
        self.resnet_backbone = resnet_backbone

        if resnet_backbone == 'ResNet18':
            layers = [2, 2, 2, 2]
        elif resnet_backbone == 'ResNet50':
            layers = [3, 4, 6, 3]
        else:
            print('given Resnet backbone does not exist')

        self.pretrained = pretrained

        # build the network architecture
        # encoder --Resnet50 without final pooling and fc
        self.pretrained_resnet = resnet50(pretrained=self.pretrained)
        self._modify_resnet(num_classes)

        # either set to bilinear or nearest and allign corners
        self.upsample = Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        #reduce to num_classes
        self.onexone = torch.nn.Conv2d(in_channels=3840, out_channels=self.num_classes, kernel_size=1)

        # Decoder path with skip connections, only upsampling and concatenation
        if self.mode is not None:
            if self.mode == 'mode1':
                # # 1x1 conv2@d to reduce filters from 2048 - 1024
                self.onexone1 = torch.nn.Conv2d(2048, 512, 1)
                self.onexone2 = torch.nn.Conv2d(1024, 256, 1)
                self.onexone3 = torch.nn.Conv2d(512, 128, 1)
                self.onexone4 = torch.nn.Conv2d(256, 64, 1)
                self.activation = torch.nn.ReLU()
                # reduce to num_classes
                self.onexone = torch.nn.Conv2d(in_channels=960, out_channels=self.num_classes, kernel_size=1)

            elif self.mode == 'mode2':
                self.onexone1 = torch.nn.Conv2d(2048, 20, 1)
                self.onexone2 = torch.nn.Conv2d(1024, 20, 1)
                self.onexone3 = torch.nn.Conv2d(512, 20, 1)
                self.onexone4 = torch.nn.Conv2d(256, 20, 1)
                self.activation = torch.nn.ReLU()
                # reduce to num_classes
                self.onexone = torch.nn.Conv2d(in_channels=80, out_channels=self.num_classes, kernel_size=1)

        #upsample to increase spatial resolution from 8x8 to 16x16


        # use a softmax if dataset is mutually exclusive.
        self.sigmoid = nn.Sigmoid()
        #todo: make these layers overridable, such that parameters can be set from training script.
        # self.softmax = nn.Softmax(dim=1)

        self.pooling = CustomPooling(beta=1, r_0=10, dim=(-1, -2), mode='Three')

        self.norm = CustomNorm(1)


    def _modify_resnet(self, num_classes):

        # modification to downsample in first layer
        # change the stride in layer1 conv2 and downsampling to 2.
        self.pretrained_resnet.layer1[0].conv2.stride = (2, 2)
        self.pretrained_resnet.layer1[0].downsample[0].stride = (2, 2)
        # modifications to allow transfer learning
        num_ftrs = self.pretrained_resnet.fc.in_features
        self.pretrained_resnet.fc = torch.nn.Linear(num_ftrs, num_classes)

        # change average pooling to max pooling as in attention paper
        self.pretrained_resnet.avgpool = torch.nn.AdaptiveMaxPool2d(1)


    def _make_layer(self, inplanes, block, planes, blocks, stride=1, dilate=False):

        downsample = None

        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample, norm_layer=nn.BatchNorm2d))
        inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(inplanes, planes, norm_layer=nn.BatchNorm2d))

        return nn.Sequential(*layers)


    def forward(self, x):
        #syage 0

        # x is prediction of resnet, which will be ignored
        # xr1 to xr4 are the resnet feature spaces beginning on top.
        x, xr1, xr2, xr3, xr4 = self.pretrained_resnet(x)

        if self.mode is not None:
            if self.mode == 'mode1':
                #reduce resnet filter, than upsample and concat with other reduced resnet features
                x_redu_4 = self.onexone1(xr4)
                x_redu_3 = self.onexone2(xr3)
                x_redu_2 = self.onexone3(xr2)
                x_redu_1 = self.onexone4(xr1)

                x_up_1 = self.upsample(x_redu_4)
                x_cat_1 = torch.cat((x_up_1, x_redu_3), dim=1)

                x_up_2 = self.upsample(x_cat_1)
                x_cat_2 = torch.cat((x_up_2, x_redu_2), dim=1)

                x_up_3 = self.upsample(x_cat_2)
                x_cat_3 = torch.cat((x_up_3, x_redu_1), dim=1)

                reduced = self.onexone(x_cat_3)

            elif self.mode == 'mode2':
                # reduce resnet filter, than upsample and concat with other reduced resnet features
                x_redu_4 = self.onexone1(xr4)
                x_redu_3 = self.onexone2(xr3)
                x_redu_2 = self.onexone3(xr2)
                x_redu_1 = self.onexone4(xr1)

                x_up_1 = self.upsample(x_redu_4)
                x_cat_1 = torch.cat((x_up_1, x_redu_3), dim=1)

                x_up_2 = self.upsample(x_cat_1)
                x_cat_2 = torch.cat((x_up_2, x_redu_2), dim=1)

                x_up_3 = self.upsample(x_cat_2)
                x_cat_3 = torch.cat((x_up_3, x_redu_1), dim=1)

                reduced = self.onexone(x_cat_3)
        else:
            #upsample
            xu1 = self.upsample(xr4)
            cat1 = torch.cat((xu1, xr3), dim=1)


            #upsample 2
            xu2 = self.upsample(cat1)
            cat2 = torch.cat((xu2, xr2), dim=1)


            #upsample 3
            xu3 = self.upsample(cat2)
            cat3 = torch.cat((xu3, xr1), dim=1)


            # final 1x1 convolution to reduce channels to num_classes
            reduced = self.onexone(cat3)

        # final sigmoid layer before the saliency maps
        # todo: change pooling parameters
        saliency_map = self.sigmoid(reduced)
        # saliency_map = self.softmax(reduced)

        class_scores = self.pooling(saliency_map)

        norm_scores = self.norm(class_scores)
        # # insert normalization layer for the class scores such that sum = 1 and in [0,1]
        #
        # # log scores for Nllloss on RSNA
        log_scores = torch.log(norm_scores + 1e-8)

        return(saliency_map, log_scores)


class Saliency_encoder(nn.Module):
    '''
    Saliency_simple creates a map with weightless decoder

    :param layers : determines type of resnet. [2,2,2,2] --> 18 layer, [3,4,6,3] --> 50 layer
    :param block: either resnet.BasicBlock or resnet.Bottleneck for deeper networks

    '''
    def __init__(self, num_classes, resnet_backbone='ResNet50', pretrained=True, mode=None):
        super(Saliency_encoder, self).__init__()
        self.mode = mode
        self.num_classes = num_classes
        self.resnet_backbone = resnet_backbone

        if resnet_backbone == 'ResNet18':
            layers = [2, 2, 2, 2]
        elif resnet_backbone == 'ResNet50':
            layers = [3, 4, 6, 3]
        else:
            print('given Resnet backbone does not exist')

        self.pretrained = pretrained

        # build the network architecture
        # encoder --Resnet50 without final pooling and fc
        self.pretrained_resnet = resnet50(pretrained=self.pretrained)
        self._modify_resnet(num_classes)



    def _modify_resnet(self, num_classes):

        # modification to downsample in first layer
        # change the stride in layer1 conv2 and downsampling to 2.
        self.pretrained_resnet.layer1[0].conv2.stride = (2, 2)
        self.pretrained_resnet.layer1[0].downsample[0].stride = (2, 2)
        # modifications to allow transfer learning
        num_ftrs = self.pretrained_resnet.fc.in_features
        self.pretrained_resnet.fc = torch.nn.Linear(num_ftrs, num_classes)

        # change average pooling to max pooling as in attention paper
        self.pretrained_resnet.avgpool = torch.nn.AdaptiveMaxPool2d(1)


    def _make_layer(self, inplanes, block, planes, blocks, stride=1, dilate=False):

        downsample = None

        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample, norm_layer=nn.BatchNorm2d))
        inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(inplanes, planes, norm_layer=nn.BatchNorm2d))

        return nn.Sequential(*layers)


    def forward(self, x):
        #syage 0

        # x is prediction of resnet, which will be ignored
        # xr1 to xr4 are the resnet feature spaces beginning on top.
        x, xr1, xr2, xr3, xr4 = self.pretrained_resnet(x)

        return(xr4, x)


class Saliency_encoder_grad_cam(nn.Module):
    '''
    Saliency_simple creates a map with weightless decoder

    :param layers : determines type of resnet. [2,2,2,2] --> 18 layer, [3,4,6,3] --> 50 layer
    :param block: either resnet.BasicBlock or resnet.Bottleneck for deeper networks

    '''
    def __init__(self, num_classes, resnet_backbone='ResNet50', pretrained=True, mode=None):
        super(Saliency_encoder_grad_cam, self).__init__()
        self.mode = mode
        self.num_classes = num_classes
        self.resnet_backbone = resnet_backbone

        if resnet_backbone == 'ResNet18':
            layers = [2, 2, 2, 2]
        elif resnet_backbone == 'ResNet50':
            layers = [3, 4, 6, 3]
        else:
            print('given Resnet backbone does not exist')

        self.pretrained = pretrained

        # build the network architecture
        # encoder --Resnet50 without final pooling and fc
        self.pretrained_resnet = resnet50(pretrained=self.pretrained)
        self._modify_resnet(num_classes)



    def _modify_resnet(self, num_classes):

        # modification to downsample in first layer
        # change the stride in layer1 conv2 and downsampling to 2.
        self.pretrained_resnet.layer1[0].conv2.stride = (2, 2)
        self.pretrained_resnet.layer1[0].downsample[0].stride = (2, 2)
        # modifications to allow transfer learning
        num_ftrs = self.pretrained_resnet.fc.in_features
        self.pretrained_resnet.fc = torch.nn.Linear(num_ftrs, num_classes)

        # change average pooling to max pooling as in attention paper
        self.pretrained_resnet.avgpool = torch.nn.AdaptiveMaxPool2d(1)


    def _make_layer(self, inplanes, block, planes, blocks, stride=1, dilate=False):

        downsample = None

        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample, norm_layer=nn.BatchNorm2d))
        inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(inplanes, planes, norm_layer=nn.BatchNorm2d))

        return nn.Sequential(*layers)


    def forward(self, x):
        #syage 0

        # x is prediction of resnet, which will be ignored
        # xr1 to xr4 are the resnet feature spaces beginning on top.
        x, xr1, xr2, xr3, xr4 = self.pretrained_resnet(x, full=False)

        return(xr4, x)


class Saliency_noskip(nn.Module):
    '''
    Saliency_simple creates a map with weightless decoder

    :param layers : determines type of resnet. [2,2,2,2] --> 18 layer, [3,4,6,3] --> 50 layer
    :param block: either resnet.BasicBlock or resnet.Bottleneck for deeper networks

    '''
    def __init__(self, num_classes, resnet_backbone='ResNet50', pretrained=True, mode=None):
        super(Saliency_noskip, self).__init__()
        self.mode = mode
        self.num_classes = num_classes
        self.resnet_backbone = resnet_backbone

        if resnet_backbone == 'ResNet18':
            layers = [2, 2, 2, 2]
        elif resnet_backbone == 'ResNet50':
            layers = [3, 4, 6, 3]
        else:
            print('given Resnet backbone does not exist')

        self.pretrained = pretrained

        # build the network architecture
        # encoder --Resnet50 without final pooling and fc
        self.pretrained_resnet = resnet50(pretrained=self.pretrained)
        self._modify_resnet(num_classes)

        # either set to bilinear or nearest and allign corners
        self.upsample = Upsample(scale_factor=8, mode='bilinear', align_corners=True)

        #reduce to num_classes
        self.onexone = torch.nn.Conv2d(in_channels=2048, out_channels=self.num_classes, kernel_size=1)

        # use a softmax if dataset is mutually exclusive.
        self.sigmoid = nn.Sigmoid()
        #todo: make these layers overridable, such that parameters can be set from training script.
        # self.softmax = nn.Softmax(dim=1)

        self.pooling = CustomPooling(beta=1, r_0=10, dim=(-1, -2), mode='Three')

        self.norm = CustomNorm(1)


    def _modify_resnet(self, num_classes):

        # modification to downsample in first layer
        # change the stride in layer1 conv2 and downsampling to 2.
        self.pretrained_resnet.layer1[0].conv2.stride = (2, 2)
        self.pretrained_resnet.layer1[0].downsample[0].stride = (2, 2)
        # modifications to allow transfer learning
        num_ftrs = self.pretrained_resnet.fc.in_features
        self.pretrained_resnet.fc = torch.nn.Linear(num_ftrs, num_classes)

        # change average pooling to max pooling as in attention paper
        self.pretrained_resnet.avgpool = torch.nn.AdaptiveMaxPool2d(1)


    def _make_layer(self, inplanes, block, planes, blocks, stride=1, dilate=False):

        downsample = None

        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample, norm_layer=nn.BatchNorm2d))
        inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(inplanes, planes, norm_layer=nn.BatchNorm2d))

        return nn.Sequential(*layers)


    def forward(self, x):

        # x is prediction of resnet, which will be ignored
        # xr1 to xr4 are the resnet feature spaces beginning on top.
        _, _, _, _, xr4 = self.pretrained_resnet(x)

        #1x1 convolution instead of at 64 res
        reduced = self.onexone(xr4)
        #upsample
        xu1 = self.upsample(reduced)


        # final sigmoid layer before the saliency maps
        saliency_map = self.sigmoid(xu1)
        # saliency_map = self.softmax(reduced)

        class_scores = self.pooling(saliency_map)

        norm_scores = self.norm(class_scores)
        # # insert normalization layer for the class scores such that sum = 1 and in [0,1]
        log_scores = torch.log(norm_scores + 1e-8)

        if self.mode == 'inference':
            return (xr4, saliency_map, log_scores)
        else:
            return(saliency_map, log_scores)


class SaliencyUNet(nn.Module):
    '''
    Saliency Unet uses u Unet style decoder path on the pretrained ResNet50 backbone

    :param layers : determines type of resnet. [2,2,2,2] --> 18 layer, [3,4,6,3] --> 50 layer
    :param block: either resnet.BasicBlock or resnet.Bottleneck for deeper networks

    '''
    def __init__(self, num_classes, resnet_backbone='ResNet50', pretrained=True, mode=None):
        super(SaliencyUNet, self).__init__()

        self.num_classes = num_classes
        self.mode = mode
        self.resnet_backbone = resnet_backbone

        if resnet_backbone == 'ResNet18':
            layers = [2, 2, 2, 2]
        elif resnet_backbone =='ResNet50':
            layers = [3, 4, 6, 3]
        else:
            print('given Resnet backbone does not exist')

        self.pretrained = pretrained

        # build the network architecture
        # encoder --Resnet50 without final pooling and fc
        self.pretrained_resnet = resnet50(pretrained=self.pretrained)
        self._modify_resnet(num_classes)



        # Decoder path with skip connections

        # 1x1 conv2@d to reduce filters from 2048 - 1024
        self.onexone1 = torch.nn.Conv2d(2048, 1024, 1)
        self.activation = torch.nn.ReLU()
        #upsample to increase spatial resolution from 8x8 to 16x16
        # either set to bilinear or nearest and allign corners
        self.upsample = Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        #concat the upsampled featurespace with the encoder feature space at resolution level 16x16
        # convolution with 3x3 filter and reduction of channels 2048 - 512
        self.conv1 = ConvRelu(2048, 512)

        #upsample 16 --> 32
        # concat with 512 res feature space
        # convolution with 3x3 filter and reduction of channels 1024 - 256
        self.conv2 = ConvRelu(1024, 256)

        #upsample 32 --> 64
        # concat with 512 res feature space
        # convolution with 3x3 filter and reduction of channels 1024 - 256
        self.conv3 = ConvRelu(512, 128)

        #reduce to num_classes
        self.onexone = torch.nn.Conv2d(in_channels=128, out_channels=self.num_classes, kernel_size=1)

        # use a softmax if dataset is mutually exclusive.
        self.sigmoid = nn.Sigmoid()
        #todo: make these layers overridable, such that parameters can be set from training script.
        # self.softmax = nn.Softmax(dim=1)

        self.pooling = CustomPooling(beta=1, r_0=10, dim=(-1, -2), mode='Three')

        self.norm = CustomNorm(1)
        # self.norm = nn.Softmax(dim=1)
        # self.norm = nn.LogSoftmax(dim=1)

    def _modify_resnet(self, num_classes):

        # modification to downsample in first layer
        # change the stride in layer1 conv2 and downsampling to 2.
        self.pretrained_resnet.layer1[0].conv2.stride = (2, 2)
        self.pretrained_resnet.layer1[0].downsample[0].stride = (2, 2)
        # modifications to allow transfer learning
        num_ftrs = self.pretrained_resnet.fc.in_features
        self.pretrained_resnet.fc = torch.nn.Linear(num_ftrs, num_classes)

        # change average pooling to max pooling as in attention paper
        self.pretrained_resnet.avgpool = torch.nn.AdaptiveMaxPool2d(1)


    def _make_layer(self, inplanes, block, planes, blocks, stride=1, dilate=False):

        downsample = None

        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample, norm_layer=nn.BatchNorm2d))
        inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(inplanes, planes, norm_layer=nn.BatchNorm2d))

        return nn.Sequential(*layers)


    def forward(self, x):
        #syage 0

        # x is prediction of resnet, which will be ignored
        # xr1 to xr4 are the resnet feature spaces beginning on top.
        x, xr1, xr2, xr3, xr4 = self.pretrained_resnet(x)

        xr4_red = self.onexone1(xr4)
        xr4_red = self.activation(xr4_red)
        #upsample
        xu1 = self.upsample(xr4_red)
        cat1 = torch.cat((xu1, xr3), dim=1)
        conv1 = self.conv1(cat1)

        #upsample 2
        xu2 = self.upsample(conv1)
        cat2 = torch.cat((xu2, xr2), dim=1)
        conv2 = self.conv2(cat2)

        #upsample 3
        xu3 = self.upsample(conv2)
        cat3 = torch.cat((xu3, xr1), dim=1)
        conv3 = self.conv3(cat3)

        # final 1x1 convolution to reduce channels to num_classes
        reduced = self.onexone(conv3)

        # final sigmoid layer before the saliency maps
        # todo: change pooling parameters
        saliency_map = self.sigmoid(reduced)
        # # saliency_map = self.softmax(reduced)
        #
        class_scores = self.pooling(saliency_map)
        #
        norm_scores = self.norm(class_scores)
        # # insert normalization layer for the class scores such that sum = 1 and in [0,1]
        #
        # # log scores for Nllloss on RSNA
        log_scores = torch.log(norm_scores + 1e-8)

        if self.mode == 'inference':
            return (xr4, saliency_map, log_scores)
        elif self.mode == 'bottle':
            return (x, saliency_map, log_scores)
        else:
            return(saliency_map, log_scores)


class SaliencyUNet_dec(nn.Module):
    '''
    Saliency Unet uses u Unet style decoder path on the pretrained ResNet50 backbone

    :param layers : determines type of resnet. [2,2,2,2] --> 18 layer, [3,4,6,3] --> 50 layer
    :param block: either resnet.BasicBlock or resnet.Bottleneck for deeper networks

    '''
    def __init__(self, num_classes, resnet_backbone='ResNet50', pretrained=True):
        super(SaliencyUNet_dec, self).__init__()

        self.num_classes = num_classes
        self.resnet_backbone = resnet_backbone

        if resnet_backbone == 'ResNet18':
            layers = [2, 2, 2, 2]
        elif resnet_backbone =='ResNet50':
            layers = [3, 4, 6, 3]
        else:
            print('given Resnet backbone does not exist')

        self.pretrained = pretrained

        # build the network architecture
        # encoder --Resnet50 without final pooling and fc
        self.pretrained_resnet = resnet50(pretrained=self.pretrained)
        self._modify_resnet(num_classes)



        # Decoder path with skip connections

        # 1x1 conv2@d to reduce filters from 2048 - 1024
        self.onexone1 = torch.nn.Conv2d(2048, 1024, 1)
        self.activation = torch.nn.ReLU()
        #upsample to increase spatial resolution from 8x8 to 16x16
        # either set to bilinear or nearest and allign corners
        self.upsample = Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        #concat the upsampled featurespace with the encoder feature space at resolution level 16x16
        # convolution with 3x3 filter and reduction of channels 2048 - 512
        self.conv1 = ConvRelu(2048, 512)

        #upsample 16 --> 32
        # concat with 512 res feature space
        # convolution with 3x3 filter and reduction of channels 1024 - 256
        self.conv2 = ConvRelu(1024, 256)

        #upsample 32 --> 64
        # concat with 512 res feature space
        # convolution with 3x3 filter and reduction of channels 512 -128
        self.conv3 = ConvRelu(512, 128)

        ###########################
        # the decoder path w/o upsaampling operations
        self.conv_decoder1 = ConvRelu(128, 64)
        self.conv_decoder2 = ConvRelu(64, 32)
        self.conv_decoder3 = ConvRelu(32, 16)
        self.onexone_decoder = torch.nn.Conv2d(in_channels=16, out_channels=3, kernel_size=1)
        ###########################

        #reduce to num_classes
        self.onexone = torch.nn.Conv2d(in_channels=128, out_channels=self.num_classes, kernel_size=1)

        # use a softmax if dataset is mutually exclusive.
        self.sigmoid = nn.Sigmoid()
        #todo: make these layers overridable, such that parameters can be set from training script.
        # self.softmax = nn.Softmax(dim=1)

        self.pooling = CustomPooling(beta=1, r_0=10, dim=(-1, -2), mode='Three')

        self.norm = CustomNorm(1)
        # self.norm = nn.Softmax(dim=1)
        # self.norm = nn.LogSoftmax(dim=1)

    def _modify_resnet(self, num_classes):

        # modification to downsample in first layer
        # change the stride in layer1 conv2 and downsampling to 2.
        self.pretrained_resnet.layer1[0].conv2.stride = (2, 2)
        self.pretrained_resnet.layer1[0].downsample[0].stride = (2, 2)
        # modifications to allow transfer learning
        num_ftrs = self.pretrained_resnet.fc.in_features
        self.pretrained_resnet.fc = torch.nn.Linear(num_ftrs, num_classes)

        # change average pooling to max pooling as in attention paper
        self.pretrained_resnet.avgpool = torch.nn.AdaptiveMaxPool2d(1)


    def _make_layer(self, inplanes, block, planes, blocks, stride=1, dilate=False):

        downsample = None

        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample, norm_layer=nn.BatchNorm2d))
        inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(inplanes, planes, norm_layer=nn.BatchNorm2d))

        return nn.Sequential(*layers)


    def forward(self, x):
        #syage 0

        # x is prediction of resnet, which will be ignored
        # xr1 to xr4 are the resnet feature spaces beginning on top.
        x, xr1, xr2, xr3, xr4 = self.pretrained_resnet(x)

        xr4_red = self.onexone1(xr4)
        xr4_red = self.activation(xr4_red)
        #upsample
        xu1 = self.upsample(xr4_red)
        cat1 = torch.cat((xu1, xr3), dim=1)
        conv1 = self.conv1(cat1)

        #upsample 2
        xu2 = self.upsample(conv1)
        cat2 = torch.cat((xu2, xr2), dim=1)
        conv2 = self.conv2(cat2)

        #upsample 3
        xu3 = self.upsample(conv2)
        cat3 = torch.cat((xu3, xr1), dim=1)
        conv3 = self.conv3(cat3)

        ########################
        #decoder path
        dec = self.upsample(conv3)
        dec = self.conv_decoder1(dec)

        dec = self.upsample(dec)
        dec = self.conv_decoder2(dec)

        dec = self.upsample(dec)
        dec = self.conv_decoder3(dec)
        dec = self.onexone_decoder(dec)
        # why the hell a sigmoid ?
        # dec = self.sigmoid(dec)
        ########################

        # final 1x1 convolution to reduce channels to num_classes
        reduced = self.onexone(conv3)

        # final sigmoid layer before the saliency maps
        # todo: change pooling parameters
        saliency_map = self.sigmoid(reduced)
        # saliency_map = self.softmax(reduced)

        class_scores = self.pooling(saliency_map)

        norm_scores = self.norm(class_scores)
        # # insert normalization layer for the class scores such that sum = 1 and in [0,1]
        #
        # # log scores for Nllloss on RSNA
        log_scores = torch.log(norm_scores + 1e-8)

        return(saliency_map, log_scores, dec)


class L_cut(nn.Module):
    '''
    The pretrained ResNet is only used until the 64x64 resolution level, the feature space is the forwarded to a
    1x1 convolution that produces the saliency maps.

    :param layers : determines type of resnet. [2,2,2,2] --> 18 layer, [3,4,6,3] --> 50 layer
    :param block: either resnet.BasicBlock or resnet.Bottleneck for deeper networks

    '''
    def __init__(self, num_classes, resnet_backbone='ResNet50', pretrained=True):
        super(L_cut, self).__init__()

        self.num_classes = num_classes
        self.resnet_backbone = resnet_backbone

        if resnet_backbone == 'ResNet18':
            layers = [2, 2, 2, 2]
        elif resnet_backbone =='ResNet50':
            layers = [3, 4, 6, 3]
        else:
            print('given Resnet backbone does not exist')

        self.pretrained = pretrained

        # build the network architecture
        # encoder --Resnet50 without final pooling and fc
        self.pretrained_resnet = resnet50(pretrained=self.pretrained)
        self._modify_resnet(num_classes)




        self.conv = ConvRelu(256, 128)

        #reduce to num_classes
        self.onexone = torch.nn.Conv2d(in_channels=128, out_channels=self.num_classes, kernel_size=1)

        # use a softmax if dataset is mutually exclusive.
        self.sigmoid = nn.Sigmoid()
        #todo: make these layers overridable, such that parameters can be set from training script.
        # self.softmax = nn.Softmax(dim=1)

        self.pooling = CustomPooling(beta=1, r_0=10, dim=(-1, -2), mode='Three')

        self.norm = CustomNorm(1)
        # self.norm = nn.Softmax(dim=1)
        # self.norm = nn.LogSoftmax(dim=1)

    def _modify_resnet(self, num_classes):

        # modification to downsample in first layer
        # change the stride in layer1 conv2 and downsampling to 2.
        self.pretrained_resnet.layer1[0].conv2.stride = (2, 2)
        self.pretrained_resnet.layer1[0].downsample[0].stride = (2, 2)
        # modifications to allow transfer learning
        num_ftrs = self.pretrained_resnet.fc.in_features
        self.pretrained_resnet.fc = torch.nn.Linear(num_ftrs, num_classes)

        # change average pooling to max pooling as in attention paper
        self.pretrained_resnet.avgpool = torch.nn.AdaptiveMaxPool2d(1)


    def _make_layer(self, inplanes, block, planes, blocks, stride=1, dilate=False):

        downsample = None

        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample, norm_layer=nn.BatchNorm2d))
        inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(inplanes, planes, norm_layer=nn.BatchNorm2d))

        return nn.Sequential(*layers)


    def forward(self, x):
        #syage 0

        # x is prediction of resnet, which will be ignored
        # xr1 to xr4 are the resnet feature spaces beginning on top.
        x, xr1, _, _, _ = self.pretrained_resnet(x)


        conv = self.conv(xr1)

        # final 1x1 convolution to reduce channels to num_classes
        reduced = self.onexone(conv)

        # final sigmoid layer before the saliency maps
        # todo: change pooling parameters
        saliency_map = self.sigmoid(reduced)
        # saliency_map = self.softmax(reduced)

        class_scores = self.pooling(saliency_map)

        norm_scores = self.norm(class_scores)
        # # insert normalization layer for the class scores such that sum = 1 and in [0,1]
        #
        # # log scores for Nllloss on RSNA
        log_scores = torch.log(norm_scores + 1e-8)

        return(saliency_map, log_scores)


class Saliency_gradientboost(nn.Module):
    '''
    Saliency Unet uses u Unet style decoder path on the pretrained ResNet50 backbone

    :param layers : determines type of resnet. [2,2,2,2] --> 18 layer, [3,4,6,3] --> 50 layer
    :param block: either resnet.BasicBlock or resnet.Bottleneck for deeper networks

    '''
    def __init__(self, num_classes, resnet_backbone='ResNet50', pretrained=True, mode=None):
        super(Saliency_gradientboost, self).__init__()

        self.num_classes = num_classes
        self.mode = mode
        self.resnet_backbone = resnet_backbone

        if resnet_backbone == 'ResNet18':
            layers = [2, 2, 2, 2]
        elif resnet_backbone =='ResNet50':
            layers = [3, 4, 6, 3]
        else:
            print('given Resnet backbone does not exist')

        self.pretrained = pretrained

        # build the network architecture
        # encoder --Resnet50 without final pooling and fc
        self.pretrained_resnet = resnet50(pretrained=self.pretrained)
        self._modify_resnet(num_classes)



        # Decoder path with skip connections

        # 1x1 conv2@d to reduce filters from 2048 - 1024
        self.onexone1 = torch.nn.Conv2d(2048, 1024, 1)
        self.activation = torch.nn.ReLU()
        #upsample to increase spatial resolution from 8x8 to 16x16
        # either set to bilinear or nearest and allign corners
        self.upsample = Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        #concat the upsampled featurespace with the encoder feature space at resolution level 16x16
        # convolution with 3x3 filter and reduction of channels 2048 - 512
        self.conv1 = ConvRelu(2048, 512)

        #upsample 16 --> 32
        # concat with 512 res feature space
        # convolution with 3x3 filter and reduction of channels 1024 - 256
        self.conv2 = ConvRelu(1024, 256)

        #upsample 32 --> 64
        # concat with 512 res feature space
        # convolution with 3x3 filter and reduction of channels 1024 - 256
        self.conv3 = ConvRelu(512, 128)

        #reduce to num_classes
        self.onexone = torch.nn.Conv2d(in_channels=128, out_channels=self.num_classes, kernel_size=1)

        # use a softmax if dataset is mutually exclusive.
        self.sigmoid = nn.Sigmoid()
        #todo: make these layers overridable, such that parameters can be set from training script.
        # self.softmax = nn.Softmax(dim=1)

        self.pooling = CustomPooling(beta=1, r_0=10, dim=(-1, -2), mode='Three')

        self.norm = CustomNorm(1)
        # self.norm = nn.Softmax(dim=1)
        # self.norm = nn.LogSoftmax(dim=1)

    def _modify_resnet(self, num_classes):

        # modification to downsample in first layer
        # change the stride in layer1 conv2 and downsampling to 2.
        self.pretrained_resnet.layer1[0].conv2.stride = (2, 2)
        self.pretrained_resnet.layer1[0].downsample[0].stride = (2, 2)
        # modifications to allow transfer learning
        num_ftrs = self.pretrained_resnet.fc.in_features
        self.pretrained_resnet.fc = torch.nn.Linear(num_ftrs, num_classes)

        # change average pooling to max pooling as in attention paper
        self.pretrained_resnet.avgpool = torch.nn.AdaptiveMaxPool2d(1)


    def _make_layer(self, inplanes, block, planes, blocks, stride=1, dilate=False):

        downsample = None

        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample, norm_layer=nn.BatchNorm2d))
        inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(inplanes, planes, norm_layer=nn.BatchNorm2d))

        return nn.Sequential(*layers)


    def forward(self, x, full=True):
        #syage 0
        # todo: modify such that gradient boosted skips are possible

        # todo: calculate forward path til bottleneck, and compute loss w.r.t. final feature maps
        if full == False:
            # output
            x, xr1, xr2, xr3, xr4 = self.pretrained_resnet(x, False)
            return x, xr1, xr2, xr3, xr4

        else:
            # todo: calculate forward path all the way with boosted skip connections, compute gradients w.r.t. all parameters, update model
            # get the boosted activations: x is a list
            inp = x[0]
            xb_1 = x[1]
            xb_2 = x[2]
            xb_3 = x[3]
            xb_4 = x[4]

            y, xr1, xr2, xr3, xr4 = self.pretrained_resnet(inp)

            xr4_red = self.onexone1(xb_4*xr4)
            xr4_red = self.activation(xr4_red)
            #upsample
            xu1 = self.upsample(xr4_red)
            cat1 = torch.cat((xu1, xb_3*xr3), dim=1)
            conv1 = self.conv1(cat1)

            #upsample 2
            xu2 = self.upsample(conv1)
            cat2 = torch.cat((xu2, xb_2*xr2), dim=1)
            conv2 = self.conv2(cat2)

            #upsample 3
            xu3 = self.upsample(conv2)
            cat3 = torch.cat((xu3, xb_1*xr1), dim=1)
            conv3 = self.conv3(cat3)

            # final 1x1 convolution to reduce channels to num_classes
            reduced = self.onexone(conv3)

            # final sigmoid layer before the saliency maps
            # todo: change pooling parameters
            saliency_map = self.sigmoid(reduced)
            # saliency_map = self.softmax(reduced)

            class_scores = self.pooling(saliency_map)

            norm_scores = self.norm(class_scores)
            # # insert normalization layer for the class scores such that sum = 1 and in [0,1]
            #
            # # log scores for Nllloss on RSNA
            log_scores = torch.log(norm_scores + 1e-8)

            if self.mode == 'inference':
                return (saliency_map, log_scores)
            else:
                return(y, saliency_map, log_scores)


class ModResNet50(nn.Module):
    def __init__(self, num_classes, resnet_backbone='ResNet50', pretrained=True):
        super(ModResNet50, self).__init__()

        self.num_classes = num_classes
        self.resnet_backbone = resnet_backbone
        if resnet_backbone == 'ResNet18':
            layers = [2, 2, 2, 2]
        elif resnet_backbone == 'ResNet50':
            layers = [3, 4, 6, 3]
        else:
            print('given Resnet backbone does not exist')

        self.pretrained = pretrained

        #load resnet50 from torch pretrained on imagenet
        self.pretrained_resnet = resnet50_short(pretrained=self.pretrained)
        # modfiy resnet, deletes the pooling and fully connected layer.
        self._modify_resnet()

        # added resnet Block to downsample to 8x 8
        # todo: fix the last layer so that it only downsamples once.
        self.layer5 = self._make_layer(inplanes=2048, block=Bottleneck, planes=1024, blocks=1, stride=2,
                                       dilate=False)

        # classification head
        self.avgpool = nn.AdaptiveMaxPool2d((1, 1))
        self.fc = nn.Linear(4096, self.num_classes)

    def _modify_resnet(self):

        # delete the last two layers (fc and avg_pooling)
        self.pretrained_resnet = nn.Sequential(*list(self.pretrained_resnet.children())[:-2])

        # # modifications to allow transfer learning
        # num_ftrs = self.pretrained_resnet.fc.in_features
        # self.pretrained_resnet.fc = torch.nn.Linear(num_ftrs, 15)
        #
        # # changw average pooling to max pooling as in attention paper
        # self.pretrained_resnet.avgpool = torch.nn.AdaptiveMaxPool2d(1)

    def _make_layer(self, inplanes, block, planes, blocks, stride=1, dilate=False):

        downsample = None

        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample, norm_layer=nn.BatchNorm2d))
        inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(inplanes, planes, norm_layer=nn.BatchNorm2d))

        return nn.Sequential(*layers)

    def forward(self, x):
        # syage 0

        xr4 = self.pretrained_resnet(x)
        xr5 = self.layer5(xr4)

        pooled = self.avgpool(xr5)
        pooled = pooled.reshape(x.size(0), -1)
        fc = self.fc(pooled)
        return fc


# testing
# model = ModResNet50(15)
# input = torch.randn(1, 3, 512, 512)
# output= model(input)
