import torch.nn as nn
import torch.nn.functional as F
from typing import Type, Any, Callable, Union, List, Optional
from torchvision._internally_replaced_utils import load_state_dict_from_url

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-b627a593.pth',
    "resnet50": "https://download.pytorch.org/models/resnet50-0676ba61.pth",
    'resnet101': 'https://download.pytorch.org/models/resnet101-63fe2227.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-394f9c45.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def conv3x3(inplanes, outplanes, stride=1, groups=1, dilation=1, padding_mode='zeros'):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        inplanes,
        outplanes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
        padding_mode=padding_mode
    )


def conv1x1(inplanes, outplanes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
            self,
            inplanes,
            planes,
            stride=1,
            shortcut=None,
            groups=1,
            base_width=64,
            dilation=1,
            norm_layer=None,
    ):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.conv1 = conv3x3(inplanes, planes, stride=1)
        self.upsample = None
        if stride != 1:
            self.upsample = nn.Upsample(scale_factor=stride, mode="bilinear")
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)

        self.shortcut = shortcut
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        if self.upsample is not None:
            out = self.upsample(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.shortcut is not None:
            identity = self.shortcut(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(
            self,
            inplanes,
            planes,
            stride=1,
            shortcut=None,
            groups=1,
            base_width=64,
            dilation=1,
            norm_layer=None,
    ):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.upsample layers upsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.upsample = None
        if stride != 1:
            self.upsample = nn.Upsample(scale_factor=stride, mode="bilinear")
        self.conv2 = conv3x3(width, width, 1, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = shortcut
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        if self.upsample is not None:
            out = self.upsample(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.shortcut is not None:
            identity = self.shortcut(x)

        out += identity
        out = self.relu(out)

        return out


class ResNetDecoder(nn.Module):
    def __init__(
            self,
            inplanes,
            block,
            layers,
            groups=1,
            width_per_group=64,
            norm_layer=None,
            initializer=None,
    ):
        super(ResNetDecoder, self).__init__()

        self.inplanes = inplanes[0]

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.dilation = 1
        layer_planes = [64, 128, 256, 512]
        layer_strides = [2, 2, 2, 1]

        self.groups = groups
        self.base_width = width_per_group
        self.layer4 = self._make_layer(
            block, layer_planes[3], layers[3], stride=layer_strides[3]
        )
        self.layer3 = self._make_layer(
            block, layer_planes[2], layers[2], stride=layer_strides[2]
        )
        self.layer2 = self._make_layer(
            block, layer_planes[1], layers[1], stride=layer_strides[1]
        )
        self.layer1 = self._make_layer(
            block, layer_planes[0], layers[0], stride=layer_strides[0]
        )
        # self.upsample1 = nn.Upsample(scale_factor=2, mode="bilinear")
        # self.conv1 = nn.Conv2d(
        #     self.inplanes, 64, kernel_size=3, stride=1, padding=1, bias=False
        # )
        # self.bn1 = norm_layer(64)
        # self.relu = nn.ReLU(inplace=True)
        # self.conv2 = nn.Conv2d(64, 3, kernel_size=1, stride=1, bias=True)
        # self.upsample2 = nn.Upsample(scale_factor=2, mode="bilinear")

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        norm_layer = self._norm_layer
        shortcut = None
        previous_dilation = self.dilation
        if stride != 1 or self.inplanes != planes * block.expansion:
            shortcut = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride=1),
                nn.Upsample(scale_factor=stride, mode="bilinear"),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                shortcut,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    @property
    def layer0(self):
        return nn.Sequential(
            self.upsample1, self.conv1, self.bn1, self.relu, self.conv2, self.upsample2,
        )

    def forward(self, x):
        # f4 = self.layer4(x)ads
        f3 = self.layer3(x)
        f2 = self.layer2(f3)
        f1 = self.layer1(f2)
        return f1, f2, f3

    def get_consist_weight(self, state_dict):
        model_state_dict = self.state_dict()
        for k in state_dict:
            if k in model_state_dict:
                if state_dict[k].shape != model_state_dict[k].shape:
                    print(f"Skip loading parameter: {k}, "
                          f"required shape: {model_state_dict[k].shape}, "
                          f"loaded shape: {state_dict[k].shape}")
                else:
                    model_state_dict[k] = state_dict[k]
            else:
                print(f"Dropping parameter {k}")
        return model_state_dict


def _resnet(
        arch: str,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        pretrained: bool,
        progress: bool,
        inplanes: List[int],
        **kwargs: Any
):
    model = ResNetDecoder(block=block, layers=layers, inplanes=inplanes, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        state_dict = model.get_consist_weight(state_dict)
        model.load_state_dict(state_dict)
    return model


def resnet34_decoder(pretrained: bool = False, progress: bool = True, **kwargs: Any):
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet50_decoder(pretrained: bool = False, progress: bool = True, **kwargs: Any):
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def wide_resnet50_decoder(pretrained: bool = False, progress: bool = True, **kwargs: Any):
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnext50_32x4d_decoder(pretrained: bool = False, progress: bool = True, **kwargs: Any):
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)

