import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding."""
    return nn.Conv3d(
        in_planes, out_planes, kernel_size=3, stride=stride,
        padding=dilation, groups=groups, bias=False, dilation=dilation
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution."""
    return nn.Conv3d(
        in_planes, out_planes, kernel_size=1, stride=stride, bias=False
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self, inplanes, planes, stride=1, downsample=None,
        groups=1, base_width=64, dilation=1, norm_layer=None
    ):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d

        if groups != 1 or base_width != 64:
            raise ValueError(
                'BasicBlock only supports groups=1 and base_width=64'
            )
        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BasicBlock"
            )

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self, inplanes, planes, stride=1, downsample=None,
        groups=1, base_width=64, dilation=1, norm_layer=None
    ):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d

        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(
            width, width, stride, groups, dilation
        )
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(
            width, planes * self.expansion
        )
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """3D ResNet model class."""

    def __init__(
        self, block, layers, in_channels=1, zero_init_residual=False,
        groups=1, width_per_group=64, replace_stride_with_dilation=None,
        norm_layer=None, initial_kernel_size=7
    ):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d

        self.inplanes = 64
        self.dilation = 1

        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        self.groups = groups
        self.base_width = width_per_group

        initial_stride = 2 if initial_kernel_size == 7 else 1
        padding = (initial_kernel_size - initial_stride + 1) // 2

        self.conv1 = nn.Conv3d(
            in_channels, self.inplanes,
            kernel_size=initial_kernel_size, stride=initial_stride,
            padding=padding, bias=False
        )
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(
            kernel_size=3, stride=2, padding=1
        )

        self.layer1 = self._make_layer(
            block, 64, layers[0]
        )
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2,
            dilate=replace_stride_with_dilation[0]
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2,
            dilate=replace_stride_with_dilation[1]
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2,
            dilate=replace_stride_with_dilation[2]
        )
        self.avgpool = nn.AdaptiveAvgPool3d(1)

        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu'
                )
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize last BN
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(
        self, block, planes, blocks, stride=1, dilate=False
    ):
        norm_layer = nn.BatchNorm3d
        downsample = None
        previous_dilation = self.dilation

        if dilate:
            self.dilation *= stride
            stride = 1

        if (
            stride != 1
            or self.inplanes != planes * block.expansion
        ):
            downsample = nn.Sequential(
                conv1x1(
                    self.inplanes, planes * block.expansion,
                    stride
                ),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample,
                self.groups, self.base_width, previous_dilation,
                norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes, planes, groups=self.groups,
                    base_width=self.base_width, dilation=self.dilation,
                    norm_layer=norm_layer
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(
            self.relu(self.bn1(self.conv1(x)))
        )
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = torch.flatten(self.avgpool(x), 1)
        return x


def resnet18(**kwargs):
    return ResNet(
        BasicBlock, [2, 2, 2, 2], **kwargs
    )


def resnet34(**kwargs):
    return ResNet(
        BasicBlock, [3, 4, 6, 3], **kwargs
    )


def resnet50(**kwargs):
    return ResNet(
        Bottleneck, [3, 4, 6, 3], **kwargs
    )


def resnet101(**kwargs):
    return ResNet(
        Bottleneck, [3, 4, 23, 3], **kwargs
    )


model_dict = {
    'resnet18': [resnet18, 512],
    'resnet34': [resnet34, 512],
    'resnet50': [resnet50, 2048],
    'resnet101': [resnet101, 2048],
}


class SupConResNet(nn.Module):
    """ResNet with a projection head for contrastive learning."""

    def __init__(self, name='resnet50', head='mlp', feat_dim=128):
        super().__init__()
        model_fun, dim_in = model_dict[name]
        self.encoder = model_fun()
        if head == 'linear':
            self.head = nn.Linear(dim_in, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )
        else:
            raise NotImplementedError(f'head not supported: {head}')

    def forward(self, x):
        feat = self.encoder(x)
        feat = F.normalize(self.head(feat), dim=1)
        return feat

    def features(self, x):
        return self.encoder(x)


class SupRegResNet(nn.Module):
    """ResNet with regression head."""

    def __init__(self, name='resnet18', dropout=False, gender_input=False):
        super().__init__()
        model_fun, dim_in = model_dict[name]
        self.encoder = model_fun()
        self.gender_input = gender_input

        if gender_input:
            dim_in += 1  # For gender feature

        layers = []
        if dropout:
            layers.append(nn.Dropout(0.5))
        layers.extend([
            nn.Linear(dim_in, 1),
            # nn.ReLU(inplace=True),
        ])
        # if dropout:
        #    layers.append(nn.Dropout(0.5))
        # layers.append(nn.Linear(256, 1))
        self.regressor = nn.Sequential(*layers)

    def forward(self, x, gender=None):
        feats = self.encoder(x)
        if self.gender_input and gender is not None:
            gender = gender.view(-1, 1).float()
            feats = torch.cat((feats, gender), dim=1)
        output = self.regressor(feats)
        return output, feats

    def features(self, x):
        return self.encoder(x)


class SupRegResNetMultiTask(nn.Module):
    """ResNet for multitask learning (age regression and gender classification)."""

    def __init__(self, name='resnet18', dropout=False):
        super().__init__()
        model_fun, dim_in = model_dict[name]
        self.encoder = model_fun()

        layers_age = []
        layers_gender = []

        if dropout:
            layers_age.append(nn.Dropout(0.5))
            layers_gender.append(nn.Dropout(0.5))

        # Age regressor
        layers_age.extend([
            nn.Linear(dim_in, 1),
            # nn.ReLU(inplace=True),
            # nn.Linear(256, 1)
        ])
        self.regressor = nn.Sequential(*layers_age)

        # Gender classifier
        layers_gender.extend([
            nn.Linear(dim_in, 1),
            # nn.ReLU(inplace=True),
            # nn.Linear(128, 1)  # Output is a single logit
        ])
        self.gender_classifier = nn.Sequential(*layers_gender)

    def forward(self, x):
        feats = self.encoder(x)
        age_output = self.regressor(feats)
        gender_output = self.gender_classifier(feats)
        return age_output, gender_output

    def features(self, x):
        return self.encoder(x)
