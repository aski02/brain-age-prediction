import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from collections import OrderedDict


class DenseNet(nn.Module):
    """3D-DenseNet-BC model class."""

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, in_channels=1,
                 memory_efficient=False):
        super(DenseNet, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv3d(in_channels, num_init_features,
                                kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm3d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool3d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each DenseBlock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers, num_input_features=num_features,
                bn_size=bn_size, growth_rate=growth_rate,
                memory_efficient=memory_efficient
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate

            if i != len(block_config) - 1:
                trans = _Transition(
                    num_input_features=num_features,
                    num_output_features=num_features // 2
                )
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        self.num_features = num_features

        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = F.adaptive_avg_pool3d(features, 1)
        out = torch.flatten(out, 1)
        return out  # Do not squeeze, keep batch dimension


def _bn_function_factory(norm, relu, conv):
    def bn_function(*inputs):
        concatenated_features = torch.cat(inputs, 1)
        bottleneck_output = conv(relu(norm(concatenated_features)))
        return bottleneck_output
    return bn_function


class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, memory_efficient=False):
        super(_DenseLayer, self).__init__()
        self.norm1 = nn.BatchNorm3d(num_input_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(
            num_input_features, bn_size * growth_rate,
            kernel_size=1, stride=1, bias=False
        )
        self.norm2 = nn.BatchNorm3d(bn_size * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(
            bn_size * growth_rate, growth_rate,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.memory_efficient = memory_efficient

    def forward(self, *prev_features):
        bn_function = _bn_function_factory(self.norm1, self.relu1, self.conv1)
        if self.memory_efficient and any(
            prev_feature.requires_grad for prev_feature in prev_features
        ):
            bottleneck_output = cp.checkpoint(bn_function, *prev_features)
        else:
            bottleneck_output = bn_function(*prev_features)

        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        return new_features


class _DenseBlock(nn.ModuleDict):
    def __init__(self, num_layers, num_input_features,
                 bn_size, growth_rate, memory_efficient=False):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate, bn_size, memory_efficient
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for layer in self.values():
            new_features = layer(*features)
            features.append(new_features)
        return torch.cat(features, 1)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm3d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module(
            'conv',
            nn.Conv3d(
                num_input_features, num_output_features,
                kernel_size=1, stride=1, bias=False
            )
        )
        self.add_module('pool', nn.AvgPool3d(kernel_size=2, stride=2))


def densenet121(**kwargs):
    return DenseNet(
        growth_rate=32, block_config=(6, 12, 24, 16),
        num_init_features=64, **kwargs
    )


class SupConDenseNet(nn.Module):
    """DenseNet with a projection head for contrastive learning."""

    def __init__(self, head='mlp', feat_dim=128):
        super().__init__()
        self.encoder = densenet121()
        dim_in = self.encoder.num_features

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


class SupRegDenseNet(nn.Module):
    """DenseNet with regression head."""

    def __init__(self, dropout=False, gender_input=False):
        super().__init__()
        self.encoder = densenet121()
        dim_in = self.encoder.num_features
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
        # layers.append(nn.Linear(512, 1))
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


class SupRegDenseNetMultiTask(nn.Module):
    """DenseNet for multitask learning (age regression and gender classification)."""

    def __init__(self, dropout=False):
        super().__init__()
        self.encoder = densenet121()
        dim_in = self.encoder.num_features

        layers_age = []
        layers_gender = []

        if dropout:
            layers_age.append(nn.Dropout(0.5))
            layers_gender.append(nn.Dropout(0.5))

        # Age regressor
        layers_age.extend([
            nn.Linear(dim_in, 1),
            # nn.ReLU(inplace=True),
            # nn.Linear(512, 1)
        ])
        self.regressor = nn.Sequential(*layers_age)

        # Gender classifier
        layers_gender.extend([
            nn.Linear(dim_in, 1),
            # nn.ReLU(inplace=True),
            # nn.Linear(128, 1)
        ])
        self.gender_classifier = nn.Sequential(*layers_gender)

    def forward(self, x):
        feats = self.encoder(x)
        age_output = self.regressor(feats)
        gender_output = self.gender_classifier(feats)
        return age_output, gender_output

    def features(self, x):
        return self.encoder(x)
