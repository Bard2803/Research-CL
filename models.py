import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SimpleCNNGrayScale(nn.Module):
    """
    Convolutional Neural Network

    **Example**::

        >>> from avalanche.models import SimpleCNN
        >>> n_classes = 10 # e.g. MNIST
        >>> model = SimpleCNN(num_classes=n_classes)
        >>> print(model) # View model details
    """

    def __init__(self, num_classes=10):
        super(SimpleCNNGrayScale, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.25),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.25),
            nn.Conv2d(64, 64, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool2d(1),
            nn.Dropout(p=0.25),
        )
        self.classifier = nn.Sequential(nn.Linear(64, num_classes))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class SimpleCNNWithBN(nn.Module):
    """
    Convolutional Neural Network

    **Example**::

        >>> from avalanche.models import SimpleCNN
        >>> n_classes = 10 # e.g. MNIST
        >>> model = SimpleCNN(num_classes=n_classes)
        >>> print(model) # View model details
    """

    def __init__(self, num_classes=10, num_channel=1, image_size=28):
        super(SimpleCNNWithBN, self).__init__()

        self.layersize = 32
        finalSize = int(math.floor(image_size / (2 * 2 * 2 * 2)))
        self.outSize = finalSize * finalSize * self.layersize

        self.features = nn.Sequential(
            nn.Conv2d(num_channel, self.layersize, kernel_size=3, stride=1, padding=1), #conv1
            nn.BatchNorm2d(self.layersize),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(self.layersize, self.layersize, kernel_size=3, stride=1, padding=1),          #conv2
            nn.BatchNorm2d(self.layersize),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #nn.Dropout(p=0.25),
            nn.Conv2d(self.layersize, self.layersize, kernel_size=3, stride=1, padding=1),          #conv3
            nn.BatchNorm2d(self.layersize),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(self.layersize, self.layersize, kernel_size=3, stride=1, padding=1),        # conv4
            nn.BatchNorm2d(self.layersize),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #nn.Dropout(p=0.25),
            #nn.Conv2d(64, 64, kernel_size=1, padding=0),
            #nn.ReLU(inplace=True),
            #nn.AdaptiveMaxPool2d(1),
            #nn.Dropout(p=0.25),
        )
        self.classifier = nn.Sequential(nn.Linear(self.outSize, num_classes, bias=False))

    def forward(self, x):
        x = self.features(x)
        #x = x.view(x.size(0), -1)
        x = x.view(x.size()[0], -1)
        x = self.classifier(x)
        return x


## Relevance mapping Networks based models (RMN)
class AConv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False,
                 datasets=1,
                 same_init=False, Beta=False):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation,
                         bias=bias)

        self.multi = False
        self.adjx = nn.ParameterList(
            [nn.Parameter(torch.Tensor(self.weight.shape).uniform_(0, 1), requires_grad=True) for i in
             range(datasets)])
        if same_init:
            for ix in range(1, datasets):
                self.adjx[ix] = self.adjx[0]
        if Beta:
            self.Beta = Beta
            self.beta = nn.Parameter(torch.Tensor(1).random_(70, 130))
            self.initial_beta = self.beta
        else:
            self.Beta = False

    def soft_round(self, x, beta=100):
        return (1 / (1 + torch.exp(-(beta * (x - 0.5)))))

    def forward(self, input, dataset, round_=False):
        if round_:
            if self.Beta:
                return F.conv2d(input,
                                (self.soft_round(self.adjx[dataset], self.beta).round().float()) * self.weight,
                                bias=self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation)
            return F.conv2d(input, (self.soft_round(self.adjx[dataset]).round().float()) * self.weight,
                            bias=self.bias,
                            stride=self.stride, padding=self.padding, dilation=self.dilation)

        if self.Beta:
            return F.conv2d(input, self.soft_round(self.adjx[dataset], self.beta) * self.weight, bias=self.bias,
                            stride=self.stride, padding=self.padding, dilation=self.dilation)
        return F.conv2d(input, self.soft_round(self.adjx[dataset]) * self.weight, bias=self.bias,
                        stride=self.stride,
                        padding=self.padding, dilation=self.dilation)

    def get_nconnections(self, dataset):
        try:
            return (self.soft_round(self.adjx[dataset]) > 0.1).sum()
        except:
            return ("DatasetError")

    def l1_loss(self, dataset):
        try:
            return self.soft_round(self.adjx[dataset]).sum()
        except:
            return ("DatasetError")

    def beta_val(self):
        return self.initial_beta.item(), self.beta.item()

def convLayer(in_channels, out_channels, keep_prob=0.0):
    """3*3 convolution with padding,ever time call it the output size become half"""
    cnn_seq = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, 1, 1),
        nn.ReLU(True),
        nn.BatchNorm2d(out_channels),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Dropout(keep_prob)
    )
    return cnn_seq

def AconvLayer(in_channels, out_channels, keep_prob=0.0):
    """3*3 convolution with padding,ever time call it the output size become half"""
    cnn_seq = nn.Sequential(
        AConv2d(in_channels, out_channels, 3, 1, 1),
        nn.ReLU(True),
        nn.BatchNorm2d(out_channels),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Dropout(keep_prob)
    )
    return cnn_seq

class ALinear(nn.Linear):

    def __init__(self, in_features, out_features, bias=False, datasets=1, same_init=False, Beta=False, multi=False):
        super().__init__(in_features, out_features, bias)

        self.adjx = nn.ParameterList(
            [nn.Parameter(torch.Tensor(self.weight.shape).uniform_(0, 1), requires_grad=True) for i in range(datasets)])
        if same_init:
            for ix in range(1, datasets):
                self.adjx[ix] = self.adjx[0]

        self.multi = multi
        if self.multi:
            self.weightx = nn.ParameterList(
                [nn.Parameter(torch.Tensor(self.weight), requires_grad=True) for i in range(datasets)])
            for ix in range(datasets):
                self.adjx[ix] = nn.Parameter(torch.ones(*self.adjx[ix].shape), requires_grad=False)

        # if Beta:
        #     self.Beta = Beta
        #     self.beta = nn.Parameter(torch.Tensor(1).random_(70, 130))
        #     self.initial_beta = self.beta

    def soft_round(self, x, beta=100):
        return (1 / (1 + torch.exp(-(beta * (x - 0.5)))))

    def forward(self, input, dataset, round_=False):
        if self.multi:
            weight = self.weightx[dataset]
        else:
            weight = self.weight

        if round_:
            try:
                return F.linear(input, (self.soft_round(self.adjx[dataset]).round().float()) * weight, self.bias)
            except Exception as e:
                print("DatasetError: {}".format(e))

        try:
            return F.linear(input, self.soft_round(self.adjx[dataset]) * weight, self.bias)
        except Exception as e:
            print("DatasetError: {}".format(e))

    def get_nconnections(self, dataset):
        try:
            return (self.soft_round(self.adjx[dataset]) > 0.1).sum()
        except:
            return ("DatasetError")

    def l1_loss(self, dataset):
        try:
            return self.soft_round(self.adjx[dataset]).sum()
        except:
            return ("DatasetError")

    # def beta_val(self):
    #     return self.initial_beta.item(), self.beta.item()

class ClassifierCNN2D(nn.Module):
    def __init__(self, layer_size=32, output_shape=55, num_channels=1, keep_prob=1.0, image_size=28, tasks=1,
                 bn_boole=False):
        super(ClassifierCNN2D, self).__init__()
        """
        Build a CNN to produce embeddings
        :param layer_size:64(default)
        :param num_channels:
        :param keep_prob:
        :param image_size:
        """
        self.conv1 = AConv2d(num_channels, layer_size, 3, 1, 1, datasets=tasks)
        self.conv2 = AConv2d(layer_size, layer_size, 3, 1, 1, datasets=tasks)
        self.conv3 = AConv2d(layer_size, layer_size, 3, 1, 1, datasets=tasks)
        self.conv4 = AConv2d(layer_size, layer_size, 3, 1, 1, datasets=tasks)

        self.bn1 = nn.ModuleList([nn.BatchNorm2d(layer_size) for j in range(tasks)])
        self.bn2 = nn.ModuleList([nn.BatchNorm2d(layer_size) for j in range(tasks)])
        self.bn3 = nn.ModuleList([nn.BatchNorm2d(layer_size) for j in range(tasks)])
        self.bn4 = nn.ModuleList([nn.BatchNorm2d(layer_size) for j in range(tasks)])

        self.mp1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.mp2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.mp3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.mp4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.do = nn.Dropout(keep_prob)
        self.relu = nn.ReLU()
        self.sm = nn.Sigmoid()

        finalSize = int(math.floor(image_size / (2 * 2 * 2 * 2)))
        self.outSize = finalSize * finalSize * layer_size

        self.linear = ALinear(self.outSize, output_shape, datasets=tasks, multi=True)
        self._weight_init()

    def _weight_init(self):
        for m in self.modules():
            if isinstance(m, AConv2d):
                m.weight.data.normal_(0, 1e-2)
                if m.bias is not None:
                    m.bias.data.normal_(0.5, 1e-2)
            elif isinstance(m, ALinear):
                m.weight.data.normal_(0, 2.0 * 1e-1)
                if m.bias is not None:
                    m.bias.data.normal_(0.5, 1e-2)

        # self.linear = ALinear(self.outsize,1)

    def forward(self, image_input, task=0, round_=False):
        """
        Use CNN defined above
        :param image_input:
        :return:
        """
        x = self.features(image_input, task=task, round_=round_)
        x = x.view(x.size()[0], -1)
        x = self.linear(x, dataset=task, round_=round_)
        # x = self.sm(x)
        return x

    def features(self, x, task=0, round_=False):
        x = self.mp1(self.relu(self.bn1[task]((self.conv1(x, dataset=task, round_=round_)))))
        x = self.mp2(self.relu(self.bn2[task](self.conv2(x, dataset=task, round_=round_))))
        x = self.mp3(self.relu(self.bn3[task](self.conv3(x, dataset=task, round_=round_))))
        x = self.mp4(self.relu(self.bn4[task](self.conv4(x, dataset=task, round_=round_))))
        return x

    def forward_single_task(self, x, task=0, round_=False):
        """
        Use MLP defined above
        :param x: input image
        :return:
        """
        # x = x.contiguous()
        # x = x.view(x.size(0), -1)
        # print("Current task:", task)
        x = self.features(x, task=task, round_=round_)
        x = x.view(x.size()[0], -1)
        x = self.linear(x, dataset=task, round_=round_)
        # x = self.sm(x)
        return x

    def get_features(self, x, task=0, round_=False):
        # x = x.contiguous()
        # x = x.view(x.size(0), -1)
        x = self.features(x, task=task, round_=round_)
        return x
