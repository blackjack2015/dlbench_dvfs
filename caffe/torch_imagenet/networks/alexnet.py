import torch
import os
import torch.nn as nn
import torch.nn.init as inital
import torch.utils.model_zoo as model_zoo


__all__ = ['AlexNet', 'alexnet']

class LRN(nn.Module):
    def __init__(self, local_size=1, alpha=1.0, beta=0.75, ACROSS_CHANNELS=True):
        super(LRN, self).__init__()
        self.ACROSS_CHANNELS = ACROSS_CHANNELS
        if ACROSS_CHANNELS:
            self.average=nn.AvgPool3d(kernel_size=(local_size, 1, 1),
                    stride=1,
                    padding=(int((local_size-1.0)/2), 0, 0))
        else:
            self.average=nn.AvgPool2d(kernel_size=local_size,
                    stride=1,
                    padding=int((local_size-1.0)/2))
        self.alpha = alpha
        self.beta = beta


    def forward(self, x):
        if self.ACROSS_CHANNELS:
            div = x.pow(2).unsqueeze(1)
            div = self.average(div).squeeze(1)
            div = div.mul(self.alpha).add(1.0).pow(self.beta)
        else:
            div = x.pow(2)
            div = self.average(div)
            div = div.mul(self.alpha).add(1.0).pow(self.beta)
        x = x.div(div)
        return x

class AlexNet111(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
            nn.ReLU(inplace=True),
            LRN(local_size=5, alpha=0.0001, beta=0.75),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2),
            nn.ReLU(inplace=True),
            LRN(local_size=5, alpha=0.0001, beta=0.75),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x
    
class AlexNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0)
        inital.constant_(self.conv1.bias, 0)
        self.relu1 = nn.ReLU(inplace=True)
        self.lrn1 = LRN(local_size=5, alpha=0.0001, beta=0.75)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2)
        inital.constant_(self.conv2.bias, 1)
        self.relu2 = nn.ReLU(inplace=True)
        self.lrn2 = LRN(local_size=5, alpha=0.0001, beta=0.75)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, padding=1)
        inital.constant_(self.conv3.bias, 0)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2)
        inital.constant_(self.conv4.bias, 1)
        self.relu4 = nn.ReLU(inplace=True)
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2)
        inital.constant_(self.conv5.bias, 1)
        self.relu5 = nn.ReLU(inplace=True)
        self.maxpool5 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.linear6 = nn.Linear(256 * 6 * 6, 4096)
        inital.constant_(self.linear6.bias, 1)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout()
        self.linear7 = nn.Linear(4096, 4096)
        inital.constant_(self.linear7.bias, 1)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout()
        self.linear8 = nn.Linear(4096, num_classes)
        inital.constant_(self.linear8.bias, 1)
        
        
        # self.features = nn.Sequential(
        #     nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
        #     nn.ReLU(inplace=True),
        #     LRN(local_size=5, alpha=0.0001, beta=0.75),
        #     nn.MaxPool2d(kernel_size=3, stride=2),
        #     nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2),
        #     nn.ReLU(inplace=True),
        #     LRN(local_size=5, alpha=0.0001, beta=0.75),
        #     nn.MaxPool2d(kernel_size=3, stride=2),
        #     nn.Conv2d(256, 384, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=3, stride=2),
        # )
        # self.classifier = nn.Sequential(
        #     nn.Linear(256 * 6 * 6, 4096),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(),
        #     nn.Linear(4096, 4096),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(),
        #     nn.Linear(4096, num_classes),
        # )

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.lrn1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.lrn2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.conv5(x)
        x = self.relu5(x)
        x = self.maxpool5(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.linear6(x)
        x = self.relu6(x)
        x = self.drop6(x)
        x = self.linear7(x)
        x = self.relu7(x)
        x = self.drop7(x)
        x = self.linear8(x)
        return x

def alexnet(pretrained=False, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = AlexNet(**kwargs)
    if pretrained:
        model_path = 'model_list/alexnet.pth.tar'
        pretrained_model = torch.load(model_path)
        model.load_state_dict(pretrained_model['state_dict'])
    return model
    