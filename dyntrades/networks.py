from .modules.vgg import VGG
from .modules.resnet import ResNet18


def vgg13(num_classes):
    return VGG('VGG13', num_classes)


def vgg16(num_classes):
    return VGG('VGG16', num_classes)


def resnet18(num_classes):
    return ResNet18(num_classes)
