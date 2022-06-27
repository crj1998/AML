from model import resnet, wideresnet

models = {
    "resnet18": resnet.resnet18,
    "resnet50": resnet.resnet50,
    "wideresnet-28-10": wideresnet.wideresnet_28_10,
    "wideresnet-34-16": wideresnet.wideresnet_34_16
}

def build(arch, num_classes):
    assert arch in models, f"Unknown arch: {arch}."
    model = models[arch](num_classes)
    return model