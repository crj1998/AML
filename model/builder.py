from model import resnet, preactresnet, wideresnet, convformer

models = {
    "resnet-18": resnet.resnet18,
    "resnet-50": resnet.resnet50,
    "preactresnet-18": preactresnet.preactresnet18,
    "preactresnet-50": preactresnet.preactresnet50,
    "wideresnet-28-10": wideresnet.wideresnet_28_10,
    "wideresnet-34-16": wideresnet.wideresnet_34_16,
    "convformer-tiny": convformer.convformerTiny
}

def build(arch, num_classes):
    assert arch in models, f"Unknown arch: {arch}."
    model = models[arch](num_classes)
    return model