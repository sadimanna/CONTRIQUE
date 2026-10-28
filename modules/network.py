import torchvision
import torchvision.models as models

def get_network(name, pretrained=False):
    """
    Returns the specified backbone network.
    Uses modern torchvision 'weights' API when available while maintaining backwards
    compatibility with legacy 'pretrained' argument.
    Instantiates only the requested model lazily.
    """
    factories = {
        "VGG16": (models.vgg16, "VGG16_Weights"),
        "VGG16_bn": (models.vgg16_bn, "VGG16_BN_Weights"),
        "resnet18": (models.resnet18, "ResNet18_Weights"),
        "resnet34": (models.resnet34, "ResNet34_Weights"),
        "resnet50": (models.resnet50, "ResNet50_Weights"),
        "resnet101": (models.resnet101, "ResNet101_Weights"),
        "resnet152": (models.resnet152, "ResNet152_Weights"),
    }
    
    if name not in factories:
        raise KeyError(f"{name} is not a valid network architecture. Supported: {list(factories.keys())}")
    
    model_fn, weight_enum_name = factories[name]
    
    # Modern torchvision (>= 0.13) uses weights enum
    if hasattr(models, weight_enum_name):
        weight_enum = getattr(models, weight_enum_name)
        weights = weight_enum.DEFAULT if pretrained else None
        try:
            return model_fn(weights=weights)
        except TypeError:
            pass
            
    # Legacy torchvision fallback
    return model_fn(pretrained=pretrained)
