import logging
from models.classifier import Classifier
from models.img_resnet import ResNet50, DISIFLF
from models.genertor import Discriminator, DouGen

def build_model(config, num_identities, num_clothes, OnlyDSIFLF = False):
    logger = logging.getLogger('reid.model')

    # Build backbone
    logger.info("Initializing model: Resnet50")
    if not OnlyDSIFLF:
        model = ResNet50(config, num_clothes)
    else:
        model = DISIFLF(config)
    discriminator = Discriminator(feature_dim=config.MODEL.FEATURE_DIM)
    genertor = DouGen()
    logger.info("Model  size: {:.5f}M".format(sum(p.numel() for p in model.parameters())/1000000.0))
    # Build classifier
    identity_classifier = Classifier(feature_dim=config.MODEL.FEATURE_DIM, num_classes=num_identities)
    return model, identity_classifier, discriminator, genertor