import torch.nn as nn
from models.da_models import classifier


class PreTrainModel(nn.Module):
    """Pre-training model used for source-only evaluation and TTA initialization."""

    def __init__(self, backbone, configs, hparams):
        super(PreTrainModel, self).__init__()
        self.feature_extractor = backbone(configs)
        self.classifier = classifier(configs)
        self.network = nn.Sequential(self.feature_extractor, self.classifier)
        self.configs = configs
        self.hparams = hparams

    def forward(self, x):
        if isinstance(x, dict):
            x = x["data"]
        feat, _ = self.feature_extractor(x)
        out = self.classifier(feat)
        return out
