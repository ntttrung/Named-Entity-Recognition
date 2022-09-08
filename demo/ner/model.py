from abc import ABC
from typing import List, Optional

import torch
import datasets
from pytorch_lightning import LightningModule
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
)




class NERModel(LightningModule, ABC):
    def __init__(
        self,
        model_name_or_path: str,
        num_labels: int,
        tags_list: List[str],
    ):
        super().__init__()

        self.tags_list = tags_list

        self.config = AutoConfig.from_pretrained(model_name_or_path, num_labels=num_labels)
        self.model = AutoModelForTokenClassification.from_config(self.config)
        self.metrics = datasets.load_metric('seqeval')

    def forward(self, **inputs):
        return self.model(**inputs)

