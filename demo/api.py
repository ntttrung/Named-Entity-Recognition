from pydantic import BaseModel
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

import torch
import numpy as np
from transformers import AutoTokenizer

from ner.model import NERModel
from utils.preprocess import preprocess_text
from utils.postprocess import concat_tag

import warnings
warnings.filterwarnings("ignore")
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'


api = FastAPI(title="NER", version='0.1.0')
origins = ["*"]
api.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    max_age=600,
)


class TextInput(BaseModel):
    text: str


tags_list = ["B-ORG", "I-ORG",
                 "B-DEGREE", "I-DEGREE",
                 "B-MAJOR", "I-MAJOR",
                 "B-TIME", "I-TIME",
                 "B-POSITION", "I-POSITION",
                 "B-SCORE", "I-SCORE",
                 "B-SOFT_SKILL", "I-SOFT_SKILL",
                 "B-TECH_SKILL", "I-TECH_SKILL",
                 "B-LOC", "I-LOC",
                 "B-NAME", 'I-NAME',
                 "B-PHONE", "I-PHONE",
                 "O",
                 "B-EMAIL", "I-EMAIL"]

# load model
# checkpoint = torch.load(args.checkpoint_path)

weight_path = '/media/Z/TrungNT108/NER/checkpoints/all_label_1662635325.6424944/epoch=26--val_overall_f1=0.83.ckpt'
model = NERModel(
    model_name_or_path='xlm-roberta-base',
    num_labels=len(tags_list),
    tags_list=tags_list
).load_from_checkpoint(weight_path)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base', use_fast=True)


@api.post('/api/predict')
def predict(text_input: TextInput):
    text = text_input.text
    normed_text = preprocess_text(text)

    words_list = normed_text.split(' ')

    # tokenize text
    tokenized_text = tokenizer.encode_plus(
        normed_text,
        add_special_tokens=True,
        max_length=256,
        padding=False,
        truncation=True,
        return_attention_mask=True,
        return_tensors='np',
    )

    encoding = tokenized_text.data
    if device.type == 'cpu':
        encoding['input_ids'] = torch.LongTensor(encoding['input_ids'], device=device)
        encoding['attention_mask'] = torch.LongTensor(encoding['attention_mask'], device=device)
    elif device.type == 'cuda':
        encoding['input_ids'] = torch.cuda.LongTensor(encoding['input_ids'], device=device)
        encoding['attention_mask'] = torch.cuda.LongTensor(encoding['attention_mask'], device=device)

    word_ids = tokenized_text.word_ids()
    results = model(**encoding)['logits']
    logit = results.detach().cpu().numpy()
    prediction = np.argmax(logit, axis=-1).squeeze()
    tag_prediction = []

    pre_word_index = None
    for i in range(len(prediction)):
        origin_index = word_ids[i]
        id_tag = prediction[i]
        tag = tags_list[id_tag]

        if origin_index not in [pre_word_index, None]:
            tag_prediction.append((words_list[origin_index], tag))
            pre_word_index = origin_index

    words_list, entities_list = concat_tag(iob_format=tag_prediction)

    return {'words': words_list, 'tags': entities_list}

