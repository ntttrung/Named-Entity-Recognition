import time
from argparse import ArgumentParser

import torch
import mlflow.pytorch
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from model import NERModel
from dataset import NERDataModule


import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':

    seed_everything(43)

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

    parser = ArgumentParser()
    parser.add_argument('--run_name', type=str, required=True)

    parser.add_argument('--model_name_or_path', type=str, default='xlm-roberta-base')
    parser.add_argument('--dataset_version', type=str, required=True)
    parser.add_argument('--label_all_tokens', type=bool, default=False)
    parser.add_argument('--max_seq_length', type=int, default=128)
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--eval_batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=15)

    # model specific arguments
    parser.add_argument('--use_crf', type=bool, default=False)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--adam_epsilon', type=float, default=1e-8)
    parser.add_argument('--warmup_steps', type=int, default=0)
    parser.add_argument('--weight_decay', type=float, default=0.0)

    args = parser.parse_args()
    run_name = args.run_name
    run_name += '_{}'.format(time.time())
    # mlflow.pytorch.autolog()
    mlf_logger = MLFlowLogger(experiment_name="fpt_ner_logs",
                              tracking_uri="file:./mlruns",
                              run_name=run_name)

    dm = NERDataModule(model_name_or_path=args.model_name_or_path,
                       dataset_version=args.dataset_version,
                       tags_list=tags_list,
                       label_all_tokens=args.label_all_tokens,
                       max_seq_length=args.max_seq_length,
                       train_batch_size=args.train_batch_size,
                       eval_batch_size=args.eval_batch_size)
    dm.setup(stage="fit")

    checkpoint_callback = ModelCheckpoint(
        monitor='val_overall_f1',
        dirpath='checkpoints/' + run_name,
        filename='{epoch:02d}--{val_overall_f1:.2f}',
        save_top_k=2,
        mode="max",
        save_weights_only=True)

    early_stop_callback = EarlyStopping(monitor='val_overall_f1', patience=5, verbose=True, mode='max')

    model = NERModel(model_name_or_path=dm.model_name_or_path,
                     num_labels=dm.num_labels,
                     tags_list=dm.tags_list,
                     train_batch_size=dm.train_batch_size,
                     eval_batch_size=dm.eval_batch_size,
                     use_crf=args.use_crf,
                     learning_rate=args.learning_rate,
                     adam_epsilon=args.adam_epsilon,
                     warmup_steps=args.warmup_steps,
                     weight_decay=args.weight_decay)

    AVAIL_GPUS = min(1, torch.cuda.device_count())

    trainer = Trainer(
        max_epochs=args.num_epochs,
        gpus=AVAIL_GPUS,
        logger=mlf_logger,
        callbacks=[checkpoint_callback, early_stop_callback])

    trainer.fit(
        model,
        datamodule=dm)

    mlf_logger.experiment.log_artifact(mlf_logger.run_id, 'checkpoints/' + run_name)

    trainer.test(model, datamodule=dm)