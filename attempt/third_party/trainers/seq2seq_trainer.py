from packaging import version
import torch
from torch import nn
from typing import Any, Dict, List, Optional, Tuple, Union

from torch.utils.data.dataset import Dataset
from transformers import Seq2SeqTrainer
from .trainer import BaseTrainer
from transformers.file_utils import is_datasets_available

# my import
from torch.utils.data import DataLoader
import datasets

if version.parse(torch.__version__) >= version.parse("1.6"):
    from torch.cuda.amp import autocast


class Seq2SeqTrainer(Seq2SeqTrainer, BaseTrainer):
    def __init__(self, train_dataset_sizes=None, shared=False, multiple_metrics=None, adapter_config=None, shuffle=False, save_checkpoint=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.adapter_config = adapter_config
        self.multiple_metrics = multiple_metrics
        self.train_dataset_sizes = train_dataset_sizes
        self.shared = shared
        self.shuffle = shuffle
        self.save_checkpoint = save_checkpoint

    def get_train_dataloader(self):
        if self.shuffle:
            return super().get_train_dataloader()

        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        if isinstance(train_dataset, torch.utils.data.dataset.IterableDataset):
            return super().get_train_dataloader()

        return DataLoader(
            train_dataset,
            batch_size=self.args.train_batch_size,
            shuffle=self.shuffle,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def _save_checkpoint(self, model, trial, metrics=None):
        if not self.save_checkpoint:
            print("======================================")
            print("Skipping save_checkpoint")
            print("======================================")
        else:
            super()._save_checkpoint(model, trial, metrics)

    def evaluate(
        self,
        eval_dataset: Optional[Dict[str, Dataset]] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        max_length: Optional[int] = None,
        num_beams: Optional[int] = None,
    ) -> Dict[str, float]:
        self._max_length = max_length
        self._num_beams = num_beams,
        print("======================================")
        print("Evaluation ")
        print("======================================")
        metrics = super().evaluate(eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)
        print(metrics)
        self.log_metrics("eval", metrics)
        breakpoint()
        return metrics

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on :obj:`model` using obj:`inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (:obj:`nn.Module`):
                The model to evaluate.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (:obj:`bool`):
                Whether or not to return the loss only.

        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
            labels (each being optional).
        """
        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)
        gen_kwargs = {
            "max_length": self._max_length if self._max_length is not None else self.model.config.max_length,
            "num_beams": self._num_beams if self._num_beams is not None else self.model.config.num_beams,
            "task": inputs["task"] if "task" in inputs else "all"
        }

        generated_tokens = self.model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            **gen_kwargs,
        )

        # in case the batch is shorter than max length, the output should be padded
        if generated_tokens.shape[-1] < gen_kwargs["max_length"]:
            generated_tokens = self._pad_tensors_to_max_len(
                generated_tokens, gen_kwargs["max_length"])

        with torch.no_grad():
            if self.use_amp:
                with autocast():
                    outputs = model(**inputs)
            else:
                outputs = model(**inputs)
            if has_labels:
                if self.label_smoother is not None:
                    loss = self.label_smoother(
                        outputs, inputs["labels"]).mean().detach()
                else:
                    loss = (outputs["loss"] if isinstance(
                        outputs, dict) else outputs[0]).mean().detach()
            else:
                loss = None

        if self.args.prediction_loss_only:
            return (loss, None, None)

        labels = inputs["labels"]
        if labels.shape[-1] < gen_kwargs["max_length"]:
            labels = self._pad_tensors_to_max_len(
                labels, gen_kwargs["max_length"])

        return (loss, generated_tokens, labels)
