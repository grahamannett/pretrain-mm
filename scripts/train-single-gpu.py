from dataclasses import dataclass


import torch
from simple_parsing import ArgumentParser


from configs.fuyu_config import FuyuConfig
from pretrain_mm.datasets.dataloader import DataCollator
from pretrain_mm.datasets import get_dataset, get_dataset_dir
from pretrain_mm.datasets.task_adapter import TaskAdapterProcessor
from pretrain_mm.model.model_utils import setup_model
from pretrain_mm.processesor.post_processor import fuyu_post_processor
from pretrain_mm.datasets import DatasetsAvailable

"""
this script is mostly for testing out if data/model loading/etc works.
actual training in train-fsdp.py
"""

@dataclass
class TrainConfig:
    model_name: str = "adept/fuyu-8b"

    model_config = FuyuConfig
    auto_wrap_policy: bool = True

    # dataset
    dataset_name: str = "silatus_websites"
    dataset_dir: str = get_dataset_dir("SILATUS_DATA_DIR")



loss_fct = torch.nn.CrossEntropyLoss()


def loss_fn(logits, labels):
    b, l, c = logits.shape

    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    # Flatten the tokens

    shift_logits = shift_logits.view(-1, c)
    shift_labels = shift_labels.view(-1)
    # Enable model parallelism
    shift_labels = shift_labels.to(shift_logits.device)
    loss = loss_fct(shift_logits.float(), shift_labels)
    return loss


def train(model, dataloader):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    for idx, batch in enumerate(dataloader):

        input_ids = batch.input_ids
        attention_mask = batch.attention_mask
        image_patches = batch.image_patches
        image_patches_indices = batch.image_patches_indices

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_patches=image_patches,
            image_patches_indices=image_patches_indices,
        )

        loss = loss_fn(outputs.logits, input_ids)
        if ((idx + 1)  % 2) == 0:
            loss.backward()
            optimizer.step()

            optimizer.zero_grad()
        print(f"idx:{idx}")





if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(TrainConfig, dest="train_config")
    args = parser.parse_args()
    train_config: TrainConfig = args.train_config
    model_config = train_config.model_config

    dataset_info = DatasetsAvailable["silatus_websites"]
    dataset = dataset_info.make(**dataset_info.dataset_kwargs)

    model_kwargs = {**model_config.model_kwargs, "device_map": "auto"}

    model, processor = setup_model(
        train_config.model_name,
        model_kwargs=model_kwargs,
        tokenizer_kwargs=model_config.tokenizer_kwargs,
        ModelCls=model_config.ModelCls,
        ProcessorCls=model_config.ProcessorCls,
    )


    task_dataset = TaskAdapterProcessor(dataset, processor=processor, post_processor=fuyu_post_processor)
    collate_fn = DataCollator(processor.pad_token_id)
    dataloader = torch.utils.data.DataLoader(task_dataset, batch_size=1, collate_fn=collate_fn)

    train(model, dataloader)
