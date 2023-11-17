from dataclasses import dataclass

import torch
from simple_parsing import ArgumentParser

from config.fuyu import FuyuInfo
from pretrain_mm.datasets import get_dataset
from pretrain_mm.datasets.dataloader import DataCollator
from pretrain_mm.datasets.task_adapter import TaskAdapterProcessor
from pretrain_mm.processesor.post_processor import fuyu_post_processor


@dataclass
class TrainConfig:
    model_name: str = FuyuInfo.model_name  # "adept/fuyu-8b"
    model_config = FuyuInfo

    auto_wrap_policy: bool = True

    chop_model: int = -1

    # dataset
    dataset_name: str = "silatus_websites"
    task_func: str = "TitleWebsiteTask"

    def get_task_func(self, dataset_info):
        if self.task_func:
            task_func = getattr(dataset_info.tasks, self.task_func)
            return task_func()
        else:
            return dataset_info.task


loss_fct = torch.nn.CrossEntropyLoss()


def dev_load_model(model_name, model_kwargs, ModelCls, train_config):
    if train_config.chop_model > 0:
        model = ModelCls.from_pretrained(model_name, **model_kwargs)
        model.language_model.model.layers = model.language_model.model.layers[: train_config.chop_model]
        model.to("cuda")
    else:
        model = ModelCls.from_pretrained(model_name, device_map="auto", **model_kwargs)
    return model


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
        if ((idx + 1) % 2) == 0:
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

    dataset, dataset_info = get_dataset(train_config.dataset_name)

    task_func = train_config.get_task_func(dataset_info)

    processor = model_config.ProcessorCls.from_pretrained(model_config.model_name, **model_config.tokenizer_kwargs)
    model = dev_load_model(
        model_name=train_config.model_name,
        model_kwargs=model_config.model_kwargs,
        ModelCls=model_config.ModelCls,
        train_config=train_config,
    )

    task_dataset = TaskAdapterProcessor(
        dataset, task_func=task_func, processor=processor, postprocessor=fuyu_post_processor
    )
    collate_fn = DataCollator(processor.pad_token_id, device="cuda")
    dataloader = torch.utils.data.DataLoader(task_dataset, batch_size=1, collate_fn=collate_fn)

    # model = model_config.ModelCls.from_pretrained(model_config.model_name, **model_config.model_kwargs)
    # model.language_model.model.layers = model.language_model.model.layers[:2]
    # model.to("cuda")

    # model = ModelCls.from_pretrained(model_name, torch_dtype=torch.float16)

    # model_kwargs = {**model_config.model_kwargs, "device_map": "auto"}
    # model, processor = setup_model(
    #     train_config.model_name,
    #     model_kwargs=model_kwargs,
    #     tokenizer_kwargs=model_config.tokenizer_kwargs,
    #     ModelCls=model_config.ModelCls,
    #     ProcessorCls=model_config.ProcessorCls,
    # )

    train(model, dataloader)
