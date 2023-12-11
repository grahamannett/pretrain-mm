from pretrain_mm import logger
import wandb
import torch


class CallbackHandler:
    def __init__(self, callbacks: dict):
        self.cb = callbacks

    def __getattr__(self, item):
        return self.cb.get(item, [])


class Trainer(object):
    def __init__(self, config, callbacks: dict = {}):
        self.config = config
        self.callbacks = CallbackHandler(callbacks)

    @property
    def last_lr(self):
        return self.scheduler.get_last_lr()[0]

    def save_model(self, epoch: int = None):
        if self.config.output_dir is None:
            return

        output_path = f"{self.config.output_dir}"
        if self.config.save_every == "epoch":
            output_path += f"/epoch_{epoch}"

        self.model.save_pretrained(output_path)
        logger.info(f"model for epoch: {epoch} saved to: {output_path}")

    def train_step(self, model, batch):
        batch.to(model.device)
        outputs = model(**batch)

        loss = outputs.loss / self.config.grad_accum_steps
        loss.backward()

        if self.config.gradient_clipping is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.gradient_clipping)

        return loss.item()

    def post_train_step(self, log_fn: callable = None):
        if log_fn:
            log_fn()

    def setup_train(self, model=None, optimizer=None, scheduler=None, callbacks=None):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

    def _do_callbacks(self, cbs: list[callable], **kwargs):
        for cb in cbs:
            cb(**kwargs)

    def train(
        self,
        model,
        train_dataloader,
        test_dataloader,
        optimizer,
        scheduler,
        eval_with_generate_kwargs: dict = None,
        post_train_step_log_fn: callable = None,
    ):
        def do_grad_accum_step(batch_idx: int) -> bool:
            # handle this first
            if batch_idx == 0:
                return False  # dont do it for batch 0
            if (
                (batch_idx % self.config.grad_accum_steps == 0)
                or (batch_idx == self.config.num_iters)
                or (batch_idx == len(train_dataloader) - 1)
            ):
                return True
            return False

        for epoch in range(self.config.epochs):
            # setup for train/batch loop
            self.batch_loss, self.epoch_loss = 0, 0
            model.train()

            for batch_idx, batch in enumerate(train_dataloader):
                self.batch_loss += self.train_step(model=model, batch=batch)

                if do_grad_accum_step(batch_idx):
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.epoch_loss += self.batch_loss
                    self.batch_loss = 0

                self._do_callbacks(self.callbacks.train_step, model=model, batch_idx=batch_idx, trainer=self)

                if self.config.num_iters and (self.config.num_iters < batch_idx):
                    break

            self._do_callbacks(self.callbacks.train_epoch, model=model, epoch=epoch, trainer=self)
            self.save_model(epoch=epoch)


class FSDPTrainer(Trainer):
    pass
