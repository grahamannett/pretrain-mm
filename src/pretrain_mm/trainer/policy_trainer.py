class LoraDPOTrainer(Trainer):
    def setup_train(
        self,
        lora_config: lora_utils.BaseLoraConfig,
        model=None,
        optimizer=None,
        scheduler=None,
        callbacks=None,
        **kwargs,
    ):
        super().setup_train(model, optimizer, scheduler, callbacks)
        self.model, self.lora_adapter = lora_utils.setup_lora(model, lora_config=lora_config)

    def _get_batch_logps(self, all_logits: torch.Tensor, all_labels: torch.Tensor, ignore_index: int = -100):
        bs = all_logits.shape[0]

        logits = all_logits[:, :-1]
        labels = all_labels[:, 1:]

        # labels[labels == -100] = 0
        mask = labels != ignore_index

        # calculate and then mask
        per_token_logps = torch.gather(
            logits.log_softmax(-1),
            dim=-1,
            index=labels.unsqueeze(2),
        )

        per_token_logps = per_token_logps.squeeze(-1)
        per_token_logps = per_token_logps[mask]
        return per_token_logps

    def _dpo_loss(self, policy_logps: torch.Tensor, ref_logps: torch.Tensor):
        logits = policy_logps - ref_logps
        loss = -torch.nn.functional.logsigmoid(logits)

    def train_step(self, model: torch.nn.Module, batch: Batch):
        batch.to(model.device)

        # compute
        model.enable_adapters()
        policy_outputs = model(**batch)

        policy_logps = self._get_batch_logps(all_logits=policy_outputs.logits, all_labels=batch.labels)

        with torch.no_grad():
            ref_outputs = model(**batch)
            ref_logps = self._get_batch_logps(all_logits=ref_outputs.logits, all_labels=batch.labels)

        dpo_loss = self._dpo_loss(policy_logps=policy_logps, ref_logps=ref_logps)

        loss.backward()

        if self.gradient_clipping is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.gradient_clipping)

        return loss.item()

    def compare_policy(self, output1, output2, *args, **kwargs):
        output1_embeds = get_embeddings(
            model,
            output1,
        )

    def _generate_step(self, model):
        pass

    def _example_approval_method(self, model, batch):
        """
        thinking through how i would do the approval/rejection for policy generation
        """

        generated_output1 = self.generate_output(model, batch)
        generated_output2 = self.generate_output(model, batch)

        # compare outputs via embedding or something
        approved, rejected = self.compare_policy(generated_output1, generated_output2)


class DDPTrainer(Trainer):
    def __init__(self, config, callbacks: dict = {}):
        super().__init__(config, callbacks)
        self.rank = int(os.environ.get("LOCAL_RANK", None))
        self.world_size = int(os.environ.get("WORLD_SIZE", None))

    def setup_train(self, model=None, optimizer=None, scheduler=None, callbacks=None):
        super().setup_train(model, optimizer, scheduler, callbacks)

        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[self.rank], find_unused_parameters=False)

        # TODO turn on gradient checkpointing
        model._set_static_graph()
