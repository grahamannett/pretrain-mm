import torch
import torch.nn.functional as F


class DPOLoss(torch.nn.Module):
    """original DPO Loss"""

    def __init__(self, beta, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.beta = beta

    def forward(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: torch.Tensor,
        reference_rejected_logps: torch.Tensor,
    ) -> torch.Tensor:
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps

        logits = pi_logratios - ref_logratios # (policy_chosen_logps - policy_rejected_logps) - (reference_chosen_logps - reference_rejected_logps)
        # = (p_chosen + ref_rej) - (p_rej + ref_chosen)
        # = log(pc*rr) - log(pr*rc)
        # = log((pc*rr)/(pr*rc))
        losses = -F.logsigmoid(self.beta * logits)
        loss = losses.mean()
        chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps).detach()

        return loss, chosen_rewards, rejected_rewards


class IDPOLoss(torch.nn.Module):
    """ """

    def __init__(self, beta, *args, **kwargs):
        super().__init__()
        self.beta = beta
        self.distance_reg

    def forward(self, chosen_logps: torch.Tensor, rejected_logps: torch.Tensor, distance_from: int = 1):
        logits = (chosen_logps - rejected_logps) / distance_from
        losses = (-F.logsigmoid(self.beta * logits)).mean()
        chosen_rewards = self.beta * chosen_logps.detach()
        rejected_rewards = self.beta * rejected_logps.detach()

        return losses, chosen_rewards, rejected_rewards

class IDPOTrainer:
    """
    indirect policy optimization

    should be similar to but as I dont have enough GPU for reference model need to ???
    also

    https://github.com/OpenLLMAI/OpenRLHF/blob/main/openrlhf/trainer/dpo_trainer.py
    """

    def __init__(self, model, num_generations: int = 2):
        self.model = model
        self.num_generations = num_generations

    # def create_generations()

    def _get_batch_logps(
        self, logits: torch.FloatTensor, labels: torch.LongTensor, average_log_prob: bool = False
    ) -> torch.FloatTensor:
        """Compute the log probabilities of the given labels under the given logits.

        Args:
            logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
            labels: Labels for which to compute the log probabilities. Label tokens with a value of -100 are ignored. Shape: (batch_size, sequence_length)
            average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.

        Returns:
            A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
        """
        assert logits.shape[:-1] == labels.shape

        labels = labels[:, 1:].clone()
        logits = logits[:, :-1, :]
        loss_mask = labels != self.tokenizer.pad_token_id

        # dummy token; we'll ignore the losses on these tokens later
        labels[labels == self.tokenizer.pad_token_id] = 0
        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

        if average_log_prob:
            return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
        else:
            return (per_token_logps * loss_mask).sum(-1)

    def run(self):

        # for sample in self.dataset:
        loss_func = IDPOLoss()
        for n_iter in range(self.num_iterations):
            sample = self.dataset.sample()
            generations = self.create_generations(self.model, sample, self.num_generations)
            approved, rejected = self.compare_generations(generations, target=sample)

            approved_output = self.model(**approved)
            rejected_output = self.model(**rejected)
            # presumably labels is the input_ids shifted 1:
            approved_logps = self._get_batch_logps(logits=approved_output.logits, labels=approved.labels)
            rejected_logps = self._get_batch_logps(logits=rejected_output.logits, labels=rejected.labels)

            loss =
