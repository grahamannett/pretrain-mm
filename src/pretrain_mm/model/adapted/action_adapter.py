import torch
from transformers import AutoModel


class ActionAdapterModel(torch.nn.Module):
    def __init__(self, model_name, adapter_dim):
        super(ActionAdapterModel, self).__init__()

        # Load the base model
        self.base_model = AutoModel.from_pretrained(model_name)

        # Add an adapter layer on top of the base model output
        self.adapter = torch.nn.Linear(self.base_model.config.hidden_size, adapter_dim)

    def forward(self, input_ids, attention_mask):
        # Pass the input through the base model
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)

        # Get the last hidden state from the base model output
        last_hidden_state = outputs.last_hidden_state

        # Pass the last hidden state through the adapter layer
        adapter_output = self.adapter(last_hidden_state)

        return adapter_output
