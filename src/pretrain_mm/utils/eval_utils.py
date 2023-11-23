def mse_bbox(target_pos, sequence, tokenizer):
    """
    this is a metric that can be used if i am training with the bbox task.  model should output the sequence in <box>int, int, int, int<box>
    """