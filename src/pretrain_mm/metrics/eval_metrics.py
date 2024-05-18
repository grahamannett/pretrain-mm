import torchmetrics

from pretrain_mm.constants import IGNORE_INDEX


infolm = torchmetrics.text.infolm.InfoLM(
    "google/bert_uncased_L-2_H-128_A-2",
    idf=False,
    verbose=False,
    information_measure="l2_distance",
)

edit_distance = torchmetrics.text.ExtendedEditDistance()
match_error_rate = torchmetrics.text.MatchErrorRate()

# tensor based
perplexity = torchmetrics.text.Perplexity(ignore_index=IGNORE_INDEX)
