from dataclasses import asdict, dataclass, field
from typing import Literal

import torchmetrics

from pretrain_mm.constants import IGNORE_INDEX


# tried using subclasses w/ __init__subclass__ and factory for creating the MetricArgs and
# default_factory type but seems like the cli kwargs do not attach to the instance in that case
class MetricConfig:
    def setup(self, *args, **kwargs):
        return self._from, {**asdict(self) | kwargs}


MetricAnnotation = dict[str, MetricConfig]


class MetricHelper:
    @staticmethod
    def make_collection(metrics, prefix=""):
        _metrics = [metric_func.setup() for metric_func in metrics.values()]
        _metrics = [metric_cls(**metric_vals) for metric_cls, metric_vals in _metrics]

        return torchmetrics.MetricCollection(_metrics, prefix=prefix)


@dataclass
class CharErrorRate(MetricConfig):
    _from = torchmetrics.text.CharErrorRate


@dataclass
class ExtendedEditDistance(MetricConfig):
    _from = torchmetrics.text.ExtendedEditDistance


@dataclass
class MatchErrorRate(MetricConfig):
    _from = torchmetrics.text.MatchErrorRate


InfoLMInformationMeasure = Literal[
    "kl_divergence",
    "l2_distance",
    "alpha_divergence",
    "beta_divergence",
    "ab_divergence",
    "renyi_divergence",
    "l1_distance",
    "l_infinity_distance",
    "fisher_rao_distance",
]


@dataclass
class InfoLMConfig(MetricConfig):
    _from = torchmetrics.text.infolm.InfoLM
    # model name default is "google/bert_uncased_L-2_H-128_A-2"/"bert-base-uncased"
    model_name_or_path: str = "google/bert_uncased_L-2_H-128_A-2"
    idf: bool = False
    verbose: bool = False
    information_measure: InfoLMInformationMeasure = "kl_divergence"
    temperature: float = 0.25


@dataclass
class Perplexity(MetricConfig):
    _from = torchmetrics.text.Perplexity
    ignore_index = IGNORE_INDEX


MetricArgs = field(
    default_factory={
        "infolm": InfoLMConfig(),
        "charerror": CharErrorRate(),
        "matcherror": MatchErrorRate(),
        "extendededit": ExtendedEditDistance(),
    }.copy
)

IntMetricArgs = field(
    default_factory={
        "perplexity": Perplexity(),
    }.copy
)
