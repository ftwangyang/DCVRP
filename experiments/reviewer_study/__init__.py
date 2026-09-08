"""Reproducible reviewer-study implementation.

The public attributes are loaded lazily so pure CPU reviewer runners do not
need to import PyTorch merely to enter this package.
"""

__all__ = [
    "ExperimentalAttentionLearner",
    "ExperimentalEnvironment",
    "build_selector",
    "generate_dataset",
]


def __getattr__(name):
    if name == "generate_dataset":
        from .data_generation import generate_dataset

        return generate_dataset
    if name == "ExperimentalEnvironment":
        from .environment import ExperimentalEnvironment

        return ExperimentalEnvironment
    if name in ("ExperimentalAttentionLearner", "build_selector"):
        from .model import ExperimentalAttentionLearner, build_selector

        return {
            "ExperimentalAttentionLearner": ExperimentalAttentionLearner,
            "build_selector": build_selector,
        }[name]
    raise AttributeError(name)
