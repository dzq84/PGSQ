"""Compatibility guard for the removed ambiguous legacy codec factory."""

from __future__ import annotations


class InitialCodec:
    def __init__(self, *args, **kwargs) -> None:
        raise RuntimeError(
            "InitialCodec was removed because codec_name='dMel' incorrectly loaded "
            "the learned grouped-FSQ model. Use OursAdapter, "
            "DmelSameFrontendAdapter, or DACAdapter from "
            "dmel_codec.evaluation.adapters."
        )
