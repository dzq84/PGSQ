from __future__ import annotations

import importlib.resources


def test_namespace_subpackages_are_importable() -> None:
    import dmel_codec.dataset
    import dmel_codec.evaluation
    import dmel_codec.models.modules.bigvgan
    import dmel_codec.tools

    assert dmel_codec.dataset is not None
    assert dmel_codec.evaluation is not None
    assert dmel_codec.models.modules.bigvgan is not None
    assert dmel_codec.tools is not None


def test_hydra_config_is_packaged() -> None:
    root = importlib.resources.files("dmel_codec")
    assert root.joinpath("config/config.yaml").is_file()
    assert root.joinpath("config/codec/ours_m.yaml").is_file()
