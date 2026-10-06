from __future__ import annotations

from pathlib import Path

import hydra
import pytest
from omegaconf import OmegaConf

from dmel_codec.train_codec import (
    CONFIG_DIR,
    STOP_AFTER_VALIDATION_ENV,
    _instantiate_callbacks,
    _require_external_run_dir,
    _scientific_configuration,
    _validate_resume_configuration,
    _validated_scientific_configuration,
    get_config,
    validate_runtime_config,
)
from dmel_codec.utils.callbacks import StopAfterValidationCheckpoint


CODECS = ("ours_075", "ours_l", "ours_m", "ours_h")
EXPERIMENTS = ("smoke", "pilot_clean100", "full_libritts")
ABLATIONS = ("ablation_random_groups", "ablation_joint_fullband")
PILOT_REPAIRS = ("pilot_clean100_repair2_100k",)
REFINEMENTS = (
    "refinement_mel_control",
    "refinement_waveform_frozen",
    "refinement_waveform_joint_tail",
    "refinement_waveform_joint_gan",
    "refinement_waveform_codec_gan",
    "refinement_waveform_vocoder_gan",
    "refinement_waveform_full_codec_gan",
    "refinement_waveform_codec_gan_100k",
    "refinement_waveform_vocoder_full_gan_100k",
    "refinement_waveform_full_codec_full_vocoder_gan_100k",
)
MPLUS_STRATEGIES = (
    (
        "refinement_waveform_codec_gan",
        "waveform_full_codec_frozen_vocoder_gan",
        False,
    ),
    (
        "refinement_waveform_vocoder_gan",
        "waveform_vocoder_tail_only_gan",
        True,
    ),
    (
        "refinement_waveform_full_codec_gan",
        "waveform_joint_full_codec_gan",
        False,
    ),
)
MPLUS_LONG_STRATEGIES = (
    (
        "refinement_waveform_codec_gan_100k",
        "waveform_full_codec_frozen_vocoder_gan",
        False,
    ),
    (
        "refinement_waveform_vocoder_full_gan_100k",
        "waveform_vocoder_full_only_gan",
        True,
    ),
    (
        "refinement_waveform_full_codec_full_vocoder_gan_100k",
        "waveform_joint_full_codec_full_vocoder_gan",
        False,
    ),
)


@pytest.mark.parametrize("codec", CODECS)
@pytest.mark.parametrize("experiment", EXPERIMENTS)
def test_codec_configs_resolve(codec: str, experiment: str) -> None:
    config = get_config([f"codec={codec}", f"experiment={experiment}"])
    resolved = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    assert resolved["codec"]["model"]["_target_"] == (
        "dmel_codec.models.codec_lit_modules.VQGAN"
    )


def test_ours_075_only_changes_fsq_operating_point_from_ours_l() -> None:
    resolved = []
    for codec in ("ours_l", "ours_075"):
        config = get_config(
            [f"codec={codec}", "experiment=pilot_clean100_repair2_100k"]
        )
        value = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
        value["codec"]["name"] = "normalized-rate-point"
        value["codec"]["system_name"] = "Ours-rate-point"
        value["codec"]["model"]["quantizer"]["levels"] = [0]
        value["experiment"]["run_dir"] = "normalized-run-dir"
        value["experiment"]["callbacks"]["checkpoint"]["dirpath"] = (
            "normalized-checkpoint-dir"
        )
        value["experiment"]["logger"]["save_dir"] = "normalized-log-dir"
        resolved.append(value)

    assert resolved[0] == resolved[1]


def test_config_sources_have_no_private_paths_or_missing_values() -> None:
    forbidden = ("???", "/home/", "/data0/", "/sdb/")
    checked = set()
    for path in CONFIG_DIR.rglob("*.yaml"):
        if "config/lm" in path.as_posix():
            continue
        text = path.read_text(encoding="utf-8")
        assert not any(token in text for token in forbidden), path
        checked.add(path.relative_to(CONFIG_DIR).as_posix())
    assert {
        "config.yaml",
        "data/libritts_24k.yaml",
        "codec/base.yaml",
        "codec/ours_075.yaml",
        "codec/ours_l.yaml",
        "codec/ours_m.yaml",
        "codec/ours_h.yaml",
        "experiment/smoke.yaml",
        "experiment/pilot_clean100.yaml",
        "experiment/pilot_clean100_repair2_100k.yaml",
        "experiment/full_libritts.yaml",
        "experiment/ablation_random_groups.yaml",
        "experiment/ablation_joint_fullband.yaml",
        "experiment/refinement_base.yaml",
        "experiment/refinement_mel_control.yaml",
        "experiment/refinement_waveform_frozen.yaml",
        "experiment/refinement_waveform_joint_tail.yaml",
        "experiment/refinement_waveform_joint_gan.yaml",
        "experiment/refinement_waveform_codec_gan.yaml",
        "experiment/refinement_waveform_vocoder_gan.yaml",
        "experiment/refinement_waveform_full_codec_gan.yaml",
        "experiment/refinement_waveform_codec_gan_100k.yaml",
        "experiment/refinement_waveform_vocoder_full_gan_100k.yaml",
        "experiment/refinement_waveform_full_codec_full_vocoder_gan_100k.yaml",
        "evaluation/test_system_requirements.yaml",
    } <= checked


def test_test_gate_requirements_cover_every_preregistered_final_role() -> None:
    requirements = OmegaConf.to_container(
        OmegaConf.load(CONFIG_DIR / "evaluation/test_system_requirements.yaml"),
        resolve=True,
    )
    roles = requirements["roles"]
    assert requirements["schema_version"] == 1
    assert set(roles) == {
        "ground_truth",
        "mel_oracle",
        "dmel_same_frontend",
        "dmel_quantization_aware_decoder",
        "dac_q1",
        "dac_q2",
        "dac_q3",
        "dac_q4",
        "ours_075_seed2026",
        "ours_l_seed2026",
        "base_m_seed2026",
        "flagship_m_seed2026",
        "flagship_m_seed2027",
        "flagship_m_seed2028",
        "ours_h_seed2026",
        "joint_fullband_seed2026",
        "random_groups_seed2026",
    }
    assert [roles[f"dac_q{q}"]["nominal_bitrate_kbps"] for q in range(1, 5)] == [
        0.75,
        1.5,
        2.25,
        3.0,
    ]
    assert roles["ours_075_seed2026"]["nominal_bitrate_kbps"] == pytest.approx(
        0.7429511719
    )
    flagship_systems = roles["flagship_m_seed2026"]["systems"]
    assert "Ours-M" in flagship_systems
    assert all(
        roles[f"flagship_m_seed{seed}"]["systems"] == flagship_systems
        for seed in (2026, 2027, 2028)
    )
    display = requirements["paper_display"]
    assert display["schema_version"] == 1
    assert set(display["main_table_roles"]) == {
        "mel_oracle",
        "dmel_same_frontend",
        "dmel_quantization_aware_decoder",
        "dac_q2",
        "dac_q3",
        "ours_075_seed2026",
        "ours_l_seed2026",
        "base_m_seed2026",
        "flagship_m_seed2026",
        "flagship_m_seed2027",
        "flagship_m_seed2028",
    }
    assert set(display["rate_quality_roles"]) == {
        "dac_q1",
        "dac_q2",
        "dac_q3",
        "dac_q4",
        "ours_075_seed2026",
        "ours_l_seed2026",
        "flagship_m_seed2026",
        "flagship_m_seed2027",
        "flagship_m_seed2028",
        "ours_h_seed2026",
    }
    assert set(display["ablation_table_roles"]) == {
        "base_m_seed2026",
        "joint_fullband_seed2026",
        "random_groups_seed2026",
    }
    assert "base_m_seed2026" not in display["rate_quality_roles"]
    assert "ours_h_seed2026" not in display["main_table_roles"]


@pytest.mark.parametrize("experiment", ABLATIONS)
def test_ablation_configs_resolve(experiment: str) -> None:
    config = get_config([f"experiment={experiment}", "codec=ours_m"])
    OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    assert config.experiment.name == experiment
    assert config.codec.system_name.startswith("Ours-M")


def test_smoke_model_instantiates_on_cpu() -> None:
    config = get_config(["codec=ours_m", "experiment=smoke"])
    model = hydra.utils.instantiate(config.codec.model, _convert_="partial")
    assert model.vocoder is None
    assert next(model.parameters()).device.type == "cpu"


def test_training_config_fails_fast_without_bigvgan(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("BIGVGAN_CKPT", raising=False)
    monkeypatch.delenv("BIGVGAN_CONFIG", raising=False)
    config = get_config(["experiment=full_libritts"])
    with pytest.raises(FileNotFoundError, match="BIGVGAN_CKPT"):
        validate_runtime_config(config)


def test_init_and_resume_are_mutually_exclusive(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.ckpt"
    checkpoint.touch()
    config = get_config(
        [
            "experiment=smoke",
            f"init_from={checkpoint}",
            f"resume_from={checkpoint}",
        ]
    )
    with pytest.raises(ValueError, match="mutually exclusive"):
        validate_runtime_config(config)


def test_runtime_validation_stop_is_explicit_and_not_scientific(
    monkeypatch,
) -> None:
    config = get_config(["experiment=smoke"])
    scientific = _scientific_configuration(config)
    monkeypatch.setenv(STOP_AFTER_VALIDATION_ENV, "2000")

    validate_runtime_config(config)
    callbacks = _instantiate_callbacks(config)

    assert isinstance(callbacks[-1], StopAfterValidationCheckpoint)
    assert callbacks[-1].step == 2000
    assert _scientific_configuration(config) == scientific

    monkeypatch.setenv(STOP_AFTER_VALIDATION_ENV, "invalid")
    with pytest.raises(ValueError, match="positive integer"):
        validate_runtime_config(config)


def test_resume_requires_the_original_scientific_configuration(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.ckpt"
    checkpoint.touch()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    original = get_config(["experiment=smoke", f"experiment.run_dir={run_dir}"])
    OmegaConf.save(original, run_dir / "resolved_config.yaml")

    resumed = get_config(
        [
            "experiment=smoke",
            f"experiment.run_dir={run_dir}",
            f"resume_from={checkpoint}",
        ]
    )
    _validate_resume_configuration(resumed)
    assert _scientific_configuration(resumed) == _scientific_configuration(original)

    resumed.seed += 1
    with pytest.raises(ValueError, match="differs from the original run"):
        _validate_resume_configuration(resumed)


def test_initialized_run_resumes_without_reapplying_initial_weights(
    tmp_path: Path,
) -> None:
    initialization = tmp_path / "initialization.ckpt"
    resume_checkpoint = tmp_path / "last.ckpt"
    initialization.touch()
    resume_checkpoint.touch()
    run_dir = tmp_path / "initialized-run"
    run_dir.mkdir()
    original = get_config(
        [
            "experiment=smoke",
            f"experiment.run_dir={run_dir}",
            f"+experiment.init_from={initialization}",
            f"init_from={initialization}",
        ]
    )
    OmegaConf.save(original, run_dir / "resolved_config.yaml")

    resumed = get_config(
        [
            "experiment=smoke",
            f"experiment.run_dir={run_dir}",
            "+experiment.init_from=null",
            "init_from=null",
            f"resume_from={resume_checkpoint}",
        ]
    )

    validate_runtime_config(resumed)
    _validate_resume_configuration(resumed)
    assert resumed.init_from is None
    assert resumed.experiment.init_from is None
    assert _validated_scientific_configuration(resumed) == (
        _scientific_configuration(original)
    )


def test_resume_requires_existing_run_metadata(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.ckpt"
    checkpoint.touch()
    config = get_config(
        [
            "experiment=smoke",
            f"experiment.run_dir={tmp_path / 'missing-run'}",
            f"resume_from={checkpoint}",
        ]
    )
    with pytest.raises(FileNotFoundError, match="resolved_config.yaml"):
        _validate_resume_configuration(config)


def test_formal_run_dir_must_be_absolute_and_outside_git(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="absolute"):
        _require_external_run_dir("relative/run")
    with pytest.raises(ValueError, match="outside"):
        _require_external_run_dir(str(CONFIG_DIR.parent / "runs"))
    assert _require_external_run_dir(str(tmp_path / "run")) == tmp_path / "run"


@pytest.mark.parametrize(
    "experiment",
    (
        "pilot_clean100",
        *PILOT_REPAIRS,
        "full_libritts",
        *ABLATIONS,
        *REFINEMENTS,
    ),
)
def test_formal_checkpoint_runs_after_validation(experiment: str) -> None:
    config = get_config([f"experiment={experiment}"])
    checkpoint = hydra.utils.instantiate(config.experiment.callbacks.checkpoint)
    assert checkpoint.monitor == "val_loss"
    assert checkpoint._every_n_train_steps == 0
    assert checkpoint._every_n_epochs == 1
    assert checkpoint._save_on_train_epoch_end is False
    assert checkpoint.auto_insert_metric_name is False


@pytest.mark.parametrize(
    ("experiment", "mode"),
    (
        ("refinement_mel_control", "mel_control"),
        ("refinement_waveform_frozen", "waveform_frozen_vocoder"),
        ("refinement_waveform_joint_tail", "waveform_joint_vocoder_tail"),
        (
            "refinement_waveform_joint_gan",
            "waveform_joint_vocoder_tail_gan",
        ),
    ),
)
def test_refinement_budget_and_frozen_bitstream(experiment: str, mode: str) -> None:
    config = get_config([f"experiment={experiment}", "codec=ours_m"])
    assert config.codec.model._target_.endswith(".CodecRefinement")
    assert config.codec.model.refinement_mode == mode
    assert config.codec.model.freeze_encoder is True
    assert config.codec.model.discriminator is None
    assert config.codec.model.weight_adv == 0.0
    gan_mode = mode == "waveform_joint_vocoder_tail_gan"
    expected_steps = 3_000 if gan_mode else 5_000
    assert config.experiment.trainer.max_steps == expected_steps
    assert config.experiment.trainer.val_check_interval == (500 if gan_mode else 1_000)
    assert config.data.module.train_max_durations == (3.0 if gan_mode else 6.0)
    assert config.data.module.val_cuts_path.endswith("dev-clean-hash64_eval.jsonl.gz")
    assert config.codec.model.optimizer.lr == 1e-5
    assert (
        config.codec.model.lr_scheduler.lr_lambda.num_training_steps == expected_steps
    )


@pytest.mark.parametrize(("experiment", "mode", "freeze_encoder"), MPLUS_STRATEGIES)
def test_mplus_strategy_configs_differ_only_in_trainable_boundary(
    experiment: str, mode: str, freeze_encoder: bool
) -> None:
    config = get_config([f"experiment={experiment}", "codec=ours_m"])
    model = config.codec.model

    assert model.refinement_mode == mode
    assert model.freeze_encoder is freeze_encoder
    assert model.waveform_discriminator is not None
    assert model.discriminator is None
    assert model.weight_adv == 0.0
    assert config.experiment.trainer.max_steps == 1_500
    assert config.experiment.trainer.val_check_interval == 250
    assert config.data.module.train_max_durations == 3.0
    assert model.waveform_loss_weight == 0.2
    assert model.validation_waveform_loss_weight == 0.2
    assert model.optimizer.lr == 2e-6
    assert model.encoder_lr_scale == 0.5
    assert model.vocoder_lr_scale == 0.1
    assert model.discriminator_lr_scale == 10.0
    assert model.gan_adversarial_weight == 0.1
    assert model.gan_feature_weight == 20.0
    assert model.lr_scheduler.lr_lambda.num_training_steps == 1_500


@pytest.mark.parametrize(
    ("experiment", "mode", "freeze_encoder"), MPLUS_LONG_STRATEGIES
)
def test_mplus_long_strategy_budget_and_boundary(
    experiment: str, mode: str, freeze_encoder: bool
) -> None:
    config = get_config([f"experiment={experiment}", "codec=ours_m"])
    model = config.codec.model

    assert model.refinement_mode == mode
    assert model.freeze_encoder is freeze_encoder
    assert model.waveform_discriminator is not None
    assert config.experiment.trainer.max_steps == 100_000
    assert config.experiment.trainer.val_check_interval == 2_000
    assert config.data.module.train_max_durations == 3.0
    assert config.data.module.train_crop_seed == 2026
    assert model.waveform_loss_weight == 0.2
    assert model.validation_waveform_loss_weight == 0.2
    assert model.optimizer.lr == 2e-6
    assert model.encoder_lr_scale == 0.5
    expected_vocoder_scale = 0.1 if experiment.endswith("codec_gan_100k") else 0.5
    assert model.vocoder_lr_scale == expected_vocoder_scale
    assert model.discriminator_lr_scale == 10.0
    assert model.gan_adversarial_weight == 0.1
    assert model.gan_feature_weight == 20.0
    assert model.lr_scheduler.lr_lambda.num_warmup_steps == 1_000
    assert model.lr_scheduler.lr_lambda.num_training_steps == 100_000


def test_joint_decoder_vocoder_training_requires_waveform_gan() -> None:
    config = get_config(["experiment=refinement_waveform_joint_tail", "codec=ours_m"])
    with pytest.raises(ValueError, match="requires a waveform GAN"):
        validate_runtime_config(config)

    gan_config = get_config(
        ["experiment=refinement_waveform_joint_gan", "codec=ours_m"]
    )
    gan_config.experiment.require_vocoder = False
    gan_config.experiment.trainer.accelerator = "cpu"
    validate_runtime_config(gan_config)


@pytest.mark.parametrize("experiment", ("full_libritts", *ABLATIONS))
def test_full_budget_matches_measured_pilot_dynamic_batch(experiment: str) -> None:
    config = get_config([f"experiment={experiment}"])
    assert config.experiment.trainer.max_steps == 100_000
    assert config.data.module.train_max_durations == 60.0
    assert config.data.module.val_max_durations == 30.0
    assert config.codec.model.optimizer.lr == 0.0001
    assert config.codec.model.lr_scheduler.lr_lambda.num_training_steps == 100_000
    assert config.codec.model.lr_scheduler.lr_lambda.final_lr_ratio == 0.1


@pytest.mark.parametrize("experiment", ("pilot_clean100_repair2_100k", *ABLATIONS))
def test_clean100_scale_matches_data_and_codec_gan(experiment: str) -> None:
    config = get_config([f"experiment={experiment}", "codec=ours_m"])
    assert config.data.module.train_cuts_path.endswith("train-clean-100_codec.jsonl.gz")
    assert config.experiment.trainer.max_steps == 100_000
    assert config.codec.model.weight_adv == 0.2
    assert config.codec.model.discriminator is not None


def test_repair2_changes_only_the_preregistered_training_budget() -> None:
    repair1 = get_config(
        [
            "experiment=pilot_clean100",
            "codec=ours_m",
            "codec.model.optimizer.lr=0.0001",
        ]
    )
    repair2 = get_config(["experiment=pilot_clean100_repair2_100k", "codec=ours_m"])

    assert repair2.experiment.trainer.max_steps == 100_000
    assert repair2.codec.model.lr_scheduler.lr_lambda.num_training_steps == 100_000
    assert repair2.codec.model.optimizer.lr == 0.0001
    assert repair2.init_from is None
    assert repair2.resume_from is None

    repair1.experiment.name = repair2.experiment.name
    repair1.experiment.run_dir = repair2.experiment.run_dir
    repair1.experiment.trainer.max_steps = 100_000
    repair1.codec.model.lr_scheduler.lr_lambda.num_training_steps = 100_000
    assert OmegaConf.to_container(repair1, resolve=True) == OmegaConf.to_container(
        repair2, resolve=True
    )


def test_joint_ablation_matches_parameters_rate_and_stream_count() -> None:
    grouped = get_config(["experiment=ablation_random_groups", "codec=ours_m"])
    joint = get_config(["experiment=ablation_joint_fullband", "codec=ours_m"])

    def codec_parameters(config) -> int:
        modules = [
            hydra.utils.instantiate(config.codec.model.encoder),
            hydra.utils.instantiate(config.codec.model.quantizer),
            hydra.utils.instantiate(config.codec.model.decoder),
        ]
        projection = config.codec.model.get("quantized_projection")
        if projection:
            modules.append(hydra.utils.instantiate(projection))
        pre_projection = config.codec.model.get("pre_quantized_projection")
        if pre_projection:
            modules.append(hydra.utils.instantiate(pre_projection))
        return sum(
            parameter.numel() for module in modules for parameter in module.parameters()
        )

    grouped_parameters = codec_parameters(grouped)
    joint_parameters = codec_parameters(joint)
    grouped_quantizer = hydra.utils.instantiate(grouped.codec.model.quantizer)
    joint_quantizer = hydra.utils.instantiate(joint.codec.model.quantizer)

    assert abs(grouped_parameters - joint_parameters) / grouped_parameters < 0.002
    assert grouped_quantizer.num_streams == joint_quantizer.num_streams == 10
    assert grouped_quantizer.codebook_cardinality == 175
    assert joint_quantizer.codebook_cardinality == 175
    assert grouped.codec.token_frame_rate == joint.codec.token_frame_rate == 23.4375
