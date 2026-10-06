# PGSQ: Parallel Grouped-FSQ Spectral Codec

Reference implementation and audio demo for

> **Low-Rate Speech Coding with Shared Mel-Band Encoding and Parallel Quantization**
> Zheqi Dai, Zhiyue Wu, Haolin He, Chunyat Wu, Qiuqiang Kong
> IEEE ICASSP 2027 (submitted)

- 🔊 Audio demo: https://dzq84.github.io/PGSQ/
- ⚙️ Code: this repository

## Overview

PGSQ is a low-rate speech codec. It splits a 100-bin log-mel spectrogram into
10 adjacent bands and compresses every band in time with a **shared
WaveNet-style encoder**. Each band has its own affine projection followed by
**single-stage finite scalar quantization (FSQ)**, so every band/frame produces
one integer code and bands are quantized in parallel. A **full-band WaveNet
decoder** reconstructs the mel spectrogram, and **BigVGAN-v2** synthesizes the
24 kHz waveform.

Base models are trained on LibriTTS `train-clean-100` with a frozen BigVGAN.
`Ours-M-FT` starts from `Ours-M` and jointly fine-tunes the codec and vocoder
with additional data and updates. Reported rates are nominal and exclude
metadata/container overhead.

| Config | FSQ levels | Codes/band | Nominal rate |
| --- | --- | ---: | ---: |
| Ours-VL | `(3,3)` | 9 | 0.743 kb/s |
| Ours-L | `(5,5)` | 25 | 1.088 kb/s |
| Ours-M / Ours-M-FT | `(7,5,5)` | 175 | 1.746 kb/s |
| Ours-H | `(8,5,5,5)` | 1000 | 2.336 kb/s |

## Results

LibriTTS reconstruction. `M-FT` is the three-seed mean; other codec rows are
single runs. Lower WER is better; higher ViSQOL/PESQ/SpkSim is better.

| System | Rate (kb/s) | test-clean WER | ViSQOL | PESQ | SpkSim | test-other WER | ViSQOL | PESQ | SpkSim |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DAC q=2 | 1.500 | 6.47 | 3.649 | 1.530 | 0.460 | 20.17 | 3.524 | 1.504 | 0.454 |
| DAC q=3 | 2.250 | 3.77 | 3.965 | 1.981 | 0.600 | 11.22 | 3.876 | 1.917 | 0.593 |
| Ours-VL | 0.743 | 10.46 | 3.755 | 1.535 | 0.558 | 32.94 | 3.566 | 1.437 | 0.497 |
| Ours-L | 1.088 | 4.99 | 3.928 | 1.733 | 0.641 | 16.66 | 3.776 | 1.607 | 0.595 |
| Ours-M | 1.746 | 2.88 | 4.166 | 1.975 | 0.748 | 8.76 | 4.055 | 1.800 | 0.708 |
| **Ours-M-FT** | **1.746** | **2.68** | **4.251** | **2.174** | **0.751** | **8.18** | **4.137** | **1.958** | **0.725** |
| Ours-H | 2.336 | 2.82 | 4.241 | 2.103 | 0.779 | 7.21 | 4.114 | 1.872 | 0.744 |

`Ours-M-FT` improves WER, ViSQOL, PESQ and speaker similarity over three-codebook
DAC on both splits at a 22.4% lower nominal bitrate.

## Repository layout

```
dmel_codec/   Python package (models, training, evaluation, configuration)
scripts/      Training and evaluation entry points
tests/        CPU test suite (GPU tests are marked)
docs/         Static audio demo served by GitHub Pages
```

The codec is implemented in `dmel_codec/models/codec_lit_modules.py`,
`dmel_codec/models/modules/wavenet.py` and
`dmel_codec/models/modules/downsample_fsq.py`.

## Installation

Verified with Python 3.10, PyTorch/Torchaudio 2.6.0 and CUDA 12.4.

```bash
conda create -n pgsq python=3.10 pip -y
conda activate pgsq
python -m pip install torch==2.6.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124
python -m pip install -e ".[train,dev]"
python -m pip install -e ".[eval,dac]"   # evaluation metrics and DAC baseline
```

## Data and pretrained assets

Large paths stay outside the checkout:

```bash
export LIBRITTS_ROOT=/datasets/LibriTTS
export DMEL_RUN_ROOT=/absolute/writable/path/runs
export DMEL_CACHE_ROOT=/absolute/writable/path/cache
export BIGVGAN_CKPT=/absolute/path/bigvgan_generator.pt
export BIGVGAN_CONFIG=/absolute/path/config.json
```

`BIGVGAN_CKPT` / `BIGVGAN_CONFIG` must be the NVIDIA BigVGAN-v2 24 kHz,
100-band, hop-256 checkpoint. DAC and the Apple dMel baseline use their own
public checkpoints; these third-party assets are not redistributed here.

```bash
python -m dmel_codec.tools.doctor --data-root "$LIBRITTS_ROOT"
pytest -q -m "not gpu"
```

## Training

```bash
# inspect a fully resolved configuration
python -m dmel_codec.train_codec codec=ours_m experiment=smoke --cfg job --resolve

# train one base model, one process per seed/GPU
export CUDA_VISIBLE_DEVICES=0 NPROC=1
scripts/train.sh pilot_clean100_repair2_100k ours_m 2026
```

`init_from` loads weights only; `resume_from` restores full Trainer, optimizer,
scheduler and global-step state. Use `scripts/train.sh` for `ours_m`, `ours_l`,
`ours_h` and the ablations; the formal low-rate point uses its own release-bound
wrapper, `scripts/train_ours_075.sh`.

## Evaluation

```bash
export DMEL_METRICS=stoi,estoi,pesq_wb,visqol_speech,speaker_cosine,si_sdr,mr_stft,ms_log_mel,f0_correlation,wer
scripts/evaluate.sh identity dev-clean identity-dev-clean

export DAC_CKPT=/absolute/path/weights_24khz.pth
export DAC_QUANTIZERS=2
scripts/evaluate.sh dac dev-clean dac-q2-dev-clean
```

For Ours, `scripts/evaluate.sh` exports `CUBLAS_WORKSPACE_CONFIG=:4096:8` before
Python starts so encoding is deterministic across batch compositions.

## Audio demo

`docs/` is a static page comparing 8 LibriTTS test-clean utterances across the
reference waveform, the mel oracle, all Ours operating points, DAC at four
rates, the NVIDIA spectral codec and adapted dMel.

- Hosted: https://dzq84.github.io/PGSQ/
- Local: `cd docs && python -m http.server 8000`, then open
  `http://localhost:8000/`.

## Citation

```bibtex
@inproceedings{dai2027pgsq,
  title     = {Low-Rate Speech Coding with Shared Mel-Band Encoding and Parallel Quantization},
  author    = {Dai, Zheqi and Wu, Zhiyue and He, Haolin and Wu, Chunyat and Kong, Qiuqiang},
  booktitle = {IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year      = {2027},
  note      = {Submitted}
}
```

## Acknowledgements

This work uses or builds on
[Apple dMel](https://github.com/apple/dmel),
[NVIDIA BigVGAN](https://github.com/NVIDIA/BigVGAN),
[DAC](https://github.com/descriptinc/descript-audio-codec),
[ViSQOL](https://github.com/google/visqol) and
[LibriTTS](https://www.openslr.org/60/).

## License

MIT, see [LICENSE](LICENSE).
