import glob
import os
import torch
from matplotlib import pyplot as plt
from dmel_codec.utils.logger import RankedLogger
from typing import Optional

log = RankedLogger(__name__, rank_zero_only=True)


def find_lastest_ckpt(directory):
    if directory is None:
        return None
    ckpt_file = glob.glob(os.path.join(directory, "*.ckpt"))

    if not ckpt_file:
        log.info(f"No ckpt files found in this directory: {directory}")
        return None

    latest_ckpt_file = max(ckpt_file, key=os.path.getmtime)
    return latest_ckpt_file


def plot_mel(data, titles=None):
    fig, axes = plt.subplots(len(data), 1, squeeze=False)

    if titles is None:
        titles = [None for i in range(len(data))]

    plt.tight_layout()

    for i in range(len(data)):
        mel = data[i]

        if isinstance(mel, torch.Tensor):
            mel = mel.float().detach().cpu().numpy()

        axes[i][0].imshow(mel, origin="lower")
        axes[i][0].set_aspect(2.5, adjustable="box")
        axes[i][0].set_ylim(0, mel.shape[0])
        axes[i][0].set_title(titles[i], fontsize="medium")
        axes[i][0].tick_params(labelsize="x-small", left=False, labelleft=False)
        axes[i][0].set_anchor("W")

    return fig


def sequence_mask(length: torch.Tensor, max_length: int | None = None):
    # length: (batch_size, ) or (1, batch_size)
    if length.ndim == 2:
        length = length.squeeze(0)
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


def avg_with_mask(x, mask):
    assert mask.dtype == torch.float, "Mask should be float"

    if mask.ndim == 2:
        mask = mask.unsqueeze(1)

    if mask.shape[1] == 1:
        mask = mask.expand_as(x)

    return (x * mask).sum() / mask.sum()


def open_filelist(filelist_path, file_num=None):
    audio_path_list = []
    with open(filelist_path, "r") as f:
        filelist = f.readlines()
        if file_num is None:
            file_num = len(filelist)
        for file in filelist[:file_num]:
            audio_path = file.strip()
            audio_path_list.append(audio_path)
    return audio_path_list


def sample_one_token_from_logits(
    logits: Optional[torch.Tensor] = None,
    previous_token: Optional[
        torch.Tensor
    ] = None,  # for audio logits, text_previous_token is None
    temperature=0.7,
    top_k=50,
    top_p=0.7,
    repetition_penalty=1.2,
):
    probs = logits_to_probs(
        logits,
        previous_tokens=previous_token,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
    )
    idx_next = multinomial_sample_one_no_sync(probs)
    return idx_next, probs


def logits_to_probs(
    logits: torch.Tensor,
    previous_tokens: Optional[torch.Tensor] = None,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
) -> torch.Tensor:
    # Apply repetition penalty
    if previous_tokens is not None:
        previous_tokens = previous_tokens.long()
        score = torch.gather(logits, dim=-1, index=previous_tokens)
        score = torch.where(
            score < 0, score * repetition_penalty, score / repetition_penalty
        )
        logits.scatter_(dim=-1, index=previous_tokens, src=score)

    # Step 1: Apply top-k filtering
    if top_k > 0:
        top_k_logits, _ = torch.topk(logits, top_k, dim=-1)
        min_top_k_logits = top_k_logits[..., -1, None]
        logits = torch.where(
            logits < min_top_k_logits,
            torch.full_like(logits, -float("Inf")),
            logits,
        )

    # Step 2: Apply top-p filtering
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cum_probs = torch.cumsum(
            torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1
        )
        sorted_indices_to_remove = cum_probs > top_p
        sorted_indices_to_remove[..., 0] = False  # keep at least one option
        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove
        )
        logits = logits.masked_fill(indices_to_remove, -float("Inf"))

    logits = logits / max(temperature, 1e-5)
    probs = torch.nn.functional.softmax(logits, dim=-1)

    # 确保传入的是一个概率分布
    assert torch.all(probs >= 0), "Probabilities must be non-negative"

    # # 确保概率分布的和为1
    # assert torch.isclose(
    #     probs.sum(), torch.tensor(1.0, device=probs.device, dtype=probs.dtype)
    # ), "Probabilities must sum to 1"

    return probs


def multinomial_sample_one_no_sync(probs_sort):
    # 使用 torch.multinomial 进行采样
    idx_next = torch.multinomial(probs_sort, 1)
    return idx_next.to(dtype=torch.long)