from __future__ import annotations

import hashlib
import heapq
import math
from typing import Sequence

import torch
from sklearn import __version__ as sklearn_version
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics.cluster import expected_mutual_information


class CodeDependencyAccumulator:
    """Keep a deterministic uniform frame sample for stream-dependency analysis."""

    def __init__(
        self,
        cardinalities: Sequence[int],
        *,
        sample_limit: int = 50_000,
        seed: int = 2026,
    ) -> None:
        if len(cardinalities) < 2:
            raise ValueError("Dependency analysis requires at least two streams")
        if any(cardinality < 2 for cardinality in cardinalities):
            raise ValueError("Every stream cardinality must be at least two")
        if sample_limit < 2:
            raise ValueError("sample_limit must be at least two")
        self.cardinalities = tuple(int(value) for value in cardinalities)
        self.sample_limit = int(sample_limit)
        self.seed = int(seed)
        self.observed_frames = 0
        self._reservoir: list[tuple[int, str, tuple[int, ...]]] = []

    def _priority(self, identity: str) -> int:
        digest = hashlib.sha256(
            f"dependency-bottom-k-v1:{self.seed}:{identity}".encode("utf-8")
        ).digest()
        return int.from_bytes(digest[:8], "big")

    def update(self, codes: torch.Tensor, cut_id: str) -> None:
        if codes.ndim != 2 or codes.shape[0] != len(self.cardinalities):
            raise ValueError("codes must have shape [S,F] matching cardinalities")
        codes = codes.detach().cpu().long()
        for stream, cardinality in enumerate(self.cardinalities):
            values = codes[stream]
            if torch.any(values < 0) or torch.any(values >= cardinality):
                raise ValueError(f"Stream {stream} has an out-of-range symbol")

        for frame in range(codes.shape[1]):
            identity = f"{cut_id}:{frame}"
            priority = self._priority(identity)
            should_keep = len(self._reservoir) < self.sample_limit or (
                priority < -self._reservoir[0][0]
            )
            if not should_keep:
                continue
            row = tuple(int(value) for value in codes[:, frame].tolist())
            entry = (-priority, identity, row)
            if len(self._reservoir) < self.sample_limit:
                heapq.heappush(self._reservoir, entry)
            else:
                heapq.heapreplace(self._reservoir, entry)
        self.observed_frames += codes.shape[1]

    def _sample(self) -> tuple[list[str], torch.Tensor]:
        ordered = sorted(
            ((-negative, identity, row) for negative, identity, row in self._reservoir),
            key=lambda item: (item[0], item[1]),
        )
        identities = [identity for _, identity, _ in ordered]
        rows = torch.tensor([row for _, _, row in ordered], dtype=torch.long)
        return identities, rows

    @staticmethod
    def _entropy(counts: torch.Tensor) -> tuple[float, int]:
        counts = counts.double()
        nonzero = counts > 0
        probabilities = counts[nonzero] / counts.sum()
        entropy = float(-(probabilities * probabilities.log2()).sum().item())
        return entropy, int(nonzero.sum().item())

    @classmethod
    def _normalized_mutual_information(
        cls,
        left: torch.Tensor,
        right: torch.Tensor,
        left_cardinality: int,
        right_cardinality: int,
    ) -> dict[str, float | int]:
        observations = left.numel()
        joint = torch.bincount(
            left * right_cardinality + right,
            minlength=left_cardinality * right_cardinality,
        ).reshape(left_cardinality, right_cardinality)
        left_counts = joint.sum(dim=1)
        right_counts = joint.sum(dim=0)
        left_entropy, left_used = cls._entropy(left_counts)
        right_entropy, right_used = cls._entropy(right_counts)
        joint_double = joint.double()
        nz_left, nz_right = torch.nonzero(joint_double, as_tuple=True)
        probabilities = joint_double[nz_left, nz_right] / observations
        left_probabilities = left_counts.double() / observations
        right_probabilities = right_counts.double() / observations
        plugin_mi = float(
            (
                probabilities
                * (
                    probabilities
                    / (
                        left_probabilities[nz_left]
                        * right_probabilities[nz_right]
                    )
                ).log2()
            ).sum().item()
        )
        normalizer = math.sqrt(left_entropy * right_entropy)
        plugin_nmi = min(1.0, plugin_mi / normalizer) if normalizer else 0.0
        occupied_joint = joint[left_counts > 0][:, right_counts > 0]
        expected_mi = float(
            expected_mutual_information(occupied_joint.numpy(), observations)
            / math.log(2)
        )
        adjusted_nmi = float(
            adjusted_mutual_info_score(
                left.numpy(), right.numpy(), average_method="geometric"
            )
        )
        if not math.isfinite(adjusted_nmi):
            raise RuntimeError("Adjusted mutual information is not finite")
        return {
            "observations": observations,
            "left_symbols_used": left_used,
            "right_symbols_used": right_used,
            "plugin_mutual_information_bits": plugin_mi,
            "plugin_normalized_mutual_information": plugin_nmi,
            "expected_mutual_information_bits": expected_mi,
            "adjusted_normalized_mutual_information": adjusted_nmi,
            # Keep the generic field consumed by paper assets bound to the
            # chance-adjusted estimator, not the high-cardinality-biased plug-in NMI.
            "normalized_mutual_information": adjusted_nmi,
        }

    def _probe_split(self, identities: Sequence[str]) -> torch.Tensor:
        values = []
        hashes = []
        for identity in identities:
            digest = hashlib.sha256(
                f"dependency-probe-split-v1:{self.seed}:{identity}".encode("utf-8")
            ).digest()
            value = int.from_bytes(digest[:8], "big")
            hashes.append(value)
            values.append(value % 5 == 0)
        split = torch.tensor(values, dtype=torch.bool)
        if split.numel() >= 2 and not torch.any(split):
            split[hashes.index(min(hashes))] = True
        elif split.numel() >= 2 and not torch.any(~split):
            split[hashes.index(max(hashes))] = False
        return split

    def _probe(self, identities: list[str], sample: torch.Tensor) -> list[dict]:
        is_test = self._probe_split(identities)
        if not torch.any(is_test) or not torch.any(~is_test):
            raise RuntimeError("Deterministic probe split produced an empty partition")
        train = sample[~is_test]
        test = sample[is_test]
        results = []
        alpha = 1.0
        for target, target_cardinality in enumerate(self.cardinalities):
            train_target = train[:, target]
            target_counts = torch.bincount(
                train_target, minlength=target_cardinality
            ).double()
            log_prior = (
                (target_counts + alpha)
                / (train.shape[0] + alpha * target_cardinality)
            ).log()
            conditional = []
            for source, source_cardinality in enumerate(self.cardinalities):
                if source == target:
                    continue
                counts = torch.bincount(
                    train_target * source_cardinality + train[:, source],
                    minlength=target_cardinality * source_cardinality,
                ).reshape(target_cardinality, source_cardinality).double()
                log_probabilities = (counts + alpha).log() - (
                    target_counts + alpha * source_cardinality
                ).log()[:, None]
                conditional.append((source, log_probabilities))

            predictions = []
            for start in range(0, test.shape[0], 512):
                chunk = test[start : start + 512]
                scores = log_prior[:, None].expand(-1, chunk.shape[0]).clone()
                for source, log_probabilities in conditional:
                    scores += log_probabilities[:, chunk[:, source]]
                predictions.append(scores.argmax(dim=0))
            predicted = torch.cat(predictions)
            accuracy = float((predicted == test[:, target]).double().mean().item())
            majority = int(target_counts.argmax().item())
            majority_accuracy = float(
                (test[:, target] == majority).double().mean().item()
            )
            results.append(
                {
                    "target_stream": target,
                    "train_frames": int(train.shape[0]),
                    "test_frames": int(test.shape[0]),
                    "laplace_alpha": alpha,
                    "accuracy": accuracy,
                    "majority_accuracy": majority_accuracy,
                    "accuracy_lift": accuracy - majority_accuracy,
                }
            )
        return results

    def as_dict(self) -> dict:
        identities, sample = self._sample()
        if sample.shape[0] < 2:
            raise ValueError("At least two sampled frames are required")
        streams = len(self.cardinalities)
        matrix = [[1.0 if left == right else 0.0 for right in range(streams)] for left in range(streams)]
        pairs = []
        for left in range(streams):
            for right in range(left + 1, streams):
                statistics = self._normalized_mutual_information(
                    sample[:, left],
                    sample[:, right],
                    self.cardinalities[left],
                    self.cardinalities[right],
                )
                nmi = float(statistics["normalized_mutual_information"])
                matrix[left][right] = nmi
                matrix[right][left] = nmi
                pairs.append(
                    {"left_stream": left, "right_stream": right, **statistics}
                )
        digest = hashlib.sha256()
        for identity, row in zip(identities, sample.tolist()):
            digest.update(identity.encode("utf-8"))
            digest.update(b"\0")
            digest.update(",".join(str(value) for value in row).encode("ascii"))
            digest.update(b"\n")
        return {
            "schema_version": 2,
            "estimator": "bottom-k frames; permutation-model adjusted mutual information; geometric-mean normalization",
            "chance_adjustment": "expected mutual information under the fixed-marginal permutation model",
            "implementation": f"scikit-learn {sklearn_version}",
            "probe": "categorical Naive Bayes with Laplace alpha=1 and deterministic 80/20 frame split",
            "seed": self.seed,
            "cardinalities": list(self.cardinalities),
            "observed_frames": self.observed_frames,
            "sample_limit": self.sample_limit,
            "sampled_frames": int(sample.shape[0]),
            "sample_sha256": digest.hexdigest(),
            "normalized_mutual_information_matrix": matrix,
            "pairs": pairs,
            "probe_results": self._probe(identities, sample),
        }
