from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from dmel_codec.evaluation.cache import sha256_file
from dmel_codec.tools.make_listening_bundle import ROUTES, SPLITS


TRAINING_ROUTES = ROUTES[1:]


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to replace listening receipt: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _item_map(
    document: dict[str, Any], *, label: str
) -> dict[tuple[str, str], dict[str, Any]]:
    items = document.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError(f"{label} has no items")
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for item in items:
        if not isinstance(item, dict):
            raise ValueError(f"{label} item must be an object")
        split = item.get("split")
        cut_id = item.get("cut_id")
        if split not in SPLITS or not isinstance(cut_id, str):
            raise ValueError(f"Invalid {label} item identity")
        identity = (split, cut_id)
        if identity in result:
            raise ValueError(f"Duplicate {label} item: {identity}")
        result[identity] = item
    return result


def _condition_routes(item: dict[str, Any]) -> dict[str, str]:
    conditions = item.get("conditions")
    if not isinstance(conditions, list) or len(conditions) != len(ROUTES):
        raise ValueError("Listening key item must contain four conditions")
    mapping: dict[str, str] = {}
    for condition in conditions:
        if not isinstance(condition, dict):
            raise ValueError("Listening key condition must be an object")
        blind_label = condition.get("blind_label")
        route = condition.get("route")
        if blind_label not in ("A", "B", "C", "D") or route not in ROUTES:
            raise ValueError("Invalid listening key condition")
        if blind_label in mapping:
            raise ValueError(f"Duplicate blind label in listening key: {blind_label}")
        mapping[blind_label] = route
    if set(mapping.values()) != set(ROUTES):
        raise ValueError("Listening key item must cover every route exactly once")
    return mapping


def complete_mplus_listening(
    *,
    direction_gate: Path,
    listening_manifest: Path,
    listening_key: Path,
    ratings: Path,
    output: Path,
) -> dict[str, Any]:
    direction_gate = direction_gate.expanduser().resolve()
    listening_manifest = listening_manifest.expanduser().resolve()
    listening_key = listening_key.expanduser().resolve()
    ratings = ratings.expanduser().resolve()
    output = output.expanduser().resolve()

    gate = _load_json(direction_gate)
    manifest = _load_json(listening_manifest)
    key = _load_json(listening_key)
    rating_document = _load_json(ratings)
    if gate.get("schema_version") != 1 or gate.get("status") != (
        "MPLUS_DIRECTION_GATE"
    ):
        raise ValueError("Invalid M+ direction gate")
    if gate.get("automatic_promotion") is not False:
        raise ValueError("Direction gate must prohibit automatic promotion")
    if manifest.get("schema_version") != 1 or manifest.get("status") != (
        "MPLUS_MATCHED_LISTENING_BUNDLE"
    ):
        raise ValueError("Invalid matched-listening manifest")
    if key.get("schema_version") != 1 or key.get("status") != (
        "MPLUS_MATCHED_LISTENING_KEY"
    ):
        raise ValueError("Invalid matched-listening key")
    if (
        rating_document.get("schema_version") != 1
        or rating_document.get("status") != "MPLUS_MATCHED_LISTENING_RATINGS"
    ):
        raise ValueError("Invalid matched-listening ratings")

    manifest_hash = sha256_file(listening_manifest)
    if rating_document.get("listening_manifest_sha256") != manifest_hash:
        raise ValueError("Ratings do not bind the listening manifest")
    if key.get("randomization_seed") != manifest.get("randomization_seed"):
        raise ValueError("Listening key and manifest randomization differ")

    manifest_items = _item_map(manifest, label="listening manifest")
    key_items = _item_map(key, label="listening key")
    rating_items = _item_map(rating_document, label="ratings")
    if set(manifest_items) != set(key_items) or set(manifest_items) != set(
        rating_items
    ):
        raise ValueError("Manifest, key and ratings cut sets differ")

    route_scores: dict[str, list[int]] = {route: [] for route in ROUTES}
    for identity in sorted(manifest_items):
        manifest_conditions = manifest_items[identity].get("conditions")
        if not isinstance(manifest_conditions, list):
            raise ValueError("Listening manifest item has no conditions")
        manifest_labels = {
            condition.get("blind_label")
            for condition in manifest_conditions
            if isinstance(condition, dict)
        }
        if manifest_labels != {"A", "B", "C", "D"}:
            raise ValueError("Listening manifest item has invalid blind labels")
        label_to_route = _condition_routes(key_items[identity])
        scores = rating_items[identity].get("scores")
        if not isinstance(scores, dict) or set(scores) != set(label_to_route):
            raise ValueError(f"Ratings must score A-D exactly once: {identity}")
        for blind_label, route in label_to_route.items():
            score = scores[blind_label]
            if (
                isinstance(score, bool)
                or not isinstance(score, int)
                or not 1 <= score <= 5
            ):
                raise ValueError(f"Listening scores must be integers 1-5: {identity}")
            route_scores[route].append(score)

    score_count = len(manifest_items)
    if any(len(scores) != score_count for scores in route_scores.values()):
        raise ValueError("Every route must receive one score per listening cut")
    means = {route: sum(scores) / len(scores) for route, scores in route_scores.items()}

    eligible = gate.get("eligible_before_listening")
    if not isinstance(eligible, list) or any(
        route not in TRAINING_ROUTES for route in eligible
    ):
        raise ValueError("Direction gate has an invalid eligibility list")
    expected_disposition = "listening_required" if eligible else "retain_source"
    if gate.get("disposition") != expected_disposition:
        raise ValueError("Direction gate disposition disagrees with eligibility")
    selected = "source"
    selection_reason = "no eligible direction exceeded source"
    if eligible:
        best_value = max(means[route] for route in eligible)
        best_routes = [route for route in eligible if means[route] == best_value]
        if len(best_routes) == 1 and best_value > means["source"]:
            selected = best_routes[0]
            selection_reason = "unique highest eligible mean exceeded source"
        elif len(best_routes) > 1 and best_value > means["source"]:
            selection_reason = "top eligible listening means tied; retained source"

    receipt = {
        "schema_version": 1,
        "status": "MPLUS_MATCHED_LISTENING_COMPLETE",
        "direction_gate_sha256": sha256_file(direction_gate),
        "selected_direction": selected,
        "evaluated_directions": list(TRAINING_ROUTES),
        "selection_rule": (
            "Select the unique gate-eligible direction with highest mean 1-5 score "
            "only when it strictly exceeds source; otherwise retain source."
        ),
        "selection_reason": selection_reason,
        "num_cuts": score_count,
        "route_scores": {
            route: {"mean": means[route], "values": route_scores[route]}
            for route in ROUTES
        },
        "listening_manifest": {
            "path": str(listening_manifest),
            "sha256": manifest_hash,
        },
        "listening_key": {
            "path": str(listening_key),
            "sha256": sha256_file(listening_key),
        },
        "ratings": {"path": str(ratings), "sha256": sha256_file(ratings)},
    }
    _atomic_json(output, receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate M+ matched-listening ratings and freeze its receipt"
    )
    parser.add_argument("--direction-gate", type=Path, required=True)
    parser.add_argument("--listening-manifest", type=Path, required=True)
    parser.add_argument("--listening-key", type=Path, required=True)
    parser.add_argument("--ratings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = complete_mplus_listening(
        direction_gate=args.direction_gate,
        listening_manifest=args.listening_manifest,
        listening_key=args.listening_key,
        ratings=args.ratings,
        output=args.output,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
