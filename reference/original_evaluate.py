"""Evaluate dynamic-memory predictions under an explicit JSONL contract.

This is an original evaluation scaffold, NOT the official DynaBench evaluator,
a DynaMem implementation, or a model. The included data are synthetic unit-test
fixtures only. Use Python 3.10+; no third-party dependencies are required.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding='utf-8') as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f'{path}:{line_number}: invalid JSON') from exc
            if not isinstance(record, dict):
                raise ValueError(f'{path}:{line_number}: expected a JSON object')
            records.append(record)
    if not records:
        raise ValueError(f'{path}: no records')
    return records


def finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'{name} must be a finite number')
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f'{name} must be finite')
    return number


def xyz(value: Any, name: str) -> tuple[float, float, float] | None:
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f'{name} must be null or a three-number list')
    return tuple(finite_number(v, name) for v in value)


def index_by_id(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    indexed = {}
    for row in rows:
        identifier = row.get('query_id')
        if not isinstance(identifier, str) or not identifier:
            raise ValueError(f'{label}: query_id must be a non-empty string')
        if identifier in indexed:
            raise ValueError(f'{label}: duplicate query_id {identifier}')
        indexed[identifier] = row
    return indexed


def summarize(outcomes: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(outcomes)
    pos = [r for r in outcomes if r['positive']]
    neg = [r for r in outcomes if not r['positive']]
    localized = [r for r in outcomes if r['returned_location']]
    def rate(num: int, den: int) -> float | None:
        return num / den if den else None
    positive_success = sum(r['correct'] for r in pos)
    negative_success = sum(r['correct'] for r in neg)
    a, b = rate(positive_success, len(pos)), rate(negative_success, len(neg))
    return {
        'queries': n,
        'positive_queries': len(pos),
        'negative_queries': len(neg),
        'overall_success': rate(sum(r['correct'] for r in outcomes), n),
        'positive_localization_success': a,
        'negative_not_found_accuracy': b,
        'negative_false_location_rate': rate(sum(r['returned_location'] for r in neg), len(neg)),
        'balanced_success': (a + b) / 2 if a is not None and b is not None else None,
        'location_coverage': rate(len(localized), n),
        'returned_location_precision': rate(sum(r['correct'] for r in localized), len(localized)),
    }


def evaluate(queries: list[dict[str, Any]], predictions: list[dict[str, Any]],
             threshold: float | None = None) -> dict[str, Any]:
    if not queries:
        raise ValueError('At least one query is required')
    if threshold is not None:
        threshold = finite_number(threshold, 'threshold')
        if not 0 <= threshold <= 1:
            raise ValueError('threshold must lie in [0, 1]')
    truth = index_by_id(queries, 'queries')
    pred = index_by_id(predictions, 'predictions')
    if truth.keys() != pred.keys():
        raise ValueError(f'IDs differ; missing={sorted(truth.keys()-pred.keys())}; '
                         f'extra={sorted(pred.keys()-truth.keys())}')
    outcomes = []
    for key, q in truth.items():
        pr = pred[key]
        scene = q.get('scene_id')
        if not isinstance(scene, str) or not scene:
            raise ValueError(f'{key}: scene_id must be a non-empty string')
        if pr.get('scene_id') != scene:
            raise ValueError(f'{key}: prediction scene_id mismatch')
        query_time = finite_number(q.get('query_time'), f'{key}.query_time')
        latest = pr.get('latest_observation_time')
        if latest is not None:
            latest = finite_number(latest, f'{key}.latest_observation_time')
            if latest >= query_time:
                raise ValueError(f'{key}: observations must be strictly before query_time')
        target = xyz(q.get('target_xyz'), f'{key}.target_xyz')
        predicted = xyz(pr.get('predicted_xyz'), f'{key}.predicted_xyz')
        if 'target_xyz' not in q or 'predicted_xyz' not in pr:
            raise ValueError(f'{key}: target_xyz and predicted_xyz are required (null allowed)')
        radius = None
        if target is not None:
            radius = finite_number(q.get('radius_m'), f'{key}.radius_m')
            if radius < 0:
                raise ValueError(f'{key}: radius_m cannot be negative')
        if predicted is not None:
            if latest is None:
                raise ValueError(f'{key}: a location requires an observation timestamp')
            evidence = finite_number(pr.get('candidate_observation_time'), f'{key}.candidate_observation_time')
            if evidence > latest or evidence >= query_time:
                raise ValueError(f'{key}: inconsistent candidate timestamp')
        if threshold is not None and predicted is not None:
            confidence = finite_number(pr.get('candidate_confidence'), f'{key}.candidate_confidence')
            if not 0 <= confidence <= 1:
                raise ValueError(f'{key}: confidence must lie in [0, 1]')
            if confidence < threshold:
                predicted = None
        if target is None:
            correct = predicted is None
        else:
            correct = predicted is not None and math.dist(target, predicted) <= radius
        outcomes.append({'query_id': key, 'scene_id': scene, 'positive': target is not None,
                         'returned_location': predicted is not None, 'correct': correct})
    scenes = sorted({r['scene_id'] for r in outcomes})
    per_scene = {s: summarize([r for r in outcomes if r['scene_id'] == s]) for s in scenes}
    return {
        'evaluation_type': 'custom_scaffold_not_official_dynabench',
        'threshold': threshold,
        'micro': summarize(outcomes),
        'macro_scene_success': sum(s['overall_success'] for s in per_scene.values()) / len(scenes),
        'per_scene': per_scene,
        'per_query': outcomes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queries', type=Path, required=True)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--threshold', type=float, default=None,
                        help='Optional fixed threshold; select only on development data.')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    try:
        results = evaluate(load_jsonl(args.queries), load_jsonl(args.predictions), args.threshold)
        output = json.dumps(results, indent=2, allow_nan=False)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(output + '\n', encoding='utf-8')
        print(output)
    except (ValueError, OSError) as exc:
        parser.exit(2, f'error: {exc}\n')

if __name__ == '__main__':
    main()
