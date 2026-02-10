from __future__ import annotations

import pytest
from pcod_common import manifest


def _minimal_payload() -> dict:
    return {
        'schema_version': manifest.SCHEMA_VERSION,
        'model_name': 'pbod',
        'precision': 'fp16',
        'device': 'cuda',
        'preprocessing': {},
        'postprocessing': {},
        'model': {},
    }


def test_validate_manifest_accepts_minimal_payload():
    manifest.validate_manifest(_minimal_payload())


def test_validate_manifest_rejects_wrong_schema_version():
    payload = _minimal_payload()
    payload['schema_version'] = '0.0'
    with pytest.raises(ValueError):
        manifest.validate_manifest(payload)


@pytest.mark.parametrize(
    ('value', 'expected'),
    [
        (None, []),
        (0.25, [0.25]),
        ([0.1, 0.2], [0.1, 0.2]),
    ],
)
def test_score_threshold_list(value, expected):
    assert manifest.score_threshold_list(value) == expected
