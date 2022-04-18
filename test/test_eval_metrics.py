"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License").
You may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import pytest
from massive.utils.training_utils import eval_preds

cases = [
# ---------------- Slot F1 ---------------
    (
        None,
        None,
        [['X', 'Y']],
        [['X', 'Y']],
        'Other',
        None,
        'slot_micro_f1',
        {'slot_micro_f1': 1.0}
    ),
    (
        # Test slot combination
        None,
        None,
        [['X', 'X', 'Z']],
        [['X', 'X', 'Y']],
        'Other',
        None,
        'slot_micro_f1',
        {'slot_micro_f1': 0.5}
    ),
    (
        # Test padding
        None,
        None,
        [['X', 'X']],
        [['X', 'X', 'Y']],
        'Other',
        None,
        'slot_micro_f1',
        {'slot_micro_f1': 0.5}
    ),
    (
        # Test truncation
        None,
        None,
        [['X', 'X', 'Z', 'Z']],
        [['X', 'X', 'Y']],
        'Other',
        None,
        'slot_micro_f1',
        {'slot_micro_f1': 0.5}
    ),
    (
        # Test recurring slots
        None,
        None,
        [['X', 'X', 'Other', 'X', 'X', 'Y', 'Z']],
        [['X', 'X', 'Other', 'X', 'X', 'Z', 'Y']],
        'Other',
        None,
        'slot_micro_f1',
        {'slot_micro_f1': 0.5}
    ),
    (
        # Test -100 merging
        None,
        None,
        [[50, -100, 50, -100, -100, 20, 20, 10, 10, -100, 20]],
        [[50, -100, 50, -100, -100, 20, 0,  20, 0,  -100, 20]],
        [0],
        [-100],
        'slot_micro_f1',
        {'slot_micro_f1': 0.75}
    ),

# ------------- Exact match acc  and intent acc ----------
    (
        ['A', 'B', 'C', 'E'],
        ['A', 'B', 'D', 'E'],
        [['X', 'Y'], ['X', 'X', 'Y'], ['X'], ['X', 'Y']],
        [['X', 'X'], ['X', 'X', 'Y'], ['X'], ['X', 'Z']],
        'Other',
        None,
        'all',
        {'ex_match_acc': 0.25, 'intent_acc': 0.75}
    ),
    (
        # Now with numbers
        [1, 2, 3, 5],
        [1, 2, 4, 5],
        [[10, 11], [10, 10, 11], [10], [10, 11]],
        [[10, 10], [10, 10, 11], [10], [10, 12]],
        'Other',
        None,
        'all',
        {'ex_match_acc': 0.25, 'intent_acc': 0.75}
    ),
]

@pytest.mark.parametrize(
    'pred_intents, lab_intents, pred_slots, lab_slots, labels_ignore, labels_merge, eval_metrics, '
    'out',
    cases
)
def test_eval_preds(
    pred_intents, lab_intents, pred_slots, lab_slots, labels_ignore, labels_merge, eval_metrics, out
):

    results = eval_preds(
        pred_intents=pred_intents,
        lab_intents=lab_intents,
        pred_slots=pred_slots,
        lab_slots=lab_slots,
        labels_ignore=labels_ignore,
        labels_merge=labels_merge,
        eval_metrics=eval_metrics
    )

    for key in out:
        assert key in results
        assert round(out[key], 2) == round(results[key], 2)
