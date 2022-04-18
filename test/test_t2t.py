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
from massive.utils.training_utils import (
    convert_input_to_t2t,
    convert_intent_slots_to_t2t,
    convert_t2t_batch_to_intents_slots
)

# format: utt, intent, slots, use_output_descrip, intent_first, slots_mixed, toks_in_output, 
#         sentinels, inside_format, output
output_tests= [
    (
        ['are', 'storms', 'likely', 'today'],
        'weather_query',
        ['Other', 'weather_descriptor', 'Other', 'date'],
        False,
        False,
        False,
        False,
        False,
        'slot_name',
        'Other weather_descriptor Other date weather_query'
    ),
    (
        ['are', 'storms', 'likely', 'july', 'fourth'],
        'weather_query',
        ['Other', 'weather_descriptor', 'Other', 'date', 'date'],
        False,
        False,
        False,
        False,
        False,
        'inside_slot_name',
        'Other weather_descriptor Other date I-date weather_query'
    ),
    (
        ['are', 'storms', 'likely', 'july', 'fourth'],
        'weather_query',
        ['Other', 'weather_descriptor', 'Other', 'date', 'date'],
        False,
        False,
        False,
        False,
        False,
        'inside',
        'Other weather_descriptor Other date I weather_query'
    ),
    (
        ['are', 'there', 'storms', 'likely', 'july', 'fourth'],
        'weather_query',
        ['Other', 'Other', 'weather_descriptor', 'Other', 'date', 'date'],
        False,
        False,
        False,
        False,
        False,
        'inside',
        'Other Other weather_descriptor Other date I weather_query'
    ),
    (
        ['are', 'storms', 'likely', 'today'],
        'weather_query',
        ['Other', 'weather_descriptor', 'Other', 'date'],
        False,
        False,
        False,
        False,
        True,
        'slot_name',
        '<extra_id_0> Other <extra_id_1> weather_descriptor <extra_id_2> Other <extra_id_3> date weather_query'
    ),
    (
        ['are', 'storms', 'likely', 'july', 'fourth'],
        'weather_query',
        ['Other', 'weather_descriptor', 'Other', 'date', 'date'],
        False,
        False,
        False,
        False,
        True,
        'inside',
        '<extra_id_0> Other <extra_id_1> weather_descriptor <extra_id_2> Other <extra_id_3> date <extra_id_4> I weather_query'
    ),
    (
        ['are', 'storms', 'likely', 'today'],
        'weather_query',
        ['Other', 'weather_descriptor', 'Other', 'date'],
        False,
        False,
        False,
        True,
        False,
        'slot_name',
        'are storms likely today Other weather_descriptor Other date weather_query'
    ),
    (
        ['are', 'storms', 'likely', 'today'],
        'weather_query',
        ['Other', 'weather_descriptor', 'Other', 'date'],
        False,
        False,
        True,
        True,
        False,
        'slot_name',
        'are Other storms weather_descriptor likely Other today date weather_query'
    ),
    (
        ['are', 'storms', 'likely', 'today'],
        'weather_query',
        ['Other', 'weather_descriptor', 'Other', 'date'],
        False,
        True,
        True,
        True,
        False,
        'slot_name',
        'weather_query are Other storms weather_descriptor likely Other today date'
    ),
    (
        ['are', 'storms', 'likely', 'today'],
        'weather_query',
        ['Other', 'weather_descriptor', 'Other', 'date'],
        True,
        True,
        True,
        True,
        False,
        'slot_name',
        'intent: weather_query annotation: are Other storms weather_descriptor likely Other today date'
    )
]

@pytest.mark.parametrize(
    "utt, intent, slots, use_output_descrip, intent_first, slots_mixed, toks_in_output, sentinels,"
    " inside_format, output",
    output_tests
)
def test_convert_intent_slots_to_t2t(
    utt,
    intent,
    slots,
    use_output_descrip,
    intent_first,
    slots_mixed,
    toks_in_output,
    sentinels,
    inside_format,
    output
):

    out = convert_intent_slots_to_t2t(
        utt,
        intent,
        slots,
        use_output_descrip,
        intent_first,
        slots_mixed,
        toks_in_output,
        sentinels,
        inside_format,
    )

    assert out == output


@pytest.mark.parametrize(
    "utt, intent, slots, use_output_descrip, intent_first, slots_mixed, toks_in_output, sentinels,"
    " inside_format, inp",
    output_tests
)
def test_convert_t2t_batch_to_intents_slots(
    utt,
    intent,
    slots,
    use_output_descrip,
    intent_first,
    slots_mixed,
    toks_in_output,
    sentinels,
    inside_format,
    inp
):

    inp = [inp]
    conv_intents, conv_slots = convert_t2t_batch_to_intents_slots(
        inp,
        use_output_descrip,
        intent_first,
        slots_mixed,
        toks_in_output,
        sentinels,
        inside_format
    )

    assert conv_intents == [intent]
    assert conv_slots == [slots]

input_tests = [
    (
        ['are', 'storms', 'likely', 'today'],
        'Annotate: ',
        False,
        ['Annotate:', 'are', 'storms', 'likely', 'today']
    ),
    (
        ['are', 'storms', 'likely', 'today'],
        'Annotate: ',
        True,
        ['Annotate:', '<extra_id_0>', 'are', '<extra_id_1>', 'storms', '<extra_id_2>', 'likely', '<extra_id_3>', 'today']
    )
]
@pytest.mark.parametrize(
    "utt, input_prompt, sentinels, conv_utt",
    input_tests
)
def test_convert_input_to_t2t(utt, input_prompt, sentinels, conv_utt):
    out = convert_input_to_t2t(utt, input_prompt, sentinels)
    assert out == conv_utt
