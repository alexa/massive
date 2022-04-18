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

This module contains code adapted from `transformers`.
Copyright and license details can be found in `NOTICE.md`.
"""

import os
import logging
import torch

logger = logging.getLogger('massive_logger')
if os.getenv('GLOBAL_RANK', 0) != 0:
    logger.setLevel(logging.NOTSET)

class CollatorMASSIVEIntentClassSlotFill:
    """
    Data collator for the MASSIVE intent classification and slot tasks
    Based on: https://github.com/huggingface/transformers/blob/v4.16.2/src/transformers/data/data_collator.py#L212

    :param tokenizer: The tokenizer
    :type tokenizer: transformers.PreTrainedTokenizerFast
    :param padding: True or 'longest' pads to longest seq in batch, 'max_length' to the specified
                    max_length, and False or 'do_not_pad' to not pad (default)
    :type padding: bool, str, or transformers.file_utils.PaddingStrategy
    :param max_length: max length for truncation and/or padding (optional)
    :type max_length: int
    :param pad_to_multiple_of: set the padding such that sequence is multiple of this (optional)
    :type pad_to_multiple_of: int
    """

    def __init__(self, tokenizer, max_length, padding='longest', pad_to_multiple_of=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = padding
        self.pad_to_multiple_of = pad_to_multiple_of

        self.col_chk = 0

    def __call__(self, batch):
        # On-the-fly tokenization and alignment -- do NOT use a pre-tokenized dataset

        tokenized_inputs = self.tokenizer(
            [item['utt'] for item in batch],
            truncation=True,
            is_split_into_words=True
        )

        # Align the labels with the tokenized utterance
        # adapted from here: https://huggingface.co/docs/transformers/custom_datasets#tok_ner
        for i, entry in enumerate(batch):
            label = entry['slots_num']
            word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to respective word
            previous_word_idx = None
            label_ids = []
            # Set the special tokens to -100.
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                # Only label the first token of a given word.
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                    previous_word_idx = word_idx
                else:
                    label_ids.append(-100)

            # Log example outputs from the collator for debugging
            if self.col_chk is not False:
                logger.info(f"Collator Check! utt: {entry['utt']}; intent label: "
                            f"{entry['intent_num']}; slot labels: {entry['slots_num']}, "
                            f"tokenized utt: {tokenized_inputs[i]}; word_ids: {word_ids}; "
                            f"label_ids: {label_ids}")
                if self.col_chk == 4:
                    self.col_chk = False
                else:
                    self.col_chk += 1

            if 'slots_num' in tokenized_inputs:
                tokenized_inputs['slots_num'].append(label_ids)
            else:
                tokenized_inputs['slots_num'] = [label_ids]

        # Pad the inputs
        pad_tok_inputs = self.tokenizer.pad(
            tokenized_inputs,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of
        )

        # Pad the slot labels
        sequence_length = torch.tensor(pad_tok_inputs["input_ids"]).shape[1]
        padding_side = self.tokenizer.padding_side
        if padding_side == "right":
            pad_tok_inputs['slots_num'] = [
                list(label) + [-100] * (sequence_length - len(label)) \
                               for label in pad_tok_inputs['slots_num']
            ]
        else:
            pad_tok_inputs['slots_num'] = [
                [-100] * (sequence_length - len(label)) + list(label) \
                 for label in pad_tok_inputs['slots_num']
            ]

        # Add in the intent labels
        pad_tok_inputs["intent_num"] = [item['intent_num'] for item in batch]

        # Convert to PyTorch tensors
        return {k: torch.tensor(v, dtype=torch.int64) for k, v in pad_tok_inputs.items()}
