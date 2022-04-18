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
import numpy as np

logger = logging.getLogger('massive_logger')
if os.getenv('GLOBAL_RANK', 0) != 0:
    logger.setLevel(logging.NOTSET)

class CollatorMASSIVET2TIntentClassSlotFill:
    """
    Data collator for formulating the MASSIVE intent classification and slot tasks as text-to-text
    Based on: https://github.com/huggingface/transformers/blob/v4.16.2/src/transformers/data/data_collator.py#L514

    :param tokenizer: The tokenizer
    :type tokenizer: transformers.PreTrainedTokenizerFast
    :param model: The model
    :type model: transformers.PreTrainedModel
    :param t2t_args: text-to-text formatting args. See docstring for convert_intent_slots_to_t2t
    :type t2t_args: dict
    :param padding: True or 'longest' pads to longest seq in batch, 'max_length' to the specified
                    max_length, and False or 'do_not_pad' to not pad (default)
    :type padding: bool, str, or transformers.file_utils.PaddingStrategy
    :param max_length: max length for truncation and/or padding (optional)
    :type max_length: int
    :param pad_to_multiple_of: set the padding such that sequence is multiple of this (optional)
    :type pad_to_multiple_of: int
    :param label_pad_token_id: ID used to ignore position for loss calculations. Default: -100
    :type label_pad_token_id: int
    :param return_tensors: the type of tensors to return, "np", "pt" (default), or "tf"
    :type return_tensors: str
    """

    def __init__(self, tokenizer, model, t2t_args, padding=True, max_length=None,
                 pad_to_multiple_of=None, label_pad_token_id=-100, return_tensors='pt'):

        self.tokenizer = tokenizer
        self.model = model
        self.padding = padding
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.label_pad_token_id = label_pad_token_id
        self.return_tensors = return_tensors

        self.t2t_args = t2t_args

        self.col_chk = True

    def __call__(self, batch, return_tensors=None):
        # Get the t2t conversion utility
        from massive.utils.training_utils import convert_input_to_t2t, convert_intent_slots_to_t2t

        if return_tensors is None:
            return_tensors = self.return_tensors

        labels = []

        for i, entry in enumerate(batch):

            t2t_out = convert_intent_slots_to_t2t(
                entry['utt'],
                entry['intent_str'],
                entry['slots_str'],
                **self.t2t_args
            )

            labels.append(t2t_out)

        if self.col_chk:
            logger.info(f"From Collator, labels before tokenization: {labels}")

        # Tokenize the labels
        tok_labels = self.tokenizer(labels, max_length=self.max_length, truncation=True)

        if self.col_chk:
            logger.info(f"From Collator, example of tokenized label IDs: {tok_labels.input_ids}")
            logger.info("From Collator, example of tokenized labels: "
                f"{[self.tokenizer.convert_ids_to_tokens(item) for item in tok_labels.input_ids]}")
        labels = tok_labels.input_ids

        # Add input prompt if needed
        for item in batch:
            item['utt'] = convert_input_to_t2t(item['utt'], **self.t2t_args)

        # tokenize the "features"
        features = self.tokenizer(
            [item['utt'] for item in batch],
            max_length=self.max_length,
            truncation=True,
            is_split_into_words=True
        )

        # convert to list of dictionaries
        new_feat = []
        for i in range(len(features['input_ids'])):
            new_feat.append({
                'input_ids': features['input_ids'][i],
                'attention_mask': features['attention_mask'][i],
                'labels': labels[i]
            })
        features = new_feat

        if self.col_chk:
            logger.info(f"From Collator, features after initial tokenization: {features}")
            logger.info("From Collator, features tokenized as tokens: "
                f"{[self.tokenizer.convert_ids_to_tokens(item['input_ids']) for item in features]}")

        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them
        # and needs them of the same length to return tensors.
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                    (max_label_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder \
                        if padding_side == "right" \
                        else remainder + feature["labels"]
                    )
                elif padding_side == "right":
                    feature["labels"] = np.concatenate(
                        [feature["labels"], remainder]
                    ).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate(
                        [remainder, feature["labels"]]
                    ).astype(np.int64)

        # add padding to everything besides the labels
        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        # prepare decoder_input_ids
        if (
            labels is not None
            and self.model is not None
            and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(
                labels=features["labels"]
            )
            features["decoder_input_ids"] = decoder_input_ids

         # Uncomment to log final features tensors
#        if self.col_chk:
#            logger.info(f"From Collator, final features: {features}")

        self.col_chk = False

        return features
