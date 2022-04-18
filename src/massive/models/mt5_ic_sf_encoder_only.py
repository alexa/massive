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

This module contains code adapted from `JointBERT`.
Copyright and license details can be found in `NOTICE.md`.
"""

# Modified from JointBERT:
# https://github.com/monologg/JointBERT/blob/master/model/modeling_jointbert.py

import torch
from torch import nn
from transformers import T5PreTrainedModel, MT5EncoderModel

class MT5IntentClassSlotFillEncoderOnly(T5PreTrainedModel):
    """
    A model based on the mT5 Encoder for joint intent classification and slot filling

    :param config: the config object
    :type config: transformers.T5Config
    :param intent_label_dict: A map from numeric representation to str representation for intent
    :type intent_label_dict: dict
    :param slot_label_dict: A map from numeric representation to str representation for slot
    :type slot_label_dict: dict
    """
    def __init__(self, config, intent_label_dict, slot_label_dict):
        super().__init__(config)
        self.config = config
        self.model_parallel = False
        self.num_intent_labels = len(intent_label_dict)
        self.num_slot_labels = len(slot_label_dict)

        # configure model to output all hidden states if not using last hidden layer
        # valid values for hidden_state_layer_for_class are 'last' or a layer number
        # note that layer count starts from 0, not 1
        self.hidden_layer_for_class = config.hidden_layer_for_class \
            if hasattr(config, 'hidden_layer_for_class') \
            else 'last'
        if self.hidden_layer_for_class in range(config.num_hidden_layers - 1):
            config.output_hidden_states = True

        self.mt5 = MT5EncoderModel(config=config)  # Load pretrained mt5 encoder

        # set defaults if not defined by user
        layer_dim = config.head_layer_dim if hasattr(config, 'head_layer_dim') else None
        num_layers = config.head_num_layers if hasattr(config, 'head_num_layers') else 1
        activation = config.head_activation if hasattr(config, 'head_activation') else 'gelu'
        dropout = config.head_dropout_rate \
            if hasattr(config, 'head_dropout_rate') \
            else config.hidden_dropout_prob
        pooling = config.head_intent_pooling \
            if hasattr(config, 'head_intent_pooling') \
            else 'first'

        # Instantiate the heads
        self.intent_classifier = IntentClassifier(
            input_dim=config.d_model,
            num_intent_labels=self.num_intent_labels,
            layer_dim=layer_dim,
            num_layers=num_layers,
            dropout_rate=dropout,
            activation=activation,
            pooling=config.head_intent_pooling
        )
        self.slot_classifier = SlotClassifier(
            input_dim=config.d_model,
            num_slot_labels=self.num_slot_labels,
            layer_dim=layer_dim,
            num_layers=config.head_num_layers,
            activation=activation,
            dropout_rate=dropout
        )

    def forward(self, input_ids, attention_mask, intent_num, slots_num):

        outputs = self.mt5(input_ids, attention_mask=attention_mask)

        # Choose the right layer for the hidden states
        if self.hidden_layer_for_class in range(self.config.num_hidden_layers - 1):
            hidden_states = outputs[1]
            attention_hidden_states = hidden_states[1:] # skip embedding outputs
            sequence_output = attention_hidden_states[self.hidden_layer_for_class]
        else:
            sequence_output = outputs[0]

        intent_logits = self.intent_classifier(sequence_output, attention_mask)
        slot_logits = self.slot_classifier(sequence_output)

        total_loss = 0

        # Intent Softmax
        if intent_num is not None:
            if self.num_intent_labels == 1:
                intent_loss_fct = nn.MSELoss()
                intent_loss = intent_loss_fct(intent_logits.view(-1), intent_num.view(-1))
            else:
                intent_loss_fct = nn.CrossEntropyLoss()
                intent_loss = intent_loss_fct(
                    intent_logits.view(-1, self.num_intent_labels), intent_num.view(-1)
                )
            total_loss += intent_loss

        # Slot Softmax
        if slots_num is not None:
            slot_loss_fct = nn.CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = slot_logits.view(-1, self.num_slot_labels)[active_loss]
                active_labels = slots_num.view(-1)[active_loss]
                slot_loss = slot_loss_fct(active_logits, active_labels)
            else:
                slot_loss = slot_loss_fct(
                    slot_logits.view(-1, self.num_slot_labels), slots_num.view(-1)
                )
            total_loss += self.config.slot_loss_coef * slot_loss

        outputs = (total_loss, (intent_logits, slot_logits))

        # (loss), logits, (hidden_states), (attentions)
        # Logits is a tuple of intent and slot logits
        return outputs


class IntentClassifier(nn.Module):
    def __init__(self, input_dim, num_intent_labels, layer_dim=None, num_layers=1,
                 activation='gelu', dropout_rate=0., pooling='first'):

        super().__init__()
        layer_dim = layer_dim if layer_dim else input_dim
        self.pooling = pooling
        ic_head = []

        # Create the intermediate layers
        if num_layers > 0:
            for l in range(num_layers):
                ic_head.append(nn.Dropout(dropout_rate))
                ic_head.append(nn.Linear(input_dim, layer_dim))
                input_dim = layer_dim

                if activation == 'gelu':
                    ic_head.append(nn.GELU())
                elif activation == 'elu':
                    ic_head.append(nn.ELU())
                elif activation == 'tanh':
                    ic_head.append(nn.Tanh())
                else:
                    raise NotImplementedError(f"Activation {activation} is not implemented")

        # Final layer, condensed to number of intent labels
        ic_head.append(nn.Dropout(dropout_rate))
        ic_head.append(nn.Linear(input_dim, num_intent_labels))

        self.ic_head = nn.Sequential(*ic_head)

    def forward(self, inp, attention_mask):

        if self.pooling == 'first':
            # Get hidden states from first token in seq
            inp = inp[:, 0]
        elif self.pooling == 'max':
            mask_expand = attention_mask.unsqueeze(-1).expand(inp.size()).float()
            inp[mask_expand == 0] = -1e9 # set padding to large negative
            inp = torch.max(inp, 1)[0]
        elif self.pooling == 'mean':
            # see: https://huggingface.co/sentence-transformers/bert-base-nli-mean-tokens
            mask_expand = attention_mask.unsqueeze(-1).expand(inp.size()).float()
            inp = torch.sum(inp * mask_expand, 1) / torch.clamp(mask_expand.sum(1), min=1e-9)
        else:
            raise NotImplementedError(f"Pooling type {self.pooling} not implemented")

        return self.ic_head(inp)


class SlotClassifier(nn.Module):
    def __init__(self, input_dim, num_slot_labels, layer_dim=None, num_layers=1,
                 activation='gelu', dropout_rate=0.):

        super().__init__()
        layer_dim = layer_dim if layer_dim else input_dim
        sf_head = []

        # Create the intermediate layers
        if num_layers > 0:
            for l in range(num_layers):
                sf_head.append(nn.Dropout(dropout_rate))
                sf_head.append(nn.Linear(input_dim, layer_dim))
                input_dim = layer_dim

                if activation == 'gelu':
                    sf_head.append(nn.GELU())
                elif activation == 'elu':
                    sf_head.append(nn.ELU())
                elif activation == 'tanh':
                    sf_head.append(nn.Tanh())
                else:
                    raise NotImplementedError(f"Activation {activation} is not implemented")

        # Final layer, condensed to number of intent labels
        sf_head.append(nn.Dropout(dropout_rate))
        sf_head.append(nn.Linear(input_dim, num_slot_labels))

        self.sf_head = nn.Sequential(*sf_head)

    def forward(self, inp):
        return self.sf_head(inp)
