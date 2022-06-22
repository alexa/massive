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

from massive.loaders.collator_ic_sf import CollatorMASSIVEIntentClassSlotFill
from massive.loaders.collator_t2t_ic_sf import CollatorMASSIVET2TIntentClassSlotFill
from massive.models.xlmr_ic_sf import XLMRIntentClassSlotFill
from massive.models.mt5_ic_sf_encoder_only import MT5IntentClassSlotFillEncoderOnly
import datasets
import json
import logging
from math import sqrt
import numpy as np
import os
from seqeval.metrics import f1_score
import sklearn.metrics as sklm
import torch
from transformers import (
    MT5Config,
    MT5ForConditionalGeneration,
    MT5TokenizerFast,
    XLMRobertaConfig,
    XLMRobertaTokenizerFast
)

logger = logging.getLogger('massive_logger')
if os.getenv('GLOBAL_RANK', 0) != 0:
    logger.setLevel(logging.WARNING)

def init_model(conf, intents, slots, return_hpo_fn=False):
    """
    Initialize a model based on the type given in the config

    :param conf: The MASSIVE configuration object
    :type conf: massive.Configuration
    :param intent: A dictionary mapping each intent's numerical index to the intent
    :type intent: dict
    :param slots: A dictionary mapping each slot's numerical index to the slot
    :type slots: dict
    :return: The loaded model
    :rtype: nn.Module
    """

    def hpo_model_init(hpo_params=None):

        if hpo_params is not None:
            logger.info(f"Chosen hyperparameter values: {hpo_params}")
            for param in hpo_params:
                if conf.get(param):
                    logger.warning(f"Overriding {param} with {hpo_params[param]}")
                    conf.override(param, hpo_params[param])
                else:
                    logger.warning(f"{param} not found in config. Assuming it's a trainer arg")

        # Get the model class and config
        config_args = conf.get('model.model_config_args')
        if conf.get('model.type') == 'xlmr intent classification slot filling':
            model_config = XLMRobertaConfig(**config_args) if config_args else None
            model_cls = XLMRIntentClassSlotFill
            model_kwargs = {'intent_label_dict': intents, 'slot_label_dict': slots}
        elif conf.get('model.type') == 'mt5 for conditional generation':
            model_config = MT5Config(**config_args) if config_args else None
            model_cls = MT5ForConditionalGeneration
            model_kwargs = {}
        elif conf.get('model.type') == 'mt5 intent classification slot filling encoder only':
            model_config = MT5Config(**config_args) if config_args else None
            model_cls = MT5IntentClassSlotFillEncoderOnly
            model_kwargs = {'intent_label_dict': intents, 'slot_label_dict': slots}
        # add more models here as additional elif statements
        else:
            raise NotImplementedError(f"Model type {conf.get('model.type')} not found!")

        # Instantiate the model
        if conf.get('model.checkpoint'):
            return model_cls.from_pretrained(conf.get('model.checkpoint'), **model_kwargs)
        else:
            model = model_cls(model_config, **model_kwargs)

        # Load pretrained weights
        if conf.get('model.pretrained_weights'):
            logger.info(f"Loading pretrained weights from {conf.get('model.pretrained_weights')}")
            mod_weights = torch.load(conf.get('model.pretrained_weights'), map_location='cpu')
            if conf.get('model.pretrained_weight_prepend'):
                prepend = conf.get('model.pretrained_weight_prepend')
                # sometimes this parses as a list, sometimes as list of lists. Remove outer list.
                mod_weights = {(prepend + k): v for k, v in mod_weights.items()}
            if conf.get('model.pretrained_weight_substring_transform'):
                rpl = conf.get('model.pretrained_weight_substring_transform')
                # sometimes this parses as a list, sometimes as list of lists. Remove outer list.
                rpl = rpl[0] if type(rpl[0]) == list else rpl
                mod_weights = {k.replace(rpl[0], rpl[1]): v for k, v in mod_weights.items()}
            load_strict = conf.get('model.strict_load_pretrained_weights')
            load_info = model.load_state_dict(mod_weights, strict=load_strict)
            logger.info(f"Finished loading pretrained model. Results: {load_info}")
        else:
            logger.info("No pretrained weights provided. All weights will be trained from scratch.")

        # Freeze layers if needed
        if conf.get('model.model_config_args.freeze_layers'):
            layers_to_freeze = conf.get('model.model_config_args.freeze_layers').split(',')
            all_names = []
            for name, param in model.named_parameters():
                all_names.append(name)
                if name in layers_to_freeze:
                    logger.info(f"Freezing layer {name}")
                    param.requires_grad = False
                    layers_to_freeze.remove(name)
            if not (layers_to_freeze == [''] or layers_to_freeze == []):
                logger.info(f"all layer names: {all_names}")
                raise KeyError(f"Could not find the following layers to freeze: {layers_to_freeze}")

        return model

    if return_hpo_fn:
        return hpo_model_init
    return hpo_model_init()

def init_tokenizer(conf):
    """
    Initialize a tokenizer based on the type given in the config
    :param conf: The MASSIVE configuration object
    :type conf: massive.Configuration
    :return: The loaded tokenizer
    :rtype: PreTrainedTokenizerFast
    """
    if conf.get('tokenizer.type') == 'xlmr base':
        return XLMRobertaTokenizerFast(**conf.get('tokenizer.tok_args'))
    elif conf.get('tokenizer.type') == 'mt5':
        return MT5TokenizerFast(**conf.get('tokenizer.tok_args'))
    # Add more tokenizers here
    else:
        raise NotImplementedError('Tokenizer type not found!')

def prepare_train_dev_datasets(conf, tokenizer, seed=42):
    """
    Prepare the training and dev datasets based on the config.

    :param conf: The MASSIVE configuration object
    :type conf: massive.Configuration
    :param tokenizer: The loaded tokenizer
    :type tokenizer: PreTrainedTokenizerFast
    :return: train set, dev set, an intent dictionary, a slot dictionary
    :rtype: tuple(Dataset, Dataset, dict, dict)
    """

    train = datasets.load_from_disk(conf.get('train_val.train_dataset'))
    train = train.shuffle(seed=seed)

    # Filter to specific train locales
    train_locales = conf.get('train_val.train_locales', default='all')
    if train_locales != 'all' and train_locales != ['all']:
        logger.info(f"Filtering train dataset to locale(s): {train_locales}")
        if type(train_locales) == str:
            train_locales = [train_locales]
        train = train.filter(lambda x: x['locale'] in train_locales)

    logger.info(f"The features of the train dataset: {train.features}")
    logger.info(f"Length of the train dataset: {len(train)}")

    dev = datasets.load_from_disk(conf.get('train_val.dev_dataset'))

    # Filter to specific dev locales
    dev_locales = conf.get('train_val.dev_locales', default='all')
    if dev_locales != 'all' and dev_locales != ['all']:
        logger.info(f"Filtering dev dataset to locale(s): {dev_locales}")
        if type(dev_locales) == str:
            dev_locales = [dev_locales]
        dev = dev.filter(lambda x: x['locale'] in dev_locales)

    # Remove specified locales in the dev set
    dev_locales_remove = conf.get('train_val.dev_locales_remove', default='none')
    if dev_locales_remove != 'none' and dev_locales_remove != ['none']:
        logger.info(f"Removing locale(s) from dev dataset: {dev_locales_remove}")
        if type(dev_locales_remove) == str:
            dev_locales_remove = [dev_locales_remove]
        dev = dev.filter(lambda x: x['locale'] not in dev_locales_remove)

    # Shuffle the dev set
    dev = dev.shuffle(seed=seed)

    # Shorten the dev set to the first N examples if desired
    if conf.get('train_val.dev_shorten_to', None):
        dev = dev.select(range(conf.get('train_val.dev_shorten_to')))

    # Load the intent and slot labels
    with open(conf.get('train_val.intent_labels'), 'r') as i:
        intent_labels = json.load(i)
    with open(conf.get('train_val.slot_labels'), 'r') as s:
        slot_labels = json.load(s)

    logger.info(f"The features of the train dataset: {train.features}")
    logger.info(f"Length of the train dataset: {len(train)}")
    logger.info(f"Length of the loaded dev dataset: {len(dev)}")

    return train, dev, intent_labels, slot_labels

def prepare_test_dataset(conf, tokenizer, seed=42):
    """
    Prepare the test dataset based on the config.

    :param conf: The MASSIVE configuration object
    :type conf: massive.Configuration
    :param tokenizer: The loaded tokenizer
    :type tokenizer: PreTrainedTokenizerFast
    :return: test dataset, an intent dictionary, a slot dictionary
    :rtype: tuple(Dataset, dict, dict)
    """

    test = datasets.load_from_disk(conf.get('test.test_dataset'))
    test = test.shuffle(seed=seed)

    # Filter to specific test locales
    test_locales = conf.get('test.test_locales', default='all')
    if test_locales != 'all' and test_locales != ['all']:
        logger.info(f"Filtering test dataset to locale(s): {test_locales}")
        if type(test_locales) == str:
            test_locales = [test_locales]
        test = test.filter(lambda x: x['locale'] in test_locales)

    # Remove specified locales in the test set
    test_locales_remove = conf.get('test.test_locales_remove', default='none')
    if test_locales_remove != 'none' and test_locales_remove != ['none']:
        logger.info(f"Removing locale(s) from test dataset: {test_locales_remove}")
        if type(test_locales_remove) == str:
            test_locales_remove = [test_locales_remove]
        test = test.filter(lambda x: x['locale'] not in test_locales_remove)

    if conf.get('test.test_shorten_to', None):
        test = test.select(range(conf.get('test.test_shorten_to')))

    logger.info(f"The features of the test dataset: {test.features}")
    logger.info(f"Length of the test dataset: {len(test)}")
    with open(conf.get('test.intent_labels'), 'r') as i:
        intent_labels = json.load(i)
    with open(conf.get('test.slot_labels'), 'r') as s:
        slot_labels = json.load(s)
    return test, intent_labels, slot_labels

def prepare_collator(conf, tokenizer, model=None):
    """
    Prepare the collator based on the config.

    :param conf: The MASSIVE configuration object
    :type conf: massive.Configuration
    :param tokenizer: The loaded tokenizer
    :type tokenizer: PreTrainedTokenizerFast
    :return: the collator object
    """
    if conf.get('collator.type') == 'massive intent class slot fill':
        return CollatorMASSIVEIntentClassSlotFill(
            tokenizer=tokenizer,
            **conf.get('collator.args')
        )
    if conf.get('collator.type') == 'massive text to text intent class slot fill':
        return CollatorMASSIVET2TIntentClassSlotFill(
            tokenizer=tokenizer,
            model=model,
            **conf.get('collator.args')
        )
    else:
        raise NotImplementedError('Collator type not found!')

def create_compute_metrics(intent_labels, slot_labels, conf, tokenizer=None, ignore_labels=None,
                           metrics='all'):
    """
    Create a `compute_metrics` function for this task

    :param intent_labels: A dictionary mapping each intent's numerical index to the intent
    :type slot_labels: dict
    :param slot_labels: A dictionary mapping each slot's numerical index to the slot
    :type slot_labels: dict
    :param conf: The MASSIVE configuration object
    :type conf: massive.Configuration
    :param ignore_labels: The labels to ignore
    :type ignore_labels: list or str
    :param metrics: The metrics to calculate
    :type metrics: list or str
    :return: the `compute_metrics` function
    :rtype: Callable
    """

    # Determine any labels that should be ignored when calculating F1 score (EX: Other)
    ignore_labels = [] if ignore_labels is None else ignore_labels
    ignore_num_lab = [int(k) for k, v in slot_labels.items() if v in ignore_labels]

    if type(metrics) != list:
        metrics = [metrics]

    # COLLATOR: MASSIVE INTENT CLASS SLOT FILL
    if conf.get('collator.type') == 'massive intent class slot fill':
        def compute_metrics(p):
            # p is named tuple with `predictions` and `label_ids`.
            # p.predictions is a tuple of two elements, the first being the intent classification
            # predictions of size num_examples and the second being the slot classification preds
            # of size num_examples. Each intent classification pred is of size num_intent_classes,
            # and each slot classification prediction is of shape (seq_len, num_slot_classes)
            # label_ids is tuple of two elements, first array of all IC labels (size num_examples)
            # The second element is size num_examples with each entry sized seq_len x num_slot_class

            intent_preds = p.predictions[0]
            slot_preds = p.predictions[1]

            intent_label_tuple = p.label_ids[0]
            slot_label_tuple = p.label_ids[1]

            intent_preds_am = [np.argmax(x) for x in intent_preds]
            slot_preds_am = [np.argmax(x, axis=1) for x in slot_preds]

            # merge -100, which we used for the subsequent subwords in a full word after tokenizing
            labels_merge = [-100]

            return eval_preds(
                pred_intents=intent_preds_am,
                lab_intents=intent_label_tuple,
                pred_slots=slot_preds_am,
                lab_slots=slot_label_tuple,
                eval_metrics=metrics,
                labels_merge=labels_merge,
                labels_ignore=ignore_num_lab,
                pad='Other'
            )

    # COLLATOR: MASSIVE TEXT TO TEXT INTENT CLASS SLOT FILL
    elif conf.get('collator.type') == 'massive text to text intent class slot fill':
        def compute_metrics(p):
            # p is named tuple with `predictions` and `label_ids`

            clean_labels = []
            for lab in p.label_ids:
                lab = lab[lab != -100]
                lab = lab[lab != tokenizer.pad_token_id]
                lab = lab[lab != tokenizer.eos_token_id]
                clean_labels.append(lab)

            clean_preds = []
            for pred in p.predictions:
                pred = pred[pred != -100]
                pred = pred[pred != tokenizer.pad_token_id]
                pred = pred[pred != tokenizer.eos_token_id]
                clean_preds.append(pred)

            clean_lab_dec = tokenizer.batch_decode(clean_labels, skip_special_tokens=True)
            clean_pred_dec = tokenizer.batch_decode(clean_preds, skip_special_tokens=True)

            t2t_args = conf.get('collator.args.t2t_args')
            intents_pred, slots_pred_all = convert_t2t_batch_to_intents_slots(
                clean_pred_dec, **t2t_args)
            intents_lab, slots_lab_all = convert_t2t_batch_to_intents_slots(
                clean_lab_dec, **t2t_args)

            return eval_preds(
                pred_intents=intents_pred,
                lab_intents=intents_lab,
                pred_slots=slots_pred_all,
                lab_slots=slots_lab_all,
                eval_metrics=metrics,
                labels_ignore=ignore_labels,
                pad='Other'
            )

    else:
        raise NotImplementedError('Could not find a compute_metrics function for your collator')

    return compute_metrics

def convert_to_bio(seq_tags, outside='Other', labels_merge=None):
    """
    Converts a sequence of tags into BIO format. EX:

        ['city', 'city', 'Other', 'country', -100, 'Other']
        to
        ['B-city', 'I-city', 'O', 'B-country', 'I-country', 'O']
        where outside = 'Other' and labels_merge = [-100]

    :param seq_tags: the sequence of tags that should be converted
    :type seq_tags: list
    :param outside: The label(s) to put outside (ignore). Default: 'Other'
    :type outside: str or list
    :param labels_merge: The labels to merge leftward (i.e. for tokenized inputs)
    :type labels_merge: str or list
    :return: a BIO-tagged sequence
    :rtype: list
    """

    seq_tags = [str(x) for x in seq_tags]

    outside = [outside] if type(outside) != list else outside
    outside = [str(x) for x in outside]

    if labels_merge:
        labels_merge = [labels_merge] if type(labels_merge) != list else labels_merge
        labels_merge = [str(x) for x in labels_merge]
    else:
        labels_merge = []

    bio_tagged = []
    prev_tag = None
    for tag in seq_tags:
        if prev_tag == None and tag in labels_merge:
            bio_tagged.append('O')
        elif tag in outside:
            bio_tagged.append('O')
            prev_tag = tag
        elif tag != prev_tag and tag not in labels_merge:
            bio_tagged.append('B-' + tag)
            prev_tag = tag
        elif tag == prev_tag or tag in labels_merge:
            if prev_tag in outside:
                bio_tagged.append('O')
            else:
                bio_tagged.append('I-' + prev_tag)

    return bio_tagged

def eval_preds(pred_intents=None, lab_intents=None, pred_slots=None, lab_slots=None,
               eval_metrics='all', labels_ignore='Other', labels_merge=None, pad='Other'):
    """
    Function to evaluate the predictions from a model

    :param pred_intents: a list of predicted intents
    :type pred_intents: list
    :param lab_intents: a list of intents labels (ground truth)
    :type lab_intents: list
    :param pred_slots: a list of predicted slots, where each entry is a list of token-based slots
    :type pred_slots: list
    :param lab_slots: a list of slots labels (ground truth)
    :type lab_slots: list
    :param eval_metrics: The metrics to include. Options are 'all', 'intent_acc', 'ex_match_acc',
                         'slot_micro_f1'
    :type eval_metrics: str
    :param labels_ignore: The labels to ignore (prune away). Default: ['Other']
    :type labels_ignore: str or list
    :param labels_merge: The labels to merge leftward (i.e. for tokenized inputs)
    :type labels_merge: str or list
    :param pad: The value to use when padding slot predictions to match the length of ground truth
    :type pad: str
    """

    results = {}

    # Check lengths
    if pred_intents is not None and lab_intents is not None:
        assert len(pred_intents) == len(lab_intents),"pred_intents and lab_intents must be same len"
    if pred_slots is not None and lab_slots is not None:
        assert len(pred_slots) == len(lab_slots), "pred_slots and lab_slots must be same length"

    if ('intent_acc' in eval_metrics) or ('all' in eval_metrics):
        intent_acc = sklm.accuracy_score(lab_intents, pred_intents)
        results['intent_acc'] = intent_acc
        # Assuming normal distribution. Multiply by z (from "z table") to get confidence int
        results['intent_acc_stderr'] = sqrt(intent_acc*(1-intent_acc)/len(pred_intents))

    if lab_slots is not None and pred_slots is not None:
        bio_slot_labels, bio_slot_preds = [], []
        for lab, pred in zip(lab_slots, pred_slots):

            # Pad or truncate prediction as needed using `pad` arg
            if type(pred) == list:
                pred = pred[:len(lab)] + [pad]*(len(lab) - len(pred))

            # Fix for Issue 21 -- subwords after the first one from a word should be ignored
            for i, x in enumerate(lab):
                if x == -100:
                    pred[i] = -100

            # convert to BIO
            bio_slot_labels.append(
                convert_to_bio(lab, outside=labels_ignore, labels_merge=labels_merge)
            )
            bio_slot_preds.append(
                convert_to_bio(pred, outside=labels_ignore, labels_merge=labels_merge)
            )

    if ('slot_micro_f1' in eval_metrics) or ('all' in eval_metrics):

        # from seqeval
        smf1 = f1_score(bio_slot_labels, bio_slot_preds)
        results['slot_micro_f1'] = smf1
        # Assuming normal distribution. Multiply by z (from "z table") to get confidence int
        total_slots = sum([len(x) for x in bio_slot_preds])
        results['slot_micro_f1_stderr'] = sqrt(smf1*(1-smf1)/total_slots)

    if ('ex_match_acc' in eval_metrics) or ('all' in eval_metrics):
        # calculate exact match accuracy (~0.01 seconds)
        matches = 0
        denom = 0
        for p_int, p_slot, l_int, l_slot in zip(pred_intents,
                                                bio_slot_preds,
                                                lab_intents,
                                                bio_slot_labels):

            if (p_int == l_int) and (p_slot == l_slot):
                matches += 1
            denom += 1
        emacc = matches / denom

        results['ex_match_acc'] = emacc
        # Assuming normal distribution. Multiply by z (from "z table") to get confidence int
        results['ex_match_acc_stderr'] = sqrt(emacc*(1-emacc)/len(pred_intents))

    return results

def output_predictions(outputs, intent_labels, slot_labels, conf, tokenizer=None,
                       combine_slots=True, remove_slots=None, add_pred_parse=True,
                       save_to_file=True):
    """
    :param outputs: The outputs from the model
    :type outputs: named_tuple
    :param intent_labels: A dictionary mapping each intent's numerical index to the intent
    :type slot_labels: dict
    :param slot_labels: A dictionary mapping each slot's numerical index to the slot
    :type slot_labels: dict
    :param conf: The MASSIVE configuration object
    :type conf: massive.Configuration
    :param tokenizer: The tokenizer
    :type tokenizer: PreTrainedTokenizerFast
    :param combine_slots: Whether or not to combine adjacent same-slotted tokens to one slot
    :type combine_slots: bool
    :param remove_slots: Slots to remove. Default ['Other']
    :type remove_slots: list
    :param add_pred_parse: Whether to add the SLURP-style parsed output
    :type add_pred_parse: bool
    :param save_to_file: Whether to save predictions to the file given in the config
    :type save_to_file: bool
    """

    remove_slots = ['Other'] if not remove_slots else remove_slots

    pred_file = conf.get('train_val.predictions_file')

    if pred_file and (conf.get('train_val.trainer_args.locale_eval_strategy') != 'all only'):
        raise NotImplementedError("You must use 'all only' as the locale_eval_strategy if you"
                                  " specify a predictions file")

    final_outputs = []

    # if there is a space within sequence of subwords that should be joined back together,
    # it's probably because the tokenizer converted a Zero Width Space to a normal space.
    # Make this False to not replace the space with a ZWSP when re-joining subwords
    replace_zwsp = conf.get('test.replace_inner_space_zwsp', default=True)

    if conf.get('collator.type') == 'massive intent class slot fill':
        # Create strings of the slot predictions
        intent_preds, slot_preds = outputs.predictions[0], outputs.predictions[1]
        intent_preds_am = [np.argmax(x) for x in intent_preds]
        intent_preds_str = [intent_labels[str(x)] for x in intent_preds_am]

        slot_preds_am = [np.argmax(x, axis=1) for x in slot_preds]
        slot_preds_str = []
        for example in slot_preds_am:
            slot_preds_str.append([slot_labels[str(x)] for x in example])

        # Iterate through the examples
        for eyed, loc, utt, tok_utt, intent_pred, slot_pred, subword_align in zip(
            outputs.ids,
            outputs.locales,
            outputs.utts,
            outputs.tok_utts,
            intent_preds_str,
            slot_preds_str,
            outputs.subword_aligns):

            line = {}
            line['id'], line['locale'], line['utt'], line['pred_intent'] = eyed,loc,utt,intent_pred

            # Determine slot predictions
            running_detok_idx, tok, slot, slots = -1, '', '', []
            for tok_idx, detok_idx in enumerate(subword_align):
                if detok_idx is None:
                    continue
                # Combine the subwords that had been broken up
                if detok_idx == running_detok_idx:
                    # If there is a \u2581 within what was a single "word" from the input data,
                    # then it's probably from a zero-width space, \u200b, which the tokenizer
                    # converted to a space. We don't want these extra spaces, so they are removed
                    if replace_zwsp:
                        tok_repl = tok_utt[tok_idx].replace(u'\u2581',u'\u200b')
                    else:
                        tok_repl = tok_utt[tok_idx]
                    tok += tok_repl

                # Record the token and slot and start a new one
                else:
                    if running_detok_idx != -1:
                        tok = tok.replace('▁',' ')
                        tok = tok.strip()
                        slots.append((tok, slot))
                    slot = slot_pred[tok_idx]
                    tok = tok_utt[tok_idx]
                running_detok_idx = detok_idx
            # Add the last token and slot
            tok = tok.replace('▁',' ')
            tok = tok.strip()
            slots.append((tok, slot))

            line['pred_slots'] = slots
            final_outputs.append(line)

    elif conf.get('collator.type') == 'massive text to text intent class slot fill':
        clean_preds = []

        # Remove padding and other special tokens
        for pred in outputs.predictions:
            pred = pred[pred != -100]
            pred = pred[pred != tokenizer.pad_token_id]
            pred = pred[pred != tokenizer.eos_token_id]
            clean_preds.append(pred)

        clean_pred_dec = tokenizer.batch_decode(clean_preds, skip_special_tokens=True)

        t2t_args = conf.get('collator.args.t2t_args')
        intents_pred, slots_pred = convert_t2t_batch_to_intents_slots(clean_pred_dec, **t2t_args)

        # Convert slots to (token, slot) format
        slots_pred_tup = []
        for utt, slots in zip(outputs.utts, slots_pred):
            new = []
            # Pad with Other to length of utt
            slots = slots[:len(utt)] + ['Other'] * (len(utt) - len(slots))
            # make tuple with token
            for tok, slot in zip(utt, slots):
                new.append((tok, slot))
            slots_pred_tup.append(new)


        # Create the list of dicts
        final_outputs = [
            {'id': eyed, 'locale': loc, 'utt': utt, 'pred_intent': intent, 'pred_slots': slot} for \
            eyed, loc, utt, intent, slot in \
            zip(outputs.ids, outputs.locales, outputs.utts, intents_pred, slots_pred_tup)
        ]

    else:
        raise NotImplementedError(f"Collator {conf.get('collator.type')} not known")


    for line in final_outputs:
        slots = []
        for tup in line['pred_slots']:
            if not slots:
                slots.append(tup)
                slots_idx = 0
            # if slot the same as previous token, combine
            elif tup[1] == slots[slots_idx][1]:
                slots[slots_idx] = (slots[slots_idx][0] + ' ' + tup[0], slots[slots_idx][1])
            # otherwise add to end
            else:
                slots.append(tup)
                slots_idx += 1

        # Create a SLURP-like version of each utterance
        if add_pred_parse:
            parse = ''
            for slot in slots:
                if slot[1] in remove_slots:
                    parse += ' ' + slot[0]
                else:
                    parse += ' [' + slot[1] + ' : ' + slot[0] + ']'
            line['pred_annot_utt'] = parse.strip()

        # If adjacent tokens have the same slot, combine them
        if combine_slots:
            line['pred_slots'] = slots

        # Remove slots in the remove_slots list
        if remove_slots:
            line['pred_slots'] = [x for x in line['pred_slots'] if x[1] not in remove_slots]

    logger.info(f"Example of final output:\n{final_outputs[:2]}")
    logger.info(f"Writing to {conf.get('test.predictions_file')}")

    # True to output escaped unicode codes or False to output unicode
    ensure_ascii = conf.get('test.predictions_ensure_ascii', default=False)

    if save_to_file:
        with open(conf.get('test.predictions_file'), 'w', encoding='utf-8') as f:
            for line in final_outputs:
                f.write(json.dumps(line, ensure_ascii=ensure_ascii) + '\n')

    return final_outputs

def convert_input_to_t2t(utt, input_prompt="Annotate: ", sentinels=False, **kwargs):
    """
    Helper function to convert an input to text-to-text format

    :param utt: A list of words pre-tokenization. The input to the model pre-formatting.
    :type utt: list
    :param input_prompt: The prompt to add before the input. Default: "Annotate: "
    :type intent: str
    :param sentinels: Whether to add T5 sentinels before each token. Default: False
                      See: https://arxiv.org/pdf/2203.08378.pdf
    :type sentinels: bool
    :return: the reformatted input to the model
    :rtype: list
    """

    if sentinels:
        new_utt, sent_id = [], 0
        for tok in utt:
            new_utt.append('<extra_id_' + str(sent_id) + '>')
            new_utt.append(tok)
            sent_id += 1
        utt = new_utt

    if input_prompt:
        utt.insert(0, input_prompt.strip())

    return utt

def convert_intent_slots_to_t2t(utt, intent, slots, use_output_descrip=False, intent_first=False,
                                slots_mixed=False, toks_in_output=False, sentinels=False,
                                inside_format='slot_name', outside_label='Other', **kwargs):
    """
    Helper function to convert an intent and 0 or more slots to a text-to-text format

    :param utt: A list of words pre-tokenization
    :type utt: list
    :param intent: The intent
    :type intent: str
    :param slots: a list of the slots
    :type slots: list
    :param use_output_descrip: Whether or not to include descriptive prompts in the output, being
                               'tokens: ' and 'annotations' for non mixed slotting or 'annotation: '
                               for mixed slotting. Default: False
    :type use_output_descrip: bool
    :param intent_first: Whether to put the intent before the slots and utterance (True) or after
                         Default: True
    :type intent_first: bool
    :param slots_mixed: Whether to put each slot after its respective token (True) or to put all
                        slots after all tokens (False). Default: False
    :type slots_mixed: bool
    :param input_prompt: The text prompt for the input. Leave blank for no prompt.
                         Default: 'Annotate: '
    :type input_prompt: str
    :param toks_in_output: Whether to put tokens in the output or not. Default: False. If this is
                           True, then slots_mixed must be False
    :type toks_in_output: bool
    :param sentinels: Whether to add T5 sentinels before each token. Overrides toks_in_output and
                      slots_mixed. Default: False
                      See: https://arxiv.org/pdf/2203.08378.pdf
    :type sentinels: bool
    :param inside_format: The slot to use for the inside of a multi-word slot. Options are
                          "slot_name", in which the slot name is repeated, "inside_slot_name",
                          in which "I-" is added to the slot name, or "inside", in which "I" is
                          used on its own.
    :type inside_format: str
    :param outside_label: The word used for non-slotted tokens. Default: Other
    :type outside_label: str

    :return: the output text
    :rtype: str
    """

    # Suppose we have:
    # intent: calendar_set
    # annot_utt: [event_name: meetings] this [date: monday]

    if sentinels:
        # using sentinels is the same as doing slots_mixed and toks_in_output and converting the
        # utterance to a sequence of sentinels
        toks_in_output = True
        slots_mixed = True
        new_utt, sent_id = [], 0
        for tok in utt:
            new_utt.append('<extra_id_' + str(sent_id) + '>')
            sent_id += 1
        utt = new_utt

    # Modify for inside format if needed
    new_slots = []
    for idx, slot in enumerate(slots):
        if idx > 0 and slot != outside_label:
            if slot == slots[idx-1]:
                if inside_format == 'inside_slot_name':
                    new_slots.append("I-"+slot)
                    continue
                if inside_format == 'inside':
                    new_slots.append("I")
                    continue
        new_slots.append(slot)
    slots = new_slots

    slot_list = slots
    toks, slots = ' '.join(utt), ' '.join(slots)

    if use_output_descrip:
        intent_pre = 'intent: '
        mixed_pre = 'annotation: '
        non_mixed_tok_pre = 'tokens: '
        non_mixed_slot_pre = 'annotations: '

    if slots_mixed:
        if toks_in_output is False:
            raise ValueError('slots_mixed cannot be True if toks_in_output is False')

        # annot_utt = "meeting event_name this Other monday date"
        annot_utt = [tok + ' ' + slot for tok, slot in zip(utt, slot_list)]
        annot_utt = ' '.join(annot_utt)
        if use_output_descrip:
            # annot_utt = "annotation: meeting event_name this Other monday date"
            annot_utt = mixed_pre + annot_utt
    else:
        # remove tokens entirely if needed
        if toks_in_output:
            toks += ' '
        else:
            toks = ''
            non_mixed_tok_pre = ''

        if use_output_descrip:
            # "tokens: meeting this monday annotations: event_name Other date"
            annot_utt = non_mixed_tok_pre + toks + non_mixed_slot_pre + slots
        else:
            # annot_utt = "meeting this monday event_name Other date"
            annot_utt = toks + slots

    # place intent either at beginning or end
    if intent_first:
        if use_output_descrip:
            return intent_pre + intent + ' ' + annot_utt
        return intent + ' ' + annot_utt
    else:
        if use_output_descrip:
            return annot_utt + ' ' + intent_pre + intent
        return annot_utt + ' ' + intent


def convert_t2t_batch_to_intents_slots(mod_out, use_output_descrip=False, intent_first=False,
                                       slots_mixed=False, toks_in_output=False, sentinels=False,
                                       inside_format='slot_name', outside_label='Other', **kwargs):
    """
    Helper function to convert an intent and 0 or more slots to a text-to-text format

    :param model_out: A list of outputs from the model, each a detokenized string
    :type model_out: list
    :param use_output_descrip: Whether or not to include descriptive prompts in the output, being
                               'tokens: ' and 'annotations' for non mixed slotting or 'annotation: '
                               for mixed slotting. Default: False
    :type use_output_descrip: bool
    :param intent_first: Whether to put the intent before the slots and utterance (True) or after
                         Default: True
    :type intent_first: bool
    :param slots_mixed: Whether to put each slot after its respective token (True) or to put all
                        slots after all tokens (False). Default: False
    :type slots_mixed: bool
    :param input_prompt: The text prompt for the input. Leave blank for no prompt.
                         Default: 'Annotate: '
    :type input_prompt: str
    :param toks_in_output: Whether to put tokens in the output or not. Default: False. If this is
                           True, then slots_mixed must be False
    :type toks_in_output: bool
    :param sentinels: Whether to add T5 sentinels before each token. Overrides toks_in_output and
                      slots_mixed. Default: False
                      See: https://arxiv.org/pdf/2203.08378.pdf
    :type sentinels: bool
    :param inside_format: The slot to use for the inside of a multi-word slot. Options are
                          "slot_name", in which the slot name is repeated, "inside_slot_name",
                          in which "I-" is added to the slot name, or "inside", in which "I" is
                          used on its own.
    :type inside_format: str
    :param outside_label: The word used for non-slotted tokens. Default: Other
    :type outside_label: str

    :return: a list of intents, a list of slot lists
    :rtype: list
    """

    if sentinels:
        # using sentinels is the same as doing slots_mixed and toks_in_output and converting the
        # utterance to a sequence of sentinels
        toks_in_output = True
        slots_mixed = True
        for example in mod_out:
            new_utt, sent_id = [], 0
            for tok in example:
                new_utt.append('<extra_id_' + str(sent_id) + '>')
                sent_id += 1
            example = new_utt

    # Get intents
    if intent_first and use_output_descrip:
        # Note: this assumes that the description is one word
        intents_pred = [x.split()[1] if len(x.split()) > 1 else '' \
                        for x in mod_out]
    elif intent_first:
        intents_pred = [x.split()[0] for x in mod_out]
    else:
        intents_pred = [x.split()[-1] for x in mod_out]

    # Determine Slots. Note: this assumes that the description is one word
    descrip_shift = 0
    if use_output_descrip:
        descrip_shift = 1

    if intent_first:
        # Everthing after the intent
        slot_chunk_pred = [x.split()[(1+2*descrip_shift):] for x in mod_out]
    else:
        # Everything until the intent
        slot_chunk_pred = [x.split()[(descrip_shift):(-1*(descrip_shift+1))] \
                           for x in mod_out]
    if toks_in_output and slots_mixed:
        # Grab every other item
        slots_pred = [x[1::2] for x in slot_chunk_pred]
    elif toks_in_output:
        slots_pred = []
        # Assume equal number of tokens and slots and take second half
        for pred in slot_chunk_pred:
            pred = pred[descrip_shift:]
            mid = len(pred)//2
            slots_pred.append(pred[mid:])
    else:
        slots_pred = slot_chunk_pred

    # Modify for inside format if needed
    for s_idx, slots in enumerate(slots_pred):
        new_slots = []
        for idx, slot in enumerate(slots):
            if idx > 0 and slot != outside_label:
                if inside_format == 'inside_slot_name':
                    if slot.startswith('I-'):
                        new_slots.append(slots[idx-1])
                        continue
                elif inside_format == 'inside':
                    if slot == 'I':
                        new_slots.append(slots[idx-1])
                        continue
            new_slots.append(slot)
        slots_pred[s_idx] = new_slots

    return intents_pred, slots_pred
