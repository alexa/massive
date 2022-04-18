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

import math
import time

from collections import namedtuple
import datasets
import transformers
from transformers.trainer_utils import EvalLoopOutput, speed_metrics

class MASSIVETrainer(transformers.Trainer):
    """
    A Trainer subclass with MASSIVE-specific functionality. Based on:
        https://github.com/huggingface/transformers/blob/master/src/transformers/trainer.py
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix='eval',
                 return_all_outputs=False):
        """
        Adapted from:
        https://github.com/huggingface/transformers/blob/master/src/transformers/trainer.py

        Run evaluation and returns metrics.
        The calling script will be responsible for providing a method to compute metrics, as they
        are task-dependent (pass it to the init :obj:`compute_metrics` argument).
        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (:obj:`Dataset`, `optional`):
                Pass a dataset if you wish to override :obj:`self.eval_dataset`. If it is an
                :obj:`datasets.Dataset`, columns not accepted by the ``model.forward()`` method
                are automatically removed. It must implement the
                :obj:`__len__` method.
            ignore_keys (:obj:`Lst[str]`, `optional`):
                A list of keys in the output of your model (if it is a dictionary) that should be
                ignored when gathering predictions.
            metric_key_prefix (:obj:`str`, `optional`, defaults to :obj:`"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics
                "bleu" will be named "eval_bleu" if the prefix is "eval" (default)

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from
            the predictions. The dictionary also contains the epoch number which comes from the
            training state.
        """

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        # use two prefixes and add step/epoch info
        first_metric_key_prefix = metric_key_prefix
        metrics = {}
        metrics['training_global_step'] = self.state.global_step
        metrics['training_epoch'] = self.state.epoch

        eval_dataset = eval_dataset if eval_dataset else self.eval_dataset

        # Create a list of eval runs based on the strategy
        if self.args.locale_eval_strategy == "all and each":
            locales = sorted(set(eval_dataset['locale']))
            locales.append('all')
        elif self.args.locale_eval_strategy == "all only":
            locales = ['all']
        # add more strategies here
        else:
            raise NotImplementedError('locale_eval_strategy not known')

        # loop through all locales (including "all") and run evaluation
        for locale in locales:

            metric_key_prefix = first_metric_key_prefix + "_" + locale
            if locale == 'all':
                # use whole dataset
                dataset = eval_dataset
            else:
                # filter only to relevant locale
                # This will warn us every time it loads a cached dataset, which is annoying
                # I agree with stas00 here: https://github.com/huggingface/datasets/issues/1948
                # We'll suppress these warnings by temporarily changing the logging level
                lvl = datasets.logging.get_verbosity()
                datasets.logging.set_verbosity(50)
                dataset = eval_dataset.filter(lambda x: x['locale'] == locale)
                datasets.logging.set_verbosity(lvl)

            eval_dataloader = self.get_eval_dataloader(dataset)
            start_time = time.time()

            eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop \
                                             else self.evaluation_loop
            output = eval_loop(
                eval_dataloader,
                description="Evaluation",
                # No point gathering the predictions if there are no metrics, otherwise we defer to
                # self.args.prediction_loss_only
                prediction_loss_only=True if self.compute_metrics is None else None,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )
            metrics.update(output.metrics)

            total_batch_size = self.args.eval_batch_size * self.args.world_size
            metrics.update(
                speed_metrics(
                    metric_key_prefix,
                    start_time,
                    num_samples=output.num_samples,
                    num_steps=math.ceil(output.num_samples / total_batch_size),
                )
            )

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control,
                                                         metrics)

        self._memory_tracker.stop_and_update_metrics(metrics)

        if self.args.locale_eval_strategy == "all and each":
            metrics = self._find_log_highest_lowest_locales(metrics)

        if return_all_outputs:
            return EvalLoopOutput(
                predictions=output.predictions,
                label_ids=output.label_ids,
                metrics=metrics,
                num_samples=output.num_samples
            )

        self.log(metrics)

        return metrics

    def predict(self, test_dataset, ignore_keys=None, metric_key_prefix='test', tokenizer=None):
        """
        Overriding this method for custom test result logging using `evaluate`
        """

        output = self.evaluate(
            test_dataset,
            ignore_keys,
            metric_key_prefix,
            return_all_outputs=True
        )

        PredictionOutput = namedtuple(
            'PredictionOutput',
            ['predictions', 'label_ids', 'metrics', 'ids', 'locales', 'utts', 'tok_utts',
             'subword_aligns']
        )

        subword_aligns=None
        if tokenizer:
            tokenized_inputs = self.tokenizer(
                [item['utt'] for item in test_dataset],
                truncation=True,
                is_split_into_words=True
            )
            subword_aligns = [tokenized_inputs.word_ids(batch_index=i) \
                              for i in range(len(test_dataset))]

        preds =  PredictionOutput(
            predictions=output.predictions,
            label_ids=output.label_ids,
            metrics=output.metrics,
            ids=[x['id'] for x in test_dataset],
            locales=[x['locale'] for x in test_dataset],
            utts=[x['utt'] for x in test_dataset],
            tok_utts=[tokenizer.convert_ids_to_tokens(x) for x in tokenized_inputs.input_ids],
            subword_aligns=subword_aligns
        )

        return preds

    def _find_log_highest_lowest_locales(self, metrics):
        """
        Method to determine the locales with the highest and lowest scores

        :param metrics: A dictionary of the metrics found during evaluation across all locales
        :type metrics: dict
        :return metrics: The updated dictionary of metrics
        :rtype metrics: dict
        """

        highest_val, highest_locale, lowest_val, lowest_locale = {}, {}, {}, {}

        # Iterate through the metrics and get the highest and lowest values and locales
        for k, v in metrics.items():
            pieces = k.split('_')

            if len(pieces) < 3:
                continue

            pre = pieces[0]
            locale = pieces[1]
            metric = '_'.join(pieces[2:])

            if metric in ['loss', 'samples_per_second', 'steps_per_second', 'runtime', 'step']:
                continue

            if 'stderr' in metric:
                continue

            if v > highest_val.get(metric, float('-inf')):
                highest_val[metric] = v
                highest_locale[metric] = locale

            if v < lowest_val.get(metric, float('inf')):
                lowest_val[metric] = v
                lowest_locale[metric] = locale

        # Iterate through the newly found keys and log and save the values
        for metric in highest_locale.keys():
            highest_locale_key = pre + '_highest-locale_' + metric
            highest_locale_val_key = pre + '_highest-locale-val_' + metric
            lowest_locale_key = pre + '_lowest-locale_' + metric
            lowest_locale_val_key = pre + '_lowest-locale-val_' + metric
            all_locale_key = pre + '_all_' + metric
            self.log({
                'training_global_step': self.state.global_step,
                'training_epoch': self.state.epoch,
                'metric': metric,
                'highest_locale': highest_locale[metric],
                'highest_val': highest_val[metric],
                'lowest_locale': lowest_locale[metric],
                'lowest_val': lowest_val[metric],
                all_locale_key: metrics[all_locale_key]
            })
            metrics.update({
                highest_locale_key: highest_locale[metric],
                highest_locale_val_key: highest_val[metric],
                lowest_locale_key: lowest_locale[metric],
                lowest_locale_val_key: lowest_val[metric],
            })

        return metrics

class MASSIVESeq2SeqTrainer(transformers.Seq2SeqTrainer):
    """
    A Seq2SeqTrainer subclass with MASSIVE-specific functionality. Based on:
        https://github.com/huggingface/transformers/blob/master/src/transformers/trainer_seq2seq.py
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix='eval',
                 max_length=None, num_beams=None, return_all_outputs=False):
        """
        Adapted from:
        https://github.com/huggingface/transformers/blob/master/src/transformers/trainer.py

        Run evaluation and returns metrics.
        The calling script will be responsible for providing a method to compute metrics, as they
        are task-dependent (pass it to the init :obj:`compute_metrics` argument).
        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (:obj:`Dataset`, `optional`):
                Pass a dataset if you wish to override :obj:`self.eval_dataset`. If it is an
                :obj:`datasets.Dataset`, columns not accepted by the ``model.forward()`` method
                are automatically removed. It must implement the
                :obj:`__len__` method.
            ignore_keys (:obj:`Lst[str]`, `optional`):
                A list of keys in the output of your model (if it is a dictionary) that should be
                ignored when gathering predictions.
            metric_key_prefix (:obj:`str`, `optional`, defaults to :obj:`"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics
                "bleu" will be named "eval_bleu" if the prefix is "eval" (default)
            max_length (int): The max length for generation. Defaults to args.generation_max_length
            num_beams (int): The number of generation beams. Defaults to args.generation_num_beams

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from
            the predictions. The dictionary also contains the epoch number which comes from the
            training state.
        """

        # for generation
        self._max_length = max_length if max_length is not None else self.args.generation_max_length
        self._num_beams = num_beams if num_beams is not None else self.args.generation_num_beams

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        # use two prefixes and add step/epoch info
        first_metric_key_prefix = metric_key_prefix
        metrics = {}
        metrics['training_global_step'] = self.state.global_step
        metrics['training_epoch'] = self.state.epoch

        eval_dataset = eval_dataset if eval_dataset else self.eval_dataset

        # Create a list of eval runs based on the strategy
        if self.args.locale_eval_strategy == "all and each":
            locales = sorted(set(eval_dataset['locale']))
            locales.append('all')
        elif self.args.locale_eval_strategy == "all only":
            locales = ['all']
        # add more strategies here
        else:
            raise NotImplementedError('locale_eval_strategy not known')

        # loop through all locales (including "all") and run evaluation
        for locale in locales:

            metric_key_prefix = first_metric_key_prefix + "_" + locale
            if locale == 'all':
                # use whole dataset
                dataset = eval_dataset
            else:
                # filter only to relevant locale
                # This will warn us every time it loads a cached dataset, which is annoying
                # I agree with stas00 here: https://github.com/huggingface/datasets/issues/1948
                # We'll suppress these warnings by temporarily changing the logging level
                lvl = datasets.logging.get_verbosity()
                datasets.logging.set_verbosity(50)
                dataset = eval_dataset.filter(lambda x: x['locale'] == locale)
                datasets.logging.set_verbosity(lvl)

            eval_dataloader = self.get_eval_dataloader(dataset)
            start_time = time.time()

            eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop \
                                             else self.evaluation_loop
            output = eval_loop(
                eval_dataloader,
                description="Evaluation",
                # No point gathering the predictions if there are no metrics, otherwise we defer to
                # self.args.prediction_loss_only
                prediction_loss_only=True if self.compute_metrics is None else None,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )
            metrics.update(output.metrics)

            total_batch_size = self.args.eval_batch_size * self.args.world_size
            metrics.update(
                speed_metrics(
                    metric_key_prefix,
                    start_time,
                    num_samples=output.num_samples,
                    num_steps=math.ceil(output.num_samples / total_batch_size),
                )
            )

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control,
                                                         metrics)

        self._memory_tracker.stop_and_update_metrics(metrics)

        if self.args.locale_eval_strategy == "all and each":
            metrics = self._find_log_highest_lowest_locales(metrics)

        if return_all_outputs:
            return EvalLoopOutput(
                predictions=output.predictions,
                label_ids=output.label_ids,
                metrics=metrics,
                num_samples=output.num_samples
            )

        self.log(metrics)

        return metrics

    def predict(self, test_dataset, ignore_keys=None, max_length=None, num_beams=None,
                metric_key_prefix='test', tokenizer=None):
        """
        Overriding this method for custom test result logging using `evaluate`
        """

        output = self.evaluate(
            test_dataset,
            ignore_keys=ignore_keys,
            max_length=max_length,
            num_beams=num_beams,
            metric_key_prefix=metric_key_prefix,
            return_all_outputs=True
        )

        PredictionOutput = namedtuple(
            'PredictionOutput',
            ['predictions', 'label_ids', 'metrics', 'ids', 'locales', 'utts']
        )

        preds =  PredictionOutput(
            predictions=output.predictions,
            label_ids=output.label_ids,
            metrics=output.metrics,
            ids=[x['id'] for x in test_dataset],
            locales=[x['locale'] for x in test_dataset],
            utts=[x['utt'] for x in test_dataset]
        )

        return preds


    def _find_log_highest_lowest_locales(self, metrics):
        """
        Method to determine the locales with the highest and lowest scores

        :param metrics: A dictionary of the metrics found during evaluation across all locales
        :type metrics: dict
        :return metrics: The updated dictionary of metrics
        :rtype metrics: dict
        """

        highest_val, highest_locale, lowest_val, lowest_locale = {}, {}, {}, {}

        # Iterate through the metrics and get the highest and lowest values and locales
        for k, v in metrics.items():
            pieces = k.split('_')

            if len(pieces) < 3:
                continue

            pre = pieces[0]
            locale = pieces[1]
            metric = '_'.join(pieces[2:])

            if metric in ['loss', 'samples_per_second', 'steps_per_second', 'runtime', 'step']:
                continue

            if 'stderr' in metric:
                continue

            if v > highest_val.get(metric, float('-inf')):
                highest_val[metric] = v
                highest_locale[metric] = locale

            if v < lowest_val.get(metric, float('inf')):
                lowest_val[metric] = v
                lowest_locale[metric] = locale

        # Iterate through the newly found keys and log and save the values
        for metric in highest_locale.keys():
            highest_locale_key = pre + '_highest-locale_' + metric
            highest_locale_val_key = pre + '_highest-locale-val_' + metric
            lowest_locale_key = pre + '_lowest-locale_' + metric
            lowest_locale_val_key = pre + '_lowest-locale-val_' + metric
            all_locale_key = pre + '_all_' + metric
            self.log({
                'training_global_step': self.state.global_step,
                'training_epoch': self.state.epoch,
                'metric': metric,
                'highest_locale': highest_locale[metric],
                'highest_val': highest_val[metric],
                'lowest_locale': lowest_locale[metric],
                'lowest_val': lowest_val[metric],
                all_locale_key: metrics[all_locale_key]
            })
            metrics.update({
                highest_locale_key: highest_locale[metric],
                highest_locale_val_key: highest_val[metric],
                lowest_locale_key: lowest_locale[metric],
                lowest_locale_val_key: lowest_val[metric],
            })

        return metrics
