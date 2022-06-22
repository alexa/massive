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

import argparse
import datetime
import logging
import os
import pprint
import sys
import time

import datasets
from massive import (
    MASSIVETrainer,
    MASSIVESeq2SeqTrainer,
    MASSIVETrainingArguments,
    create_compute_metrics,
    init_model,
    init_tokenizer,
    output_predictions,
    prepare_collator,
    prepare_test_dataset,
    read_conf,
)
from ruamel.yaml import YAML
import torch.distributed as dist
import transformers

logger = logging.getLogger('massive_logger')

def main():
    """ Run Testing/Inference """
    # parse the args
    parser = argparse.ArgumentParser(description="Testing on the MASSIVE dataset")
    parser.add_argument('-c', '--config', help='path to run configuration yaml')
    parser.add_argument('--local_rank', help='local rank of this process. Optional')
    args = parser.parse_args()

    # create the massive.Configuration master config object
    conf = read_conf(args.config)
    trainer_args = MASSIVETrainingArguments(**conf.get('test.trainer_args'))
    if args.local_rank:
        trainer_args.local_rank = int(args.local_rank)
    elif os.getenv('LOCAL_RANK'):
        trainer_args.local_rank = int(os.environ['LOCAL_RANK'])

    # Setup logging
    logging.basicConfig(
        #format="[%(levelname)s|%(name)s] %(asctime)s >> %(message)s",
        format="[%(levelname)s] %(asctime)s >> %(message)s",
        #datefmt="%Y%m%d %H:%M",
        datefmt="%H:%M",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = trainer_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    logger.info(f"Starting the run at {datetime.datetime.now()}")
    yaml = YAML(typ='safe')
    logger.info(f"Using the following config: {yaml.load(open(args.config, 'r'))}")

    # Check for right setup
    if not conf.get('test.predictions_file'):
        logger.warning("Outputs will not be saved because no test.predictions_file was given")
    if conf.get('test.predictions_file') and \
       (conf.get('test.trainer_args.locale_eval_strategy') != 'all only'):
        raise NotImplementedError("You must use 'all only' as the locale_eval_strategy if you"
                                  " include a predictions file")


    # Get all inputs to the trainer
    tokenizer = init_tokenizer(conf)
    test_ds, intents, slots = prepare_test_dataset(conf, tokenizer)
    model = init_model(conf, intents, slots)
    collator = prepare_collator(conf, tokenizer, model)
    slots_ignore = conf.get('test.slot_labels_ignore', default=[])
    metrics = conf.get('test.eval_metrics', default='all')
    compute_metrics = create_compute_metrics(intents, slots, conf, tokenizer, slots_ignore,
                                             metrics)

    # Get the right trainer
    trainer_cls = MASSIVESeq2SeqTrainer \
                  if conf.get('test.trainer') == 'massive s2s' \
                  else MASSIVETrainer

    trainer = trainer_cls(
        model = model,
        args = trainer_args,
        data_collator=collator,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer
    )

    trainer.remove_callback(transformers.integrations.TensorBoardCallback)

    outputs = trainer.predict(test_ds, tokenizer=tokenizer)

    rank = dist.get_rank() if dist.is_initialized() else 0

    if rank == 0:
        time.sleep(3)
        logger.info('CAUTION: Test with validation engine metrics are for reference only. For '
                    '"official" metrics include a test.predictions_file in the config and use the '
                    'eval.ai leaderboard')
        logger.info(f'Validation engine metrics computer readable: {outputs.metrics}')
        logger.info('Validation engine metrics pretty printed: ')
        pp = pprint.PrettyPrinter(indent=2)
        pp.pprint(outputs.metrics)

        save_to_file = True if conf.get('test.predictions_file') else False

        output_predictions(outputs, intents, slots, conf, tokenizer,
                           remove_slots=slots_ignore, save_to_file=save_to_file)

if __name__ == "__main__":
    main()
