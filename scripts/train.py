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
import sys

import datasets
from massive import (
    MASSIVETrainer,
    MASSIVESeq2SeqTrainer,
    MASSIVETrainingArguments,
    create_compute_metrics,
    init_model,
    init_tokenizer,
    prepare_collator,
    prepare_train_dev_datasets,
    read_conf,
)
import transformers
from ruamel.yaml import YAML

logger = logging.getLogger('massive_logger')

def main():
    """ Run Training """
    # parse the args
    parser = argparse.ArgumentParser(description="Training on the MASSIVE dataset")
    parser.add_argument('-c', '--config', help='path to run configuration yaml')
    parser.add_argument('--local_rank', help='local rank of this process. Optional')
    args = parser.parse_args()

    # create the massive.Configuration master config object
    conf = read_conf(args.config)
    trainer_args = MASSIVETrainingArguments(**conf.get('train_val.trainer_args'))
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

    # Get all inputs to the trainer
    tokenizer = init_tokenizer(conf)
    train_ds, dev_ds, intents, slots = prepare_train_dev_datasets(conf, tokenizer)
    model = init_model(conf, intents, slots)
    collator = prepare_collator(conf, tokenizer, model)
    slots_ignore = conf.get('train_val.slot_labels_ignore', default=[])
    metrics = conf.get('train_val.eval_metrics', default='all')
    compute_metrics = create_compute_metrics(intents, slots, conf, tokenizer, slots_ignore,
                                             metrics)


    # Get the right trainer
    trainer_cls = MASSIVESeq2SeqTrainer \
                  if conf.get('train_val.trainer') == 'massive s2s' \
                  else MASSIVETrainer

    trainer = trainer_cls(
        model = model,
        args = trainer_args,
        train_dataset = train_ds,
        eval_dataset = dev_ds,
        data_collator=collator,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer
    )

    trainer.train()

if __name__ == "__main__":
    main()
