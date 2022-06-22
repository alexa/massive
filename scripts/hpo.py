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
    prepare_hp_search_args,
    prepare_train_dev_datasets,
    read_conf
)
from ruamel.yaml import YAML
import transformers

logger = logging.getLogger(__name__)

def main():
    """ Run hyperparameter tuning """
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
    collator = prepare_collator(conf, tokenizer)
    model_init_fn = init_model(conf, intents, slots, return_hpo_fn=True)
    compute_metrics = create_compute_metrics(conf, intents, slots, tokenizer)

    # Get the right trainer
    trainer_cls = MASSIVESeq2SeqTrainer \
                  if conf.get('train_val.trainer') == 'massive s2s' \
                  else MASSIVETrainer

    # Instantiate trainer
    trainer = trainer_cls(
        args = trainer_args,
        train_dataset = train_ds,
        eval_dataset = dev_ds,
        data_collator=collator,
        model_init=model_init_fn,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer
    )

    # Prepare and then run hyperparameter tuning
    hp_args = prepare_hp_search_args(conf)
    best_trial = trainer.hyperparameter_search(**hp_args)
    logger.info("The best Trial:")
    logger.info(best_trial)

if __name__ == "__main__":
    main()
