# MASSIVE

## News

* 14 Jun 2022: We updated the evaluation code to fix a bug identified by @yichaopku ([Issue 13](https://github.com/alexa/massive/issues/13), [PR 14](https://github.com/alexa/massive/pull/14)). Please pull commit 0552cdd or later to use the remedied evaluation code. The baseline results on the [leaderboard](https://eval.ai/web/challenges/challenge-page/1697/overview) have been updated, and the updated preprint paper will be posted to arXiv early next week.
* 20 Apr 2022: Launch and release of the MASSIVE dataset, this repo, the MASSIVE paper, the leaderboard, and the Massively Multilingual NLU 2022 workshop and competition.

## Quick Links

* [MASSIVE paper](https://arxiv.org/abs/2204.08582)
* [MASSIVE Leaderboard and Massively Multilingual NLU 2022 Competition](https://eval.ai/web/challenges/challenge-page/1697/overview)
* [Massively Multilingual NLU 2022 Workshop](https://mmnlu-22.github.io/)
* [MASSIVE Blog Post](https://www.amazon.science/blog/amazon-releases-51-language-dataset-for-language-understanding)

## Introduction

MASSIVE is a parallel dataset of > 1M utterances across 51 languages with annotations for the Natural Language Understanding tasks of intent prediction and slot annotation. Utterances span 60 intents and include 55 slot types. MASSIVE was created by localizing the [SLURP](https://github.com/pswietojanski/slurp) dataset, composed of general Intelligent Voice Assistant single-shot interactions.

## Accessing and Processing the Data

The dataset can be downloaded [here](https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.0.tar.gz).

```
$ curl https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.0.tar.gz --output amazon-massive-dataset-1.0.tar.gz
$ tar -xzvf amazon-massive-dataset-1.0.tar.gz
$ tree 1.0
1.0
├── LICENSE
└── data
    ├── af-ZA.jsonl
    ├── am-ET.jsonl
    ├── ar-SA.jsonl
    ...
```

The dataset is organized into files of JSON lines. Each locale (according to ISO-639-1 and ISO-3166 conventions) has its own file containing all dataset partitions. An example JSON line for de-DE has the following:

```
{
  "id": "0",
  "locale": "de-DE",
  "partition": "test",
  "scenario": "alarm",
  "intent": "alarm_set",
  "utt": "weck mich diese woche um fünf uhr morgens auf",
  "annot_utt": "weck mich [date : diese woche] um [time : fünf uhr morgens] auf",
  "worker_id": "8",
  "slot_method": [
    {
      "slot": "time",
      "method": "translation"
    },
    {
      "slot": "date",
      "method": "translation"
    }
  ],
  "judgments": [
    {
      "worker_id": "32",
      "intent_score": 1,
      "slots_score": 0,
      "grammar_score": 4,
      "spelling_score": 2,
      "language_identification": "target"
    },
    {
      "worker_id": "8",
      "intent_score": 1,
      "slots_score": 1,
      "grammar_score": 4,
      "spelling_score": 2,
      "language_identification": "target"
    },
    {
      "worker_id": "28",
      "intent_score": 1,
      "slots_score": 1,
      "grammar_score": 4,
      "spelling_score": 2,
      "language_identification": "target"
    }
  ]
}
```

`id`: maps to the original ID in the [SLURP](https://github.com/pswietojanski/slurp) collection. Mapping back to the SLURP en-US utterance, this utterance served as the basis for this localization.

`locale`: is the language and country code accoring to ISO-639-1 and ISO-3166.

`partition`: is either `train`, `dev`, or `test`, according to the original split in [SLURP](https://github.com/pswietojanski/slurp).

`scenario`: is the general domain, aka "scenario" in SLURP terminology, of an utterance

`intent`: is the specific intent of an utterance within a domain formatted as `{scenario}_{intent}`

`utt`: the raw utterance text without annotations

`annot_utt`: the text from `utt` with slot annotations formatted as `[{label} : {entity}]`

`worker_id`: The obfuscated worker ID from MTurk of the worker completing the localization of the utterance. Worker IDs are specific to a locale and do *not* map across locales.

`slot_method`: for each slot in the utterance, whether that slot was a `translation` (i.e., same expression just in the target language), `localization` (i.e., not the same expression but a different expression was chosen more suitable to the phrase in that locale), or `unchanged` (i.e., the original en-US slot value was copied over without modification).

`judgments`: Each judgment collected for the localized utterance has 6 keys. `worker_id` is the obfuscated worker ID from MTurk of the worker completing the judgment. Worker IDs are specific to a locale and do *not* map across locales, but *are* consistent across the localization tasks and the judgment tasks, e.g., judgment worker ID 32 in the example above may appear as the localization worker ID for the localization of a different de-DE utterance, in which case it would be the same worker.

```
intent_score : "Does the sentence match the intent?"
  0: No
  1: Yes
  2: It is a reasonable interpretation of the goal

slots_score : "Do all these terms match the categories in square brackets?"
  0: No
  1: Yes
  2: There are no words in square brackets (utterance without a slot)

grammar_score : "Read the sentence out loud. Ignore any spelling, punctuation, or capitalization errors. Does it sound natural?"
  0: Completely unnatural (nonsensical, cannot be understood at all)
  1: Severe errors (the meaning cannot be understood and doesn't sound natural in your language)
  2: Some errors (the meaning can be understood but it doesn't sound natural in your language)
  3: Good enough (easily understood and sounds almost natural in your language)
  4: Perfect (sounds natural in your language)

spelling_score : "Are all words spelled correctly? Ignore any spelling variances that may be due to differences in dialect. Missing spaces should be marked as a spelling error."
  0: There are more than 2 spelling errors
  1: There are 1-2 spelling errors
  2: All words are spelled correctly

language_identification : "The following sentence contains words in the following languages (check all that apply)"
  1: target
  2: english
  3: other
  4: target & english
  5: target & other
  6: english & other
  7: target & english & other
```

Note that the en-US JSON lines will not have the `slot_method` or `judgment` keys, as there was no localization performed. The `worker_id` key in the en-US file corresponds to the worker ID from [SLURP](https://github.com/pswietojanski/slurp).

```
{
  "id": "0",
  "locale": "en-US",
  "partition": "test",
  "scenario": "alarm",
  "intent": "alarm_set",
  "utt": "wake me up at five am this week",
  "annot_utt": "wake me up at [time : five am] [date : this week]",
  "worker_id": "1"
}
```

## Preparing the Data in `datasets` format (Apache Arrow)

The data can be prepared in the `datasets` Apache Arrow format using our script:

```
python scripts/create_hf_dataset.py -d /path/to/jsonl/files -o /output/path/and/prefix
```

If you already have number-to-intent and number-to-slot mappings, those can be used when creating the `datasets`-style dataset:

```
python scripts/create_hf_dataset.py \
    -d /path/to/jsonl/files \
    -o /output/path/and/prefix \
    --intent-map /path/to/intentmap \
    --slot-map /path/to/slotmap
```

## Training an Encoder Model

We have included intent classification and slot-filling models based on the pretrained XLM-R Base or mT5 encoders coupled with JointBERT-style classification heads. Training can be conducted using the `Trainer` from `transformers`. 

We have provided some helper functions in `massive.utils.training_utils`, described below:

* `create_compute_metrics` creates the `compute_metrics` function, which is used to calculate evaluation metrics.
* `init_model` is used to initialize one of our provided models.
* `init_tokeinzer` initializes one of the pretrained tokenizers.
* `prepare_collator` prepares a collator with user-specified max length and padding strategy.
* `prepare_train_dev_datasets`, which loads the datasets prepared as described above.
* `output_predictions`, which outputs the final predictions when running test.

Training is configured in a yaml file. Examples are given in `examples/`. A given yaml file fully describes its respective experiment.

Once an experiment configuration file is created, training can be performed using our provided training script. We also have provided a conda environment configuration file with the necessary dependencies that you may choose to use.

```
conda env create -f conda_env.yml
conda activate massive
```

Set the PYTHONPATH if needed:
```
export PYTHONPATH=${PYTHONPATH}:/PATH/TO/massive/src/
```

Then run training:
```
scripts/train.py -c YOUR/CONFIG/FILE.yml
```

Distributed training can be run using `torch.distributed.launch`. For example:

```
python -m torch.distributed.launch --nproc_per_node=8 scripts/train.py -c YOUR/CONFIG/FILE.yml
```

## Seq2Seq Model Training

Sequence-to-sequence (Seq2Seq) model training is performed using the MASSIVESeq2SeqTrainer class. This class inherits from `Seq2SeqTrainer` from `transformers`. The primary difference with this class is that autoregressive generation is performed during validation, which is turned on using the `predict_with_generate` training argument. Seq2Seq models use teacher forcing during training.

For text-to-text modeling, we have included the following functions in `massive.utils.training_utils`:

* `convert_input_to_t2t`
* `convert_intents_slots_to_t2t`
* `convert_t2t_batch_to_intents_slots`

For example, mT5 Base can be trained on an 8-GPU instance as follows:

```
python -m torch.distributed.launch --nproc_per_node=8 scripts/train.py -c examples/mt5_base_t2t_20220411.yml 2>&1 | tee /PATH/TO/LOG/FILE
```

## Performing Inference on the Test Set

Test inference requires a `test` block in the configuration. See `examples/xlmr_base_test_20220411.yml` for an example. Test inference, including evaluation and output of all predictions, can be executed using the `scripts/test.py` script. For example:

```
python -m torch.distributed.launch --nproc_per_node=8 scripts/test.py -c examples/xlmr_base_test_20220411.yml 2>&1 | tee /PATH/TO/LOG/FILE
```

Be sure to include a `test.predictions_file` in the config to output the predictions.

For official test results, please upload your predictions to the eval.ai leaderboard.

## Hyperparameter Tuning

Hyperparameter tuning can be performed using the `Trainer` from `transformers`. Similarly to training, we combine all configurations into a single yaml file. An example is given here: `example/xlmr_base_hptuning_20220411.yml`.

Once a configuration file has been made, the hyperparameter tuning run can be initiated using our provided `scripts/run_hpo.py` script. Relative to `train.py`, this script uses an additional function called `prepare_hp_search_args`, which converts the hyperparameter search space provided in the configuration into an instantiated `ray` search space.

## Licenses

See `LICENSE.txt`, `NOTICE.md`, and `THIRD-PARTY.md`.

## Citation

We ask that you cite both our [MASSIVE paper](https://arxiv.org/abs/2204.08582) and the [paper for SLURP](https://aclanthology.org/2020.emnlp-main.588/), given that MASSIVE used English data from SLURP as seed data.

MASSIVE paper:
```
@misc{fitzgerald2022massive,
      title={MASSIVE: A 1M-Example Multilingual Natural Language Understanding Dataset with 51 Typologically-Diverse Languages}, 
      author={Jack FitzGerald and Christopher Hench and Charith Peris and Scott Mackie and Kay Rottmann and Ana Sanchez and Aaron Nash and Liam Urbach and Vishesh Kakarala and Richa Singh and Swetha Ranganath and Laurie Crist and Misha Britan and Wouter Leeuwis and Gokhan Tur and Prem Natarajan},
      year={2022},
      eprint={2204.08582},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

SLURP paper:
```
@inproceedings{bastianelli-etal-2020-slurp,
    title = "{SLURP}: A Spoken Language Understanding Resource Package",
    author = "Bastianelli, Emanuele  and
      Vanzo, Andrea  and
      Swietojanski, Pawel  and
      Rieser, Verena",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2020.emnlp-main.588",
    doi = "10.18653/v1/2020.emnlp-main.588",
    pages = "7252--7262",
    abstract = "Spoken Language Understanding infers semantic meaning directly from audio data, and thus promises to reduce error propagation and misunderstandings in end-user applications. However, publicly available SLU resources are limited. In this paper, we release SLURP, a new SLU package containing the following: (1) A new challenging dataset in English spanning 18 domains, which is substantially bigger and linguistically more diverse than existing datasets; (2) Competitive baselines based on state-of-the-art NLU and ASR systems; (3) A new transparent metric for entity labelling which enables a detailed error analysis for identifying potential areas of improvement. SLURP is available at https://github.com/pswietojanski/slurp."
}
```
