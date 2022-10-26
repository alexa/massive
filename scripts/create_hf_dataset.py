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
import json
import os

import datasets
from datasets import Dataset

class DatasetCreator:
    """
    This class if for creating four dataset splits, in the Huggingface Datasets Apache Arrow format,
    from the MASSIVE dataset. Each dataset split has the following columns:

        "id",
        "locale",
        "utt",
        "annot_utt",
        "domain",
        "intent_str",
        "intent_num",
        "slots_str",
        "slots_num"

    Methods
    -------
    create_datasets(data_path): Creates the dataset splits using the data_path of the MASSIVE set
    add_numeric_labels(): Create integer versions of intents and slot for modeling
    investigate_datasets(): Prints out the seventh example from each dataset split as gut check
    save_label_dicts(prefix): Saves the mappings to the integer versions of the labels
    save_datasets(out_prefix): Saves the datasets to out_prefix

    """

    def __init__(self):
        self.train = None
        self.dev = None
        self.test = None
        self.hidden_eval = None
        self.slot_dict = None
        self.intent_dict = None

    def create_datasets(self, data_paths, char_split_locales=None):
        """
        Loads the datasets, parses, and appends to the HF Dataset

        :param data_path: The path(s) to the MASSIVE dataset
        :type data_path: str or list
        :param char_split_locales: Locales that should be split by character
        :type char_split_locales: list[str]
        """

        char_split_locales = ['ja-JP', 'zh-CN', 'zh-TW'] if not char_split_locales \
                                                         else char_split_locales

        # Find locales based on the names of the files
        files = []
        data_paths = [data_paths] if type(data_paths) == str else data_paths
        for path in data_paths:
            flist = [os.path.join(path,f) \
                     for f in os.listdir(path) \
                     if os.path.isfile(os.path.join(path,f))]
            files = files + flist

        for file in files:
            print(f'Reading in data from {file}')
            massive_raw = []
            # Read in the json per line
            with open(file, 'r') as f:
                for line in f:
                    massive_raw.append(json.loads(line))

            # Parse each line
            train, dev, test, hid = self._build_in_mem_dicts(
                massive_raw,
                char_split_locales
            )

            # Either create the split or concatentate to an existing one
            if self.train is None and train['id']:
                self.train = Dataset.from_dict(train)
            elif train['id']:
                self.train = datasets.concatenate_datasets([self.train,
                                                            Dataset.from_dict(train)])
            if self.dev is None and dev['id']:
                self.dev = Dataset.from_dict(dev)
            elif dev['id']:
                self.dev = datasets.concatenate_datasets([self.dev, Dataset.from_dict(dev)])
            if self.test is None and test['id']:
                self.test = Dataset.from_dict(test)
            elif test['id']:
                self.test = datasets.concatenate_datasets([self.test, Dataset.from_dict(test)])
            if self.hidden_eval is None and hid['id']:
                self.hidden_eval = Dataset.from_dict(hid)
            elif hid['id']:
                self.hidden_eval = datasets.concatenate_datasets([self.hidden_eval,
                                                                 Dataset.from_dict(hid)])

    @staticmethod
    def _build_in_mem_dicts(massive_data, char_split=None):
        """ Parse the JSON into a flat key/value format """

        char_split = ['ja-JP', 'zh-CN', 'zh-TW'] if not char_split else char_split

        cols = ['id', 'locale', 'domain', 'intent_str', 'annot_utt', 'utt',
                'slots_str']
        train, dev = {k: [] for k in cols}, {k: [] for k in cols}
        test, hid_eval = {k: [] for k in cols}, {k: [] for k in cols}

        for row in massive_data:
            eyed, locale, split, utt = row['id'], row['locale'], row['partition'], row['utt']
            domain = row['scenario'] if 'scenario' in row else ''
            intent = row['intent'] if 'intent' in row else ''
            annot_utt = row['annot_utt'] if 'annot_utt' in row else ''

            # Split these languages by character
            if locale in char_split:
                tokens, labels = [], []
                label = 'Other'
                skip_colon = False
                if annot_utt:
                    for chunk in annot_utt.split():
                        if chunk.startswith('['):
                            label = chunk.lstrip('[')
                            skip_colon = True
                            continue
                        if chunk == ':' and skip_colon is True:
                            skip_colon = False
                            continue
                        # keep latin chars together in cases of code switching
                        if isascii(chunk):
                            tokens.append(chunk.strip().rstrip(']'))
                            labels.append(label)
                        else:
                            chars = list(chunk.strip())
                            for char in chars:
                                if char == ']':
                                    label = 'Other'
                                else:
                                    tokens.append(char)
                                    labels.append(label)
                # if no annot_utt, then make assumption latin words are space sep already
                else:
                    for chunk in utt.split():
                        if isascii(chunk):
                            tokens.append(chunk.strip())
                        else:
                            chars = list(chunk.strip())
                            for char in chars:
                                tokens.append(char)

            else:
                # Create the tokens and labels by working left to right of annotated utt
                tokens = utt.split()
                labels = []
                label = 'Other'
                split_annot_utt = annot_utt.split()
                idx = 0
                while idx < len(split_annot_utt):
                    if split_annot_utt[idx].startswith('['):
                        label = split_annot_utt[idx].lstrip('[')
                        idx += 2
                    elif split_annot_utt[idx].endswith(']'):
                        labels.append(label)
                        label = 'Other'
                        idx += 1
                    else:
                        labels.append(label)
                        idx += 1

            if len(tokens) != len(labels) and labels:
                raise ValueError(f"Len of tokens, {tokens}, doesnt match len of labels, {labels}, "
                                 f"for id {eyed} and annot utt: {annot_utt}")

            # Pick the dictionary corresponding to this split
            if split == 'train':
                dict_view = train
            elif split == 'dev':
                dict_view = dev
            elif split == 'test':
                dict_view = test
            elif split == 'MMNLU-22':
                dict_view = hid_eval
            else:
                raise ValueError(f"split {split} is not valid")

            # add the values for the keys
            dict_view['id'].append(eyed)
            dict_view['locale'].append(locale)
            dict_view['domain'].append(domain)
            dict_view['intent_str'].append(intent)
            dict_view['annot_utt'].append(annot_utt)
            dict_view['utt'].append(tokens)
            dict_view['slots_str'].append(labels)

        return train, dev, test, hid_eval

    def investigate_datasets(self):
        """ Prints out the seventh example from each split """
        for dataset in [self.train, self.dev, self.test, self.hidden_eval]:
            if dataset:
                print(f"dataset: {dataset}")
                print(f"row 7: {dataset[7]}")

    def add_numeric_labels(self):
        """ Creates integer version of the intent and slot labels, which is useful for modeling """

        if not self.intent_dict:
            # Get the unique intents and create a mapping dict
            unique_intents = set([])
            for split in [self.train, self.dev, self.test, self.hidden_eval]:
                if split:
                    unique_intents.update([i for i in split['intent_str']])
            self.intent_dict = {k: v for v, k in enumerate(unique_intents)}
            print('The following intent labels were detected across all partitions: ',
                  self.intent_dict)
        else:
            # swap key and val
            self.intent_dict = {v: int(k) for k, v in self.intent_dict.items()}

        if not self.slot_dict:
            # Get the unique slots and create a mapping dict
            unique_slots = set()
            for split in [self.train, self.dev, self.test, self.hidden_eval]:
                if split:
                    for ex_slots in split:
                        unique_slots.update(ex_slots['slots_str'])
            self.slot_dict = {k: v for v, k in enumerate(unique_slots)}
            print('The following slot labels were detected across all partitions: ', self.slot_dict)
        else:
            # swap key and val
            self.slot_dict = {v: int(k) for k, v in self.slot_dict.items()}

        # Define a function for creating numeric labels from existing text labels
        def create_numeric_labels(example):
            example['slots_num'] = [self.slot_dict[x] for x in example['slots_str']]
            example['intent_num'] = self.intent_dict[example['intent_str']]
            return example

        # Create the new numeric fields in the dataset with the map method
        print('Adding numeric intent and slot labels to the datasets')
        self.train = self.train.map(create_numeric_labels) if self.train else None
        self.dev = self.dev.map(create_numeric_labels) if self.dev else None
        self.test = self.test.map(create_numeric_labels) if self.test else None
        if self.hidden_eval is not None and self.hidden_eval[0]['intent_str']:
            self.hidden_eval = self.hidden_eval.map(create_numeric_labels)

    def save_label_dicts(self, output_prefix):
        """
        Save the dictionaries mapping numeric labels to text-based labels

        :param output_prefix: The location and file prefix for saving the dictionaries
        :type output_prefix: str
        """

        with open(output_prefix+'.intents', "w") as i, open(output_prefix+'.slots', "w") as s:
            # swap the keys and vals to use the index as key and slot as val
            json.dump({v: k for k, v in self.intent_dict.items()}, i)
            json.dump({v: k for k, v in self.slot_dict.items()}, s)

    def save_datasets(self, output_prefix):
        """
        Save the dataset splits

        :param output_prefix: The location and file prefix for saving the dataset splits
        :type output_prefix: str
        """
        for (ds, suf) in [
            (self.train, '.train'),
            (self.dev, '.dev'),
            (self.test, '.test'),
            (self.hidden_eval, '.mmnlu22')
        ]:
            if ds:
                ds.save_to_disk(output_prefix+suf)


def isascii(s):
    try:
        return s.isascii()
    except AttributeError:
        return all([ord(c) < 128 for c in s])


def main():
    parser = argparse.ArgumentParser(description="Create huggingface datasets from MASSIVE")
    parser.add_argument('-d', '--massive-data-paths', nargs='+', help='path(s) to MASSIVE dataset')
    parser.add_argument('-o', '--out-prefix', help='output path and prefix for datasets')
    parser.add_argument('--intent-map', nargs='?', default={},
                        help='optional existing intent numeric map', required=False)
    parser.add_argument('--slot-map', nargs='?', default={},
                        help='optional existing slot numeric map', required=False)
    args = parser.parse_args()

    if args.intent_map:
        with open(args.intent_map, 'r') as f:
            intent_dict = json.load(f)
    else:
        intent_dict = None

    if args.slot_map:
        with open(args.slot_map, 'r') as f:
            slot_dict = json.load(f)
    else:
        slot_dict = None

    ds_creator = DatasetCreator()
    ds_creator.create_datasets(args.massive_data_paths)
    ds_creator.intent_dict = intent_dict
    ds_creator.slot_dict = slot_dict
    ds_creator.add_numeric_labels()
    ds_creator.investigate_datasets()
    ds_creator.save_datasets(args.out_prefix)
    ds_creator.save_label_dicts(args.out_prefix)

if __name__ == "__main__":
    main()
