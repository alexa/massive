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

from transformers import TrainingArguments

class MASSIVETrainingArguments(TrainingArguments):
    """
    Child of:
    https://github.com/huggingface/transformers/blob/master/src/transformers/training_args.py

    This class adds more arguments to transformers.TrainingArguments. Additional arg(s):

    local_eval_strategy (str): Whether to evaluate on (`all only`) only all languages mixed together
        or (`all and each`) also to evaluate on each language individually

    """

    def __init__(self, *args, **kwargs):

        # add any MASSIVE-specific keyword arguments and S2S-specific kw arguments here:
        massive_kwargs = [
            'locale_eval_strategy',
            'predict_with_generate',
            'generation_max_length',
            'generation_num_beams'
        ]

        # pop MASSIVE-specific keyword args and set them
        for arg in kwargs.copy():
            if arg in massive_kwargs:
                val = kwargs.pop(arg)
                setattr(self, arg, val)

        super().__init__(*args, **kwargs)
