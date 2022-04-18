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

A collection of tools that can be used by hyperparameter optimization scripts
"""

import logging
import os
from ray import tune

logger = logging.getLogger('massive_logger')
if os.getenv('GLOBAL_RANK', 0) != 0:
    logger.setLevel(logging.WARNING)

def prepare_hp_search_args(conf):
    output = conf.get('hpo_args')
    # hp_space can either be a Trial object or a function returning a dict. This is latter.
    if 'hp_space' in output:
        hp_space_fn_out = _parse_mutations(output['hp_space'])
        output['hp_space'] = lambda x: hp_space_fn_out

    # The scheduler can have hyperparam_mutations of the same format as the hp_space dict
    if 'scheduler' in output:
        if 'hyperparam_mutations' in output['scheduler']:
            output['scheduler']['hyperparam_mutations'] = _parse_mutations(
                output['scheduler']['hyperparam_mutations'])
        sched_cls = output['scheduler'].pop('type')
        output['scheduler'] = getattr(tune.schedulers, sched_cls)(**output['scheduler'])

    # Pull the search algorithm
    if 'search_alg' in output:
        salg_name = output['search_alg']['type']
        salg_args = output['search_alg'].get('args')
        output['search_alg'] = tune.suggest.create_searcher(salg_name, **(salg_args or {}))

    # Trainer needs this `compute_objective` in order to pull out the right eval metric
    if 'metric' in output:
        def get_eval_metric(metrics):
            return metrics[output['metric']]
        output['compute_objective'] = get_eval_metric

    return output

def _parse_mutations(spec):
    spec = [spec] if type(spec) == dict else spec
    output = {}
    mut_types_tup = ['uniform', 'quniform', 'loguniform', 'qloguniform', 'randint',
                         'lograndint', 'qrandint', 'qlograndint', 'randn', 'qrandn']

    # Converts the config entries into sampling objects from ray.tune
    for item in spec:
        hp, mut_type, args = item['hp'], item['type'], item['args']
        if mut_type in mut_types_tup:
            output[hp] = getattr(tune, mut_type)(*tuple(args))
        elif mut_type == 'choice':
            output[hp] = getattr(tune, mut_type)(args)
        elif mut_type == 'list':
            output[hp] = args
        elif mut_type == 'sample_from':
            # This allows for a callable custom function
            raise NotImplementedError('mutation type sample_from is not implemented')
        else:
            raise NotImplementedError('mutation type is not implemented')

    return output
