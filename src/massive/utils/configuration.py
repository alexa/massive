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

from __future__ import print_function, unicode_literals
from ruamel.yaml import YAML

class Configuration:
    """ Class to create the config object """

    def __init__(self, conf_dict):
        self.kv = conf_dict

    def get(self, key, required=False, default=None, d=None, _original_key=None):
        if d is None:
            d = self.kv
            _original_key = key
        key_comps = key.split('.')
        try:
            # Extending the capability to get list indices in the configuration
            # Example: config.get("model_factory.model.embedding_block.embedding_layers.1.export!")
            if key_comps[0].isdigit():
                key_comps[0] = int(key_comps[0])
            v = d[key_comps[0]]
            if len(key_comps) > 1:
                return self.get(
                    key=".".join(key_comps[1:]), d=v, required=required, default=default,
                                 _original_key=_original_key)
            return v
        except KeyError:
            # Find the nearest key in the list of keys.
            if required:
                raise KeyError()
            return default

    def override(self, key, value, d=None, _original_key=None, force=False):
        def _convert_type(x, y):
            return type(x)(y)

        if d is None:
            d = self.kv
            _original_key = key

        key_comps = key.split('.')
        if isinstance(d, list):
            k = int(key_comps[0])
        else:
            k = key_comps[0]

        if len(key_comps) > 1:
            try:
                if isinstance(d, list):
                    v = d[k]
                else:
                    v = d.setdefault(k, {})
                return self.override(
                    key=".".join(key_comps[1:]), d=v, value=value, _original_key=_original_key,
                                 force=force)
            except KeyError:
                raise KeyError("key '{}' does not exist in configuration".format(_original_key))
        else:
            if not force:
                assert type(d) is dict and k in d, \
                    "Trying to override key '{}' that does not exist in configuration".format(
                        _original_key)
                d[key_comps[0]] = _convert_type(d[k], value)
            else:
                d[key_comps[0]] = value

    def get_as_dict(self):
        return self.kv


def read_conf(file_name, overrides=None):
    """
    Parse a configuration file.

    :param file_name: path to configuration file
    :param overrides: list or dict of override parameters in "path.to.key:value" value format, e.g.
        "training.optimizer.parameters.learning_rate:0.001"

    :return: Configuration object
    """

    if overrides is None:
        overrides = []

    override_dict = _parse_overrides_to_dict(overrides)

    # load configuration from file. The pyyaml's loader works fine with
    # both YAML and JSON files, as the former is a superset of the latter.
    with open(file_name) as f:
        yaml = YAML(typ="safe")
        configuration_dict = yaml.load(f.read())

        # take care of overrides later
    return Configuration(configuration_dict)

def _parse_overrides_to_dict(overrides):
    """
    Parse the list of override in the key.to.override:value format
    to a nested dictionary that can be merged with the parsed output.

    :param overrides: list of strings containing overrides provided via command line.
    :return: a dictionary containing any (nested) key to override.
    :raise RuntimeError when overrides is neither list nor dict
    """
    if isinstance(overrides, list) or isinstance(overrides, dict):
        override_dict = {}
        for override in overrides:
            if isinstance(overrides, list):
                ks, v = override.split(':', 1)
            else:
                ks, v = override, overrides.get(override)
            ks = ks.split('.')
            d = override_dict
            while True:
                if len(ks) == 1:
                    d[ks[0]] = YAML.load(v)
                    break
                d = d.setdefault(ks[0], {})
                ks = ks[1:]
        return override_dict
    else:
        raise RuntimeError("override is not a list or dictionary! \n {}".format(overrides))
