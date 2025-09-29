import argparse
from pydoc import locate
from typing import List, Union, get_origin, get_args, get_type_hints

import docstring_parser
import numpy as np
import json


def update_conf_with_parsed_args(conf, pargs):
    """
    Updates the conf object in-place with the updated parameters from parsed arguments

    Parameters
    ----------
    config: BaseConfig
        Config to be updated
    parsed: argparse
        Parsed arguments to update the config with
    """
    for arg in vars(pargs):
        val = getattr(pargs, arg)
        if isinstance(val, str):
            val = strCheck(val)

        setattr(conf, arg, val)


def get_doc_from_config(config):
    """Read the docstring of the config class and return the parameters description in a dict

    Parameters
    ----------
    config: BaseConfig
        The config class to be parsed

    Return
    ------
    config_params: dict
        Information about class variables as described in the docstring
    """
    docstring = docstring_parser.parse(config.__doc__, style=docstring_parser.common.DocstringStyle.NUMPYDOC)
    config_params = {i.arg_name: i.__dict__ for i in docstring.params}
    return config_params


def variable_type_same_as_docstring_type(t1, t2_str):
    """[summary]

    Parameters
    ----------
    t1 : type
        The type of the variable as determined by get_type_hint
    t2_str : str
        string representation of the type as mentioned in the class docstring

    Returns
    -------
    bool
        Whether the variable type is same as the one mentioned in docstring
    """
    t2_str = t2_str.strip()
    # TODO: Fix the list type checking and maybe handle dicts separately
    if (get_origin(t1) is List or get_origin(t1) is list) and 'list' in t2_str.lower():
        return True
    if get_origin(t1) is not Union:
        return t1 == locate(t2_str)
    else:  # Case of Union
        nt = type(None)
        for ar in get_args(t1):
            if ar != nt and ar != locate(t2_str):
                return False
        return True


def get_all_variables_type(conf, conf_doc, compare_type_with_docs):
    # Use type_hints to get the assigned types (which maybe be different than the type of value at that instance)
    # E.g., get_type_hints(processed_dir) -> typing.Union[str, NoneType]
    # type(processed_dir) -> NoneType
    hinted_types = get_type_hints(type(conf))
    for key in dir(conf):
        if not key.startswith("__") and not callable(key):
            if key not in hinted_types:
                kt = type(getattr(conf, key))
                # logging.info(f"get_type_hints could not determine the type of {key} -> {kt}, so adding it manually")
                hinted_types[key] = kt

    if compare_type_with_docs:
        for k, v in hinted_types.items():
            if not variable_type_same_as_docstring_type(v, conf_doc[k]['type_name'].strip()):
                raise TypeError(f"For {k}, the type of {v} doesnt match {conf_doc[k]['type_name']}")
    return hinted_types


def get_doc(conf_doc, key):
    if key in conf_doc and len(conf_doc[key]['description']) > 0:
        return conf_doc[key]['description'].lower()
    else:
        return str(key)


# Assumptions:
# Lists are of any length but fixed types.
# Tuples can contain any mixed types but are of fixed length.
# Parser only handles numeric types, bools, str, lists, tuples, dicts, np.ndarray, Union/Optional types
def config_to_argparser(conf, parser=None, compare_type_with_docs=False):
    conf_doc = get_doc_from_config(conf)
    conf_types = get_all_variables_type(conf, conf_doc, compare_type_with_docs)  # get_type_hints(type(conf))
    conf_parameters = {
        key: getattr(conf, key) for key in dir(conf) if not key.startswith("_") and not callable(getattr(conf, key))
    }
    if parser is None:
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Create parser for various datatypes
    # In most cases start by parsing all arguments as str, then they will be converted by the action
    for (key, val) in conf_parameters.items():
        var_type = conf_types[key]
        nargs = None
        action = "store"
        metavar = ""
        to = get_origin(var_type)
        if var_type == list or to == list:  # We let argparse handle lists
            nargs = "*"
            var_type = var_type.__args__[0]
        elif var_type == dict or to == dict:
            action = StoreDictKeyPair
            metavar = "JSON or K=V[,K=V...]"
            var_type = str
        elif to == Union:
            action = parse_optional(var_type)
            var_type = str
        elif var_type == tuple:
            nargs = len(val)
            types = var_type.__args__
            action = convert_tuple_args_to_type(types)
            var_type = str
        elif var_type == bool:
            var_type = str2bool(key)
        else:
            action = "store"
        parser.add_argument("--" + key.replace('_', '-'),
                            default=val,
                            type=var_type,
                            nargs=nargs,
                            action=action,
                            metavar=metavar,
                            help=get_doc(conf_doc, key))
    return parser


# returns the type of val, unless val is NoneType, then returns str
def get_safe_type(val):
    return type(val) if not isinstance(val, type(None)) else str


class StoreDictKeyPair(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        s = values.strip()
        my_dict = {}

        # If it looks like JSON, parse it
        if s.startswith("{"):
            try:
                data = json.loads(s)
            except json.JSONDecodeError as e:
                parser.error(f"{option_string}: invalid JSON for {self.dest}: {e}")
            if not isinstance(data, dict):
                parser.error(f"{option_string}: JSON must be an object/dict")
            my_dict.update(data)
            setattr(namespace, self.dest, my_dict)
            return

        # Otherwise expect key=value[,key=value...]
        for kv in values.split(","):
            k, v = kv.split("=")
            my_dict[k] = v
        setattr(namespace, self.dest, my_dict)


def convert_args_to_type(t):
    class ArgumentTypeConvertor(argparse.Action):
        def __call__(self, parser, namespace, values, option_string):
            # Separately handle the case of ndarrays
            if t == np.ndarray:
                setattr(namespace, self.dest, np.array(values))
            else:
                setattr(namespace, self.dest, t(values))

    return ArgumentTypeConvertor


def convert_tuple_args_to_type(element_types):
    class TupleElementTypeConvertor(argparse.Action):
        def __call__(self, parser, namespace, values, option_string):
            result = []
            for (val, t) in zip(values, element_types):
                if t == bool:
                    result.append(str2bool(self.dest)(val))
                else:
                    result.append(t(val))
            setattr(namespace, self.dest, tuple(result))

    return TupleElementTypeConvertor


def str2bool(arg_name):
    def str2bool_(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError(
                f'Boolean value expected for argument {arg_name}.')

    return str2bool_


def strCheck(v):
    return None if v == "None" else v


def parse_optional(t):
    class OptionalArgumentParser(argparse.Action):
        def __call__(self, parser, namespace, value, option_string):
            if value == "None":
                setattr(namespace, self.dest, None)
            else:
                main_type = t.__args__[0]
                if main_type == bool:
                    main_type = str2bool(self.dest)
                setattr(namespace, self.dest, main_type(value))

    return OptionalArgumentParser
