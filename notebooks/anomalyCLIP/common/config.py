from yacs.config import CfgNode as CN
import os
import yaml
from typing import Dict

def load_cfg_from_cfg_file(file: str) -> CN:
    """
    Load configuration parameters from a YAML file and create a CfgNode.

    :param file: The path to the YAML configuration file.
    :type file: str
    :returns: The CfgNode object containing the configurations from the YAML file.
    :rtype: CN
    """
    cfg = {}

    with open(file, 'r') as f:
        cfg_from_file = yaml.safe_load(f)

    for key in cfg_from_file:
        #add this loop due to the structure of the yaml file
        for k, v in cfg_from_file[key].items():
            cfg[k] = v

    cfg = CN(cfg)

    return cfg

def merge_cfg_from_args(cfg: CN, args: Dict) -> CN:
    """
    Merge configuration settings from args arguments into the CfgNode object.

    :param cfg: The CfgNode object representing the configuration.
    :type cfg: CN
    :param args: A dictionary containing command-line arguments.
    :type args: Dict
    :returns: The updated CfgNode object after merging the arguments.
    :rtype: CN
    """

    if not isinstance(args, dict):
        args_dict = args.__dict__
    else:
        args_dict = args

    # If cfg.model is 'FPTrans', add specific variables
    if cfg.model == 'FPTRANS':
        args_dict['num_prompt'] = 12*(1+args_dict['kshot']*cfg.bg_num)
        args_dict['pretrained'] = (
        f'./models/FPTRANS/path_to_pretrained_model/'
        f'{cfg.backbone}.{cfg.extension_pretrained}'
        )

    for k ,v in args_dict.items():
        cfg[k] = v 

    return cfg

def load_cfg_to_experiment(file: str) -> CN:
    """
    Load a YAML configuration file and convert it into a CfgNode object.

    :param file: The path to the YAML configuration file.
    :type file: str
    
    :returns: The CfgNode object containing the configurations from the YAML file.
    :rtype: CN
    """

    with open(file, 'r') as f:
        cfg_from_file = yaml.safe_load(f)

    cfg = CN(cfg_from_file)
    return cfg
