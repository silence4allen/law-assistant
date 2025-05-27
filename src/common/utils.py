# -*- coding: utf-8 -*-#
import yaml


def read_from_yaml(file_path):
    with open(file_path, "r") as f:
        return yaml.load(f, Loader=yaml.FullLoader)
