import logging
import os

from ahmedkobtan_cinema_robot_playground.src.utils.config_utils import (  # noqa
    parse_config,
)

from .mocked_dev_json import dev_configs
from .mocked_prod_json import prod_configs
from .mocked_qa_json import qa_configs

logger = logging.getLogger("default")


def test_parse_dev_config_json():
    current_dir = os.path.dirname(os.path.realpath(__file__))

    # Construct the path to the config.json file
    config_path = os.path.join(current_dir, "..", "..", "..", "config", "dev.json")

    # Construct the path to the config.json file
    output = parse_config(config_path)
    try:
        assert output == dev_configs
    except Exception as e:
        logger.error(output)
        logger.error(dev_configs)
        logger.error(e)
        raise e

    # assert output == dev_configs


def test_parse_prod_config_json():
    current_dir = os.path.dirname(os.path.realpath(__file__))

    # Construct the path to the config.json file
    config_path = os.path.join(current_dir, "..", "..", "..", "config", "prod.json")
    output = parse_config(config_path)

    try:
        assert prod_configs == output
    except Exception as e:
        print(set(prod_configs.items()) - set(output.items()))
        print(set(output.items()) - set(prod_configs.items()))
        raise ValueError(e)


def test_parse_dev_config_str():
    output = parse_config(str(dev_configs))

    assert output == dev_configs


def test_parse_prod_and_dev_have_same_upper_fields_config_json():
    current_dir = os.path.dirname(os.path.realpath(__file__))

    # Construct the path to the config.json file
    dev_config_path = os.path.join(current_dir, "..", "..", "..", "config", "dev.json")

    prod_config_path = os.path.join(
        current_dir, "..", "..", "..", "config", "prod.json"
    )

    dev_config = parse_config(dev_config_path)
    prod_config = parse_config(prod_config_path)

    assert dev_config.keys() == prod_config.keys()


def test_parse_qa_config_json():
    current_dir = os.path.dirname(os.path.realpath(__file__))

    # Construct the path to the config.json file
    config_path = os.path.join(current_dir, "..", "..", "..", "config", "qa.json")
    output = parse_config(config_path)

    assert qa_configs == output
