import ast
import json
import logging
import os

# Configure the logger
logging.basicConfig(level=logging.INFO)

# Create a logger
logger = logging.getLogger(__name__)


class NoConfigData(Exception):
    pass


def parse_config(data: str) -> dict:
    if not data:
        msg = "Emtpy Data passed"
        raise NoConfigData(msg)

    if os.path.isfile(data):
        with open(data, "r") as f:
            output = json.load(f)
    else:
        # Airflow passes a String with Python Syntax in it.
        output = ast.literal_eval(data)

    return output
