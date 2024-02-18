import logging

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(module)s.%(filename)s.%(funcName)s.%(lineno)d: %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
    level=logging.INFO,
)
