import functions
from logging import config, getLogger

config.fileConfig("logging_functions.conf")

logger = getLogger()

logger.info("info level log")
logger.debug("debug level log")

functions.test_func()