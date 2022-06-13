from logging import getLogger
logger = getLogger(__name__)

def test_func():
    logger.info("test func: info level log")
    logger.debug("test func: debug level log")