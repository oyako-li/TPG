# import json
from logging import getLogger, config
config.fileConfig('logging_debug.conf')

# ここからはいつもどおり
logger = getLogger(__name__)

logger.info(f'message{__name__}')
logger.debug('debug level log')