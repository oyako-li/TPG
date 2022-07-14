import logging
import time
import os
from datetime import datetime
# LOG = logging
# LOG.basicConfig(format='[%(asctime)s][%(levelname)s](%(filename)s:%(lineno)s) %(name)s:%(message)s')

def setup_logger(_name, _logfile='LOGFILENAME', test=False, load=True):
    _logger = logging.getLogger(_name)
    _logger.setLevel(logging.DEBUG)

    _filename = datetime.today().strftime('%Y-%m-%d_%H-%M-%S')
    print(f'{_logfile} {_filename}')
    time.sleep(0.5)
    # create file handler which logs even DEBUG messages
    if not test:
        while True:
            try:
                _fh = logging.FileHandler(f'log/{_logfile}/{_filename}.log')
                break
            except FileNotFoundError:
                os.makedirs(f'log/{_logfile}')

        _fh.setLevel(logging.INFO)
        _fh_formatter = logging.Formatter('%(asctime)s, %(levelname)s, %(filename)s, %(message)s')
        _fh.setFormatter(_fh_formatter)
        _logger.addHandler(_fh)


    # create console handler with a INFO log level
    if load:
        _ch = logging.StreamHandler()
        _ch.setLevel(logging.DEBUG)
        _ch_formatter = logging.Formatter('[{}][{}]%(name)s,%(funcName)s:%(message)s'.format(_logfile, _filename))
        _ch.setFormatter(_ch_formatter)

        # add the handlers to the logger
        _logger.addHandler(_ch)
    return _logger, _filename

# logger = setup_logger(__name__, 2)