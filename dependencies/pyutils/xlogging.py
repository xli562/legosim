# Provides logger for other modules.

import sys
import os
from loguru._handler import Message
from loguru import logger as _logger
import dotenv


dotenv.load_dotenv()
PROJECT_ROOT = os.getenv('PROJECT_ROOT')
DATA_FILE_PATH = f'{PROJECT_ROOT}/logs/xdata.log'
LOG_FILE_PATH = f'{PROJECT_ROOT}/logs/xlogging.log'

stdout_lavel = 'DEBUG'
stderr_level = 'ERROR'
logfile_level = 'DEBUG'
# Logging level above which the application will crash
CRASH_LEVEL = 'CRITICAL'

FORCE_CRASH = False


def crash_application(message:Message):
    """ Crashes application.

    :param record: provided by logger.add. Contains all contextual information
            of the logging call (time, function, file, line, level, etc.)
    """
    if message.record['level'].no >= logger.level(CRASH_LEVEL).no:
        sys.stderr.flush()
        sys.stdout.flush()
        if FORCE_CRASH:
            os._exit(1)
        else:
            sys.exit(1)

def get_logger():
    if os.getenv('is_logger_configured') != 'True':
        _logger.remove()
        _logger.add(sys.stdout,
                   level=stdout_lavel,
                   filter=lambda x: x['level'].no < _logger.level(stderr_level).no)
        _logger.add(sys.stderr, level=stderr_level)
        _logger.add(LOG_FILE_PATH, level=logfile_level, enqueue=True)
        _logger.add(crash_application, level=CRASH_LEVEL)

        # Data recording functionality
        _logger.level('DATA', no=5)
        _logger.add(DATA_FILE_PATH, level='DATA',
                   enqueue=True, mode='w',
                   format='{message}',
                   filter=lambda x: x['level'].no == _logger.level('DATA').no)

        # _logger.debug(f'Logging level set to {stdout_lavel} for stdout.')
        # _logger.debug(f'Logging level set to {stderr_level} for stderr.')
        # _logger.debug(f'Logging level set to {logfile_level} for {LOG_FILE_PATH}.')
        # _logger.debug(f'Application will crash on {CRASH_LEVEL}.')
        os.environ['is_logger_configured'] = 'True'

    return _logger

logger = get_logger()

def log_data(data):
    logger.log('DATA', str(data))
