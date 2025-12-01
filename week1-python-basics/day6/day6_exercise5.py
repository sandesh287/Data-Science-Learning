# program to log messages with timestamps into a file

import logging

logging.basicConfig(
  filename='app.log',
  level=logging.INFO,
  format='%(asctime)s - %(levelname)s - %(message)s',
  datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)

logger.info("This is an informational message.")
logger.warning("A warning occured here.")
logger.error("An error message was logged.")