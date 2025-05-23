import logging
import colorlog
import colorama


#-----------------------------------logger + ⭐️ fancy colors ⭐️ ----------------------------------------------- 

colorama.init()  # Required on Windows to support ANSI colors

handler = colorlog.StreamHandler()
handler.setFormatter(colorlog.ColoredFormatter(
    "%(log_color)s[%(levelname)s] %(message)s",
    log_colors={
        'DEBUG':    'cyan',
        'INFO':     'green',
        'WARNING':  'yellow',
        'ERROR':    'red',
        'CRITICAL': 'bold_red,bg_white',
    }
))

logger = logging.getLogger("ai_script")
logger.setLevel(logging.INFO)

# Create and attach a console handler
#formatter = logging.Formatter('[%(levelname)s] %(message)s')
#console.setFormatter(formatter)
#console = logging.StreamHandler()
logger.addHandler(handler)

# Function to toggle debug logging
def enable_debug():
    logger.setLevel(logging.DEBUG)
    logger.debug("Debugging enabled")

def disable_debug():
    logger.setLevel(logging.INFO)
    logger.info("Debugging disabled")

