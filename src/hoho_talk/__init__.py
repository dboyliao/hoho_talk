import logging
import os

from .data import ConversationContext, ConversationMessage
from .talk_agent import OllamaTalkAgent

logger = logging.getLogger(__name__)
__all__ = ["OllamaTalkAgent", "ConversationContext", "ConversationMessage"]


def __setup_logger():
    log_level = os.environ.get("HOHO_TALK_LOG_LEVEL", "INFO")
    try:
        log_level = int(log_level)
    except ValueError:
        ...
    strm_handle = logging.StreamHandler()
    strm_handle.setFormatter(
        logging.Formatter(
            fmt=f"%(asctime)s %(levelname)s %(process)d-%(threadName)s %(message)s [%(name)s @ L%(lineno)d (%(funcName)s)]",
        )
    )
    logger.addHandler(strm_handle)
    logger.setLevel(log_level)


__setup_logger()
del __setup_logger
