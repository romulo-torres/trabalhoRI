import logging
import os


def setup_logger(name="pipeline", log_file="../logs/pipeline.log", level=logging.INFO):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # evitar duplicação de logs
    if logger.hasHandlers():
        return logger

    # formato do log
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # salvar em arquivo
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    # mostrar no terminal também
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger