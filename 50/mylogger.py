import logging

logging.basicConfig(filename='log.log',
                    filemode='a', 
                    level=logging.INFO, 
                    format="%(asctime)s.%(msecs)03d - %(levelname)s - [%(filename)s:%(lineno)s:%(funcName)s] - %(message)s", 
                    datefmt="%Y-%m-%d %H:%M:%S")

logger = logging.getLogger('logger')
