import logging
from datetime import datetime
try:
    from scripts import flow
    flow.main_ui_free()
except Exception as e:
    logging.exception(e)
    with open('logExceptions.txt', 'a') as f:
        f.write(f"\nError[{str(datetime.now())}]: " + str(e))
