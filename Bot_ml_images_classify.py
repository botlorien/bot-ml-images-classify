import logging

try:
    import flow
    #from app import BotApp

    #app = BotApp()

    if __name__ == '__main__':
        flow.main_ui_free()
except Exception as e:
    logging.exception(e)
    with open('logLastException.txt', 'w') as f:
        f.write(str(e))
