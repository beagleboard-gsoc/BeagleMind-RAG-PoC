import logging

class App:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.handler = logging.FileHandler('app.log')
        self.handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(self.handler)

    def run(self):
        self.logger.debug('App started')
        # your app code here
        self.logger.info('App finished')
