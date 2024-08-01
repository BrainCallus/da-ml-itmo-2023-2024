from datetime import datetime


class Logger:
    def __init__(self, clazz):
        self.clazz = clazz

    def print_log(self, message, tag):
        print(f'{tag} {self.clazz} {datetime.now()}: {message}')

    def log_error(self, message):
        self.print_log(message, '\033[1;31m [ERROR]\033[0m')

    def warn(self, message):
        self.print_log(message, '\033[1;93m [WARN]\033[0m')

    def info(self, message):
        self.print_log(message, '\033[1;96m [INFO]\033[0m')