import random
import time


class SingletonMeta(type):

    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


class EchoModel(metaclass=SingletonMeta):
    name = 'echo'

    def __init__(self):
        self._model = None

    def prepare_data(self, data):
        time.sleep(random.randint(1, 5))
        return data

    def run(self, data: str) -> str:
        time.sleep(random.randint(3, 10))
        data = self.prepare_data(data)
        return data.upper()


model = EchoModel()
