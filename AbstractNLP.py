from abc import ABC, abstractmethod

class AbstractNLP(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def getvars(self, x):
        pass

    @abstractmethod
    def objective(self, x):
        pass

    @abstractmethod
    def gradient(self, x):
        pass

    @abstractmethod
    def constraints(self, x):
        pass

    def jacobian(self, x):
        pass