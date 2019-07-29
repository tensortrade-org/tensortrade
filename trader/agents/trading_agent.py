from abc import ABCMeta, abstractmethod

from trader.features import FeaturePipeline


class TradingAgent(object, metaclass=ABCMeta):
    @abstractmethod
    def __init__(self):
        pass

    def add_feature_pipeline(self, pipeline: FeaturePipeline):
        pass
