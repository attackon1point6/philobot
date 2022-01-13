import logging
from typing import Dict, List, Callable, Optional
from .manager import Manager
from philobot.base import Message, Response, NLUComponent


class PhiloBot(object):
    """
    聊天机器人主体
    """
    def __init__(self, name: str,
                 managers: List[Manager],
                 nlu_pipeline: List[NLUComponent],
                 **kwargs):
        # 核心部件
        self.name = name
        self.nlu_pipeline = nlu_pipeline
        self.managers = managers
        self.chair = None                   # type: Optional[Manager]

        for m in self.managers:
            m.connect(self)

        # 可选的关键字参数
        self.logger = kwargs.get('logger', logging.getLogger(__name__))
        self.low_confidence = kwargs.get('low_confidence', 0.5)
        self.default_manager = kwargs.get('default_manager', None)


    def response(self, message: Message) -> Optional[Response]:
        """获得confidence最高的Response"""
        candidates = [m.response(message) for m in self.managers]
        max_confidence = max([r.confidence for r in candidates])
        resp = [r for r in candidates if r.confidence == max_confidence][0]     # 如有多个直选第一个
        return resp


    def __call__(self, text: str) -> Optional[Response]:

        message = Message(text=text)
        for nlu in self.nlu_pipeline:
            message = nlu(message)

        if self.chair is None:
            resp = self.response(message)
        else:
            resp = self.chair.response(message)

        if resp.confidence < self.low_confidence:

            self.logger.log(2, "confidence lower than {}".format(self.low_confidence))
            if self.default_manager is not None:
                return self.default_manager.response(message)
            else:
                return None
        else:
            resp.manager.adopt()
            return resp
