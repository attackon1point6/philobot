from typing import List, Callable, Dict, Tuple, Type, Optional
from philobot.base import Message, Response, NLUData


class Manager(object):
    """
    对话管理器主体
    """

    @classmethod
    def required_data(cls) -> List[Type[NLUData]]:
        """这个Manager要求Message必须要有的数据类型，空表示只需要text"""
        return []

    @classmethod
    def check_required_data(cls, message: Message) -> bool:
        nlu_data = list(message.nlu_data.keys())
        return not (False in [str(req) in nlu_data for req in cls.required_data()])

    def __init__(self, name: str):
        self.name = name
        self.bot = None
        self.activated = False
        self.todo = None

    def connect(self, bot):
        self.bot = bot

    def response(self, message: Message) -> Response:
        """Manager根据当前用户输入给出自己的回复"""
        raise NotImplementedError

    def _activate(self):
        self.activated = True
        self.bot.chair = self

    def _deactivate(self):
        self.activated = False
        self.bot.chair = None

    def adopt(self):
        """PhiloBot采纳了Manager的回复后要做出的行动"""
        if self.todo is None:
            return
        if isinstance(self.todo, list):
            for f in self.todo:
                f()
        else:
            self.todo()
        self.todo = None
