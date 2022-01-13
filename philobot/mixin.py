from typing import List
from philobot.base import Message, Response


class ChatElement(object):
    def __init__(self, element, turn):
        assert isinstance(element, (Message, Response))
        self.element = element
        self.turn = turn


class RecoderMixin(object):
    def __init__(self):
        # 各类记录
        self.history = list()  # type: List[ChatElement]
        self.turn = 0  # 第0轮对话

    def record_history(self, element: ChatElement):
        """将某次用户输入或机器人回复记录到机器人的对话历史中"""
        self.turn += 1
        self.history.append(ChatElement(element=element, turn=self.turn))


