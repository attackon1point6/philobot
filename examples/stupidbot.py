from philobot import PhiloBot
from typing import List, Type, Callable, Optional
from philobot.base import Message, Response, NLUData, NLUComponent
from philobot.manager import Manager
import numpy as np


class StupidFeature(NLUData):
    def __init__(self, data: np.ndarray):
        self.data = data


class StupidFeaturizer(NLUComponent):
    """关键字匹配简陋特征提取器！
    """
    keyword = [
        "你好",
        "请问",
        "你会",
        "闹钟",
        "无话可说",
        "好事",
        "再见"
    ]
    def __call__(self, msg: Message) -> Message:

        exist = [word in msg.text for word in self.keyword]
        exist = np.array(exist)
        msg.set(StupidFeature(exist))

        return msg
    


class FAQManager(Manager):
    """如果输入和预设问题相似则返回对应答复"""

    @classmethod
    def required_data(cls) -> List[Type[NLUData]]:
        """这个Manager要求Message必须要有的数据类型，空表示只需要text"""
        return [StupidFeature, ]

    def response(self, message: Message) -> Response:
        if not self.check_required_data(message):
            raise RuntimeError("required nlu data missing!")

        feat = message.get(StupidFeature)
        feat = feat.data
        if feat[1] > 0.5:
            return Response(text="请脑补我回答了你{}的问题".format(message.text), confidence=1, manager=self)
        else:
            return Response(text="", confidence=0, manager=self)



class GreetManager(Manager):
    """各种和打招呼有关的逻辑，做机器人要有礼貌"""

    @classmethod
    def required_data(cls) -> List[Type[NLUData]]:
        """这个Manager要求Message必须要有的数据类型，空表示只需要text"""
        return [StupidFeature, ]

    def response(self, message: Message) -> Response:
        if not self.check_required_data(message):
            raise RuntimeError("required nlu data missing!")

        feat = message.get(StupidFeature)
        feat = feat.data
        if feat[0] > 0.5:
            return Response(text="你也好呀！", confidence=1, manager=self)
        elif feat[6] > 0.5:
            return Response(text="你也再见！", confidence=1, manager=self)
        else:
            return Response(text="", confidence=0, manager=self)

    def utter(self) -> Response:
        """主动说话"""
        return Response(text="你好呀！", confidence=1, manager=self)


class TopicTalkManager(Manager):
    """在用户没话题的时候引领话题"""

    @classmethod
    def required_data(cls) -> List[Type[NLUData]]:
        """这个Manager要求Message必须要有的数据类型，空表示只需要text"""
        return [StupidFeature, ]

    def response(self, message: Message) -> Response:
        if not self.check_required_data(message):
            raise RuntimeError("required nlu data missing!")

        feat = message.get(StupidFeature)
        feat = feat.data
        if feat[2] > 0.5 or feat[4]:
            return Response(text="我会FAQ，记录三件好事，还有定闹钟！", confidence=1, manager=self)
        else:
            return Response(text="！", confidence=0, manager=self)

    def utter(self) -> Response:
        """主动说话"""
        return Response(text="如果没啥好说的，可以问问我如何记录三件好事", confidence=1, manager=self)



class GratitudeManager(Manager):
    """三件好事"""

    def __init__(self, name: str):
        super().__init__(name)
        self.gratitude_list = []    # 三件好事


    @classmethod
    def required_data(cls) -> List[Type[NLUData]]:
        """这个Manager要求Message必须要有的数据类型，空表示只需要text"""
        return [StupidFeature, ]

    def response(self, message: Message) -> Response:
        if not self.check_required_data(message):
            raise RuntimeError("required nlu data missing!")

        text = message.text
        feat = message.get(StupidFeature)
        feat = feat.data

        # 三件好事的处理Script
        if not self.activated:
            if feat[5] > 0.5:
                self.todo = self._activate
                return Response(text="来告诉我今天发生的三件好事吧！", confidence=1, manager=self)
            else:
                return Response(text="", confidence=0, manager=self)

        # 进入自定义处理流程

        if "停止" in text:
            self.todo = self._deactivate
            rsp = Response(text="好吧不记录三件好事了", confidence=1, manager=self)
            self.gratitude_list = []
            return rsp
        else:
            self.gratitude_list.append(text)
            if len(self.gratitude_list) < 3:
                return Response(text="知道啦！下一件好事呢？", confidence=1, manager=self)
            else:
                self.todo = self._deactivate
                rsp = Response(text="三件好事我都记住了哦！" + ', '.join(self.gratitude_list), confidence=1, manager=self)
                self.gratitude_list = []
                return rsp


class TimerManager(Manager):
    def __init__(self, name: str):
        super().__init__(name)
        self.todo = None      # Union[List[Callable], Callable]

    @classmethod
    def required_data(cls) -> List[Type[NLUData]]:
        """这个Manager要求Message必须要有的数据类型，空表示只需要text"""
        return [StupidFeature, ]


    @staticmethod
    def _set_timer(second) -> Callable:
        import threading
        timer = threading.Timer(second, lambda: print("时间到啦！"))
        return lambda: timer.start()


    def response(self, message: Message) -> Response:
        if not self.check_required_data(message):
            raise RuntimeError("required nlu data missing!")

        text = message.text
        feat = message.get(StupidFeature)
        feat = feat.data


        import re


        if feat[3] > 0.5:
            digits = re.findall(r"\d+", text)
            if not digits:
                self.todo = self._activate
                return Response(text="告诉我你想定多长时间的闹钟吧", confidence=1, manager=self)
            else:
                second = int(digits[0])
                self.todo = [self._set_timer(second), self._deactivate]
                return Response(text="闹钟订好啦！{}秒后会提醒你".format(second), confidence=1, manager=self)

        if self.activated:
            digits = re.findall(r"\d+", text)
            second = int(digits[0])
            self.todo = [self._set_timer(second), self._deactivate]
            return Response(text="闹钟订好啦！{}秒后会提醒你".format(second), confidence=1, manager=self)


        return Response(text="", confidence=0, manager=self)



class SimpleUtterManager(Manager):

    def __init__(self, name: str, resp: str):
        super().__init__(name)
        self.resp = resp

    def response(self, message: Message) -> Response:
        text = message.text
        if "不错" in text or "你" in text:
            return Response(text="我是你的好朋友呀！快问问我会干什么吧？", confidence=0.1, manager=self)

        if "哈" in text:
            return Response(text="你高兴我也高兴", confidence=0.1, manager=self)
        if "哎" in text:
            return Response(text="哎？", confidence=0.1, manager=self)
        if len(text) <= 2:
            return Response(text="你{}个什么呀".format(text), confidence=0.1, manager=self)

        return Response(text="听不懂哎~", confidence=0.1, manager=self)

    def utter(self) -> Response:
        """主动说话"""
        return Response(text=self.resp, confidence=1, manager=self)



class StupidBot(PhiloBot):
    def __init__(self, **kwargs):
        name = "Mike"
        managers = [
            FAQManager("faq"),
            GreetManager("greet"),
            TopicTalkManager("topic"),
            GratitudeManager("gratitude"),
            TimerManager("timer"),
        ]
        nlu_pipeline = [
            StupidFeaturizer(),
        ]
        self.chair = None
        super().__init__(name, managers, nlu_pipeline, default_manager=SimpleUtterManager("simple", "听不懂哦~"))


if __name__ == '__main__':
    bot = StupidBot()
    while 1:
        inp = input("☯♈💖💗💗>>>")
        resp = bot(inp)
        print(resp.text)
