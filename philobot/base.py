from typing import List, Type, Union, Optional



class NLUData(object):
    """在Message中放置的各种附加信息"""

    @property
    def name(self) -> str:
        return str(self.__class__)



class Message(object):
    """存储某次用户输入信息的数据结构。包括原始文本输入和NLU流程所逐步放入的附加信息"""

    def __init__(self, text, **kwargs):
        self.text = text
        self.nlu_data = {}

    def set(self, data: NLUData):
        self.nlu_data[data.name] = data

    def get(self, name: Union[str, Type]) -> Optional[NLUData]:
        if not isinstance(name, str):
            name = str(name)
        if name in self.nlu_data:
            return self.nlu_data[name]
        return None


class Response(object):
    """存储某个Manger对某个用户输入的回应的数据结构
    包括发送给用户的文本回应内容，发送给PhiloBot用的信心值，发送的Manager实例
    """

    def __init__(self, text: str, confidence: float, manager):
        self.text = text
        self.confidence = confidence
        self.manager = manager



class NLUComponent(object):

    @classmethod
    def required_data(cls) -> List[Type[NLUData]]:
        """这个Component所依赖的NLU数据类型，空表示只需要text"""
        return []


    @property
    def name(self) -> str:
        """Returns the name of the component to be used in the model configuration.
        """
        return str(type(self).name)


    def __call__(self, msg: Message) -> Message:
        raise NotImplementedError


