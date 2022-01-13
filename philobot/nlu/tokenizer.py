from philobot.base import Message, NLUData, NLUComponent
from typing import Dict, Text, Any, List


class Token(NLUData):
    def __init__(self, data: List[str]):
        self.data = data


class JiebaTokenizer(NLUComponent):
    """This tokenizer is a wrapper for Jieba (https://github.com/fxsjy/jieba)."""

    def __call__(self, msg: Message) -> Message:
        import jieba

        text = msg.text

        tokenized = jieba.tokenize(text)
        tokens = [word for (word, start, end) in tokenized]
        msg.set(Token(tokens))

        return msg

