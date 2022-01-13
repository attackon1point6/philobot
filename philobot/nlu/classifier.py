from philobot.base import Message, NLUData, NLUComponent
from typing import Dict, Text, Any


class EmotionClass(NLUData):
    def __init__(self, data: Dict[str, float]):
        assert sum(data.values()) == 1
        self.data = data


class EmotionClassifier(NLUComponent):

    """
    一个傻瓜情感分类器
    """

    happy_words = ["高兴", "快乐", "开心"]
    sad_words = ["悲哀", "悲惨", "心里苦", "呜", "想哭"]
    angry_words = ["气死了", "生气", "讨厌", "愤怒", "怒气"]


    def __call__(self, msg: Message) -> Message:

        text = msg.text
        happy_score = sum([1 if word in text else 0 for word in self.happy_words])
        sad_score = sum([1 if word in text else 0 for word in self.sad_words])
        angry_score = sum([1 if word in text else 0 for word in self.angry_words])

        total = sum((happy_score, sad_score, angry_score))

        happy_score /= total
        sad_score /= total
        angry_score /= total

        emotion_prob = {"happy": happy_score, "sad": sad_score, "angry": angry_score}
        msg.set(EmotionClass(emotion_prob))

        return msg
