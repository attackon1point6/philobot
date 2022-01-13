from philobot.base import Message, NLUData, NLUComponent
from typing import Dict, Text, Any, Tuple, List
import numpy as np


class SentenceEmbedding(NLUData):
    def __init__(self, data: np.ndarray):
        self.data = data


class TokenEmbedding(NLUData):
    def __init__(self, data: np.ndarray):
        self.data = data


class BertFeaturizer(NLUComponent):
    """Featurizer using transformer-based language models.
    自带分词器。读入用户输入text，输出句子的向量表示。
    TODO 还没试过也不知道能用不能用……
    """

    def __init__(self) -> None:

        from transformers import BertTokenizer, BertModel

        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path="bert-base-chinese")
        self.model = BertModel.from_pretrained(pretrained_model_name_or_path="bert-base-chinese")
        self.pad_token_id = self.tokenizer.unk_token_id
        self.max_model_sequence_length = 512

        # 目前只支持直接使用 不支持训练
        self.model.eval()


    def _lm_tokenize(self, text: Text) -> Tuple[List[int], List[Text]]:
        """Pass the text through the tokenizer of the language model.
        该语言模型自用的分词器

        Args:
            text: Text to be tokenized.

        Returns: List of token ids and token strings.
        """
        split_token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        split_token_strings = self.tokenizer.convert_ids_to_tokens(split_token_ids)

        return split_token_ids, split_token_strings


    def __call__(self, message: Message) -> Message:
        """Process an incoming message by computing its tokens and dense features.
        """
        text = message.text
        token_ids, token_strings = self._lm_tokenize(text)
        sentence_embedding, token_embedding = self._lm_featurize(token_ids)
        message.set(SentenceEmbedding(sentence_embedding))
        message.set(TokenEmbedding(token_embedding))

        return message


    def _lm_featurize(self, token_ids: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        获得句子和token层面的稠密表示
        """

        # 添加bert使用的特殊token
        BERT_CLS_ID = 101
        BERT_SEP_ID = 102
        token_ids.insert(0, BERT_CLS_ID)
        token_ids.append(BERT_SEP_ID)
        if len(token_ids) > self.max_model_sequence_length:
            raise RuntimeError("句子过长超出模型处理能力")

        # 补齐padding
        actual_sequence_length = len(token_ids)
        padded_token_ids = token_ids + [self.pad_token_id] * (self.max_model_sequence_length - actual_sequence_length)

        # 计算 attention mask
        attention_mask = [1] * len(token_ids) + [0] * (self.max_model_sequence_length - actual_sequence_length)
        attention_mask = np.array(attention_mask).astype(np.float32)

        # 模型计算
        model_outputs = self.model(np.array(padded_token_ids), attention_mask=attention_mask)
        sequence_hidden_states = model_outputs[0]
        sequence_hidden_states = sequence_hidden_states.numpy()
        sequence_hidden_states = sequence_hidden_states[:actual_sequence_length]

        # 获得句子和token层面的稠密表示
        sentence_embedding = sequence_hidden_states[0]          # 第一个位置的输出是句子向量
        token_embedding = sequence_hidden_states[1:-1]          # 去除CLS,SEP
        sentence_embedding = np.array(sentence_embedding)
        token_embedding = np.array(token_embedding)

        return sentence_embedding, token_embedding
