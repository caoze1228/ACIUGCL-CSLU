import collections
import json
import os.path
from typing import Literal

import torch
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from params import Params

"""
curr_item = {
            'uuid':      uuid,
            'curr_turn': curr_turn,
            'text':      text,
            'intent':    intent,
            'history':   history[:],
            'slots':     bio.rstrip().split()
        }
"""


class Dataset(TorchDataset):

    @staticmethod
    def split(sentence: str):
        all_sentences = sentence.split('[SEP]')
        result = []
        for sent in all_sentences:
            result.extend(list(sent))
            result.append('[SEP]')

        result.pop()
        return result

    def __init__(self, args: Params, status: Literal['train', 'test', 'dev'] = 'test'):
        self.args = args

        self.corpus = json.load(open(os.path.join('dataset',
                                                  args.dataset,
                                                  f'{status}.json')))
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path='/root/cgcn/cgcn_-af-master-gpu-gca/cgcn_-af-master/bert')
        self.dataset_dir = os.path.join('dataset', args.dataset)
        with open(os.path.join(self.dataset_dir, args.label_fn)) as fp:
            lines = fp.readlines()
            self.all_intents = list(map(lambda x: str(x).rstrip(), lines))
        with open(os.path.join(self.dataset_dir, args.slot_label)) as fp:
            lines = fp.readlines()
            self.all_slots = list(map(lambda x: str(x).rstrip(), lines))
        self.intent2ids = {intent: idx for idx, intent in enumerate(self.all_intents)}
        self.ids2intent = {v: k for k, v in self.intent2ids.items()}
        self.slot2ids = {slot: idx for idx, slot in enumerate(self.all_slots)}
        self.ids2slot = {v: k for k, v in self.slot2ids.items()}
        self.SLOT_PAD_IDS = self.slot2ids['PAD']
        self.device = args.device

    @property
    def num_slots(self):
        return len(self.slot2ids)

    @property
    def num_intents(self):
        return len(self.intent2ids)

    def map_slot2ids(self, slot_str):
        return list(map(self.slot2ids.get, slot_str.split()))

    def __len__(self):
        return len(self.corpus)

    @staticmethod
    def max_len(text, history):
        result = len(text)
        for sent in history:
            if len(sent) > result:
                result = len(sent)
        return result

    def build_input_ids(self, text, max_len, slot_ids=None):

        token_id = 0
        token_ids = []
        slot_label_mask = []
        if len(text) > max_len:
            # MARK: truncate
            text = text[:max_len]

            text = [self.tokenizer.cls_token] + text + [self.tokenizer.sep_token]
            if slot_ids is not None:
                slot_ids_ = slot_ids[:max_len]
                slot_label_mask = [self.SLOT_PAD_IDS] + [self.SLOT_PAD_IDS + 1] + \
                                  [self.SLOT_PAD_IDS for _ in range(len(slot_ids_) - 1)]
                slot_ids_ = [self.slot2ids["O"]] + slot_ids_ + [self.slot2ids["O"]]
                slot_label_mask.append(self.SLOT_PAD_IDS)
            else:
                slot_ids_ = []
            attn_mask = [1 for _ in text]
            for t in text:
                if t == self.tokenizer.sep_token:
                    token_id = 1 - token_id
                token_ids.append(token_id)
        else:
            text = [self.tokenizer.cls_token] + text + [self.tokenizer.sep_token]
            if slot_ids is not None:
                slot_ids_ = slot_ids[:]
                slot_label_mask = [self.SLOT_PAD_IDS] + [self.SLOT_PAD_IDS + 1] + \
                                  [self.SLOT_PAD_IDS for _ in range(len(slot_ids_) - 1)]
                slot_ids_ = [self.slot2ids["O"]] + slot_ids_ + [self.slot2ids["O"]]
                slot_label_mask.append(self.SLOT_PAD_IDS)
            else:
                slot_ids_ = []
            attn_mask = [1 for _ in text]
            for t in text:
                if t == self.tokenizer.sep_token:
                    token_id = 1 - token_id
                token_ids.append(token_id)
            num_to_pad = max_len - len(text) + 2
            text = text + [self.tokenizer.pad_token for _ in range(num_to_pad)]
            attn_mask = attn_mask + [0 for _ in range(num_to_pad)]
            token_ids = token_ids + [0 for _ in range(num_to_pad)]
            if slot_ids is not None:
                for _ in range(num_to_pad):
                    slot_ids_.append(self.SLOT_PAD_IDS)
            else:
                slot_ids_ = []
        text_ids = self.tokenizer.convert_tokens_to_ids(text)
        if slot_ids_:
            assert len(text_ids) == len(slot_ids_) == len(attn_mask) == len(token_ids), \
                f"{text_ids=} but {slot_ids_=}, {len(slot_ids_)=} and {len(text_ids)=}"
        return {
            "input_ids":      torch.tensor(text_ids, device=self.device, dtype=torch.long),
            'token_type_ids': torch.tensor(token_ids, device=self.device, dtype=torch.long),
            'attention_mask': torch.tensor(attn_mask, device=self.device, dtype=torch.long),
            "slot_ids":       torch.tensor(slot_ids_, device=self.device, dtype=torch.long)
        }

    def __getitem__(self, index):
        item = self.corpus[index]
        if not self.args.dataset.startswith('Sim'):
            text = self.split(item['text'])
        else:
            text = item['text'].split()
        history_sentences = item['history']
        if self.args.enable_context and 0 < self.args.context_window < len(history_sentences):
            ln = len(history_sentences)
            history_sentences = history_sentences[ln - self.args.context_window:ln - 1]
        history = list(map(self.split, history_sentences))
        slot_ids = list(map(self.slot2ids.get, item['slots']))
        max_ln = min(510, self.max_len(text, history))
        sentence_input = self.build_input_ids(text, max_len=max_ln, slot_ids=slot_ids)
        if '|' in item['intent']:
            intent = item['intent'].split('|')[0]
        elif '｜' in item['intent']:
            intent = item['intent'].split('｜')[0]
        else:
            intent = item['intent']
        it_ = self.intent2ids[intent] if intent in self.intent2ids.keys() else self.intent2ids['UNK']
        intent_ids = torch.tensor(it_, device=self.device)
        history_ids = []
        for hist in history:
            hids = self.build_input_ids(hist, max_len=max_ln)
            history_ids.append(hids)
        history_ids = self.list2dict(history_ids)
        # if len(history_ids) == 0:
        #     return sentence_input, torch.tensor([], device=self.device), intent_ids
        # history_ids = self.list2dict(history_ids)
        return sentence_input, history_ids, intent_ids

    def list2dict(self, lt):
        if len(lt) == 0:
            return {
                "input_ids":      torch.tensor([], device=self.device, dtype=torch.long),
                'token_type_ids': torch.tensor([], device=self.device, dtype=torch.long),
                'attention_mask': torch.tensor([], device=self.device, dtype=torch.long),
                "slot_ids":       torch.tensor([], device=self.device, dtype=torch.long)
            }
        else:
            keys = lt[0].keys()
            result = collections.defaultdict(list)
            for item in lt:
                for key in keys:
                    result[key].append(item[key])
            for k in result.keys():
                result[k] = torch.stack(result[k])
            return result

    @staticmethod
    def debug():
        args = Params().parse_args()
        dst = Dataset(args)
        dataloader = DataLoader(dataset=dst, shuffle=False, batch_size=1)
        for idx, (sentence_input, history_ids, intent_ids) in enumerate(dataloader):
            print(f"{sentence_input=}")
            print(f"{history_ids=}")
            print(f"{intent_ids=}")
            if idx == 3:
                break


if __name__ == '__main__':
    Dataset.debug()
