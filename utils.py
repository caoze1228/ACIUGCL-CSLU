import dgl
import torch
import numpy as np
import logging
from rich.logging import RichHandler
from rich.console import Console
from typing import Callable
from logging import Logger as InternalLogger
import threading
from seqeval.metrics import precision_score, recall_score, f1_score

NELL_FILE = '/dev/null'

inner_logger = Console().log


def get_logger(filename: str = NELL_FILE):
    FORMAT = "[ln %(lineno)d] %(asctime)s %(levelname)s: %(message)s"
    inner_logger(f"Will Log in {filename}")
    logging.basicConfig(
            level="NOTSET",
            format=FORMAT,
            datefmt='%Y-%m-%d %H:%M:%S',
            filename=filename
    )
    log = logging.getLogger("rich")
    log.addHandler(RichHandler())
    return log


def move_dgl_to_cuda(g, device='cuda'):
    g.ndata.update({k: g.ndata[k].to(device) for k in g.ndata})
    g.edata.update({k: g.edata[k].to(device) for k in g.edata})
    return g


def get_sentence_frame_acc(intent_preds, intent_labels,
                           slot_preds, slot_labels):
    """For the cases that intent and all the slots are correct (in one sentence)"""
    # Get the intent comparison result
    intent_result = (intent_preds == intent_labels)

    # Get the slot comparision result
    slot_result = []
    for preds, labels in zip(slot_preds, slot_labels):
        assert len(preds) == len(labels)
        one_sent_result = True
        for p, l in zip(preds, labels):
            if p != l:
                one_sent_result = False
                break
        slot_result.append(one_sent_result)
    slot_result = np.array(slot_result)

    sementic_acc = np.multiply(intent_result, slot_result).mean()
    return {
        "sementic_frame_acc": sementic_acc
    }


def get_slot_metrics(preds, labels):
    assert len(preds) == len(labels)
    return {
        "slot_precision": precision_score(labels, preds),
        "slot_recall":    recall_score(labels, preds),
        "slot_f1":        f1_score(labels, preds)
    }


def make_state_graph(turn: int, device='cpu'):
    src, dst = [], []
    for i in range(turn):
        for j in range(i):
            src.append(j)
            dst.append(i)
            src.append(i)
            dst.append(j)
        src.append(i)
        dst.append(i)
    g = dgl.graph((torch.tensor(src), torch.tensor(dst)), num_nodes=turn)
    return g.to(device)
