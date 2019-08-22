from functools import partial
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import re

from fonduer import Meta
from fonduer.parser.models import Document
from emmental.data import EmmentalDataLoader, EmmentalDataset
from emmental.scorer import Scorer
from emmental.task import EmmentalTask

class CNN_Text(nn.Module):
    def __init__(self, embed_num, embed_dim, widths=[3, 4, 5], filters=100):
        super(CNN_Text, self).__init__()
        Ci = 1
        Co = filters
        h = embed_dim
        self.embed = nn.Embedding(embed_num, h)
        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (w, h)) for w in widths])

    def forward(self, x):
        # x is (batch, len)
        x = self.embed(x)
        x = x.unsqueeze(1)  # (batch, Ci, len, d)
        x = [
            F.relu(conv(x)).squeeze(3) for conv in self.convs1
        ]  # [(batch, Co, len), ...]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N,Co), ...]
        x = torch.cat(x, 1)
        return x

def ce_loss(task_name, immediate_ouput_dict, Y, active):
    module_name = f"{task_name}_pred_head"
    return F.cross_entropy(
        immediate_ouput_dict[module_name][0][active], (Y.view(-1) - 1)[active]
    )


def output(task_name, immediate_ouput_dict):
    module_name = f"{task_name}_pred_head"
    return immediate_ouput_dict[module_name][0]


def get_task(task_names, emb_dim, char_dict_size):

    cnn_module = CNN_Text(char_dict_size, emb_dim)
    cnn_out_dim = 300 # TODO: Get rid of this hardcode
    
    tasks = []

    for task_name in task_names:
        task = EmmentalTask(
            name=task_name,
            module_pool=nn.ModuleDict(
                {
                    "cnn_text": cnn_module, 
                    f"{task_name}_pred_head": nn.Linear(cnn_out_dim,2)
                }
            ),
            task_flow=[
                {
                    "name": "cnn_text",
                    "module": "cnn_text",
                    "inputs": [("_input_", "emb")],
                },
                {
                    "name": f"{task_name}_pred_head",
                    "module": f"{task_name}_pred_head",
                    "inputs": [("cnn_text", 0)],
                },
            ],
            loss_func=partial(ce_loss, task_name),
            output_func=partial(output, task_name),
            scorer=Scorer(metrics=["accuracy", "f1", "precision", "recall"]),
        )
        tasks.append(task)

    return tasks

def fix_spacing(text):
    """
    Removes double spaces and spaces at the beginning and end of line
    string text: text to fix spacing on
    """

    while "  " in text:
        text = text.replace("  ", " ")
    text = text.strip()

    return text


def clean_input(text):
    """
    Removes quotes, html tags, etc
    string text: string to modify
    """

    # Strip special characters
    text = (
        ("".join(c for c in text if ord(c) > 31 and ord(c) < 127))
        .encode("ascii", "ignore")
        .decode()
    )

    # Strip html tags
    text = re.sub(r"<.*?>", " ", text)
    # Strip html symbols
    text = re.sub("&.{,6}?;", " ", text)
    # Strip ascii interpretation of special characters
    text = re.sub(r"\\x[a-zA-Z0-9][a-zA-Z0-9]", " ", text)

    # String literal "\n" and other such characters are in the text strings
    text = text.replace("b'", "")
    text = text.replace("\\'", "").replace("'", "").replace('\\"', "").replace('"', "")
    text = (
        text.replace("\\n", " ")
        .replace("\\r", " ")
        .replace("\\t", " ")
        .replace("\\\\\\", "")
    )
    text = text.replace("{", " ").replace("}", " ").replace(";", "")

    text = fix_spacing(text)

    return text


def get_posting_html_fast(text, search_term):
    """
    Returns ad posting from html document in memex_raw_data
    string text: memex_raw_data string
    term: regex of term to find
    """
    title_term = r"<[Tt]itle>([\w\W]*?)</[Tt]itle>"
    body_term = r"<div.{0,20}[Cc]ontent.{0,20}>([\w\W]*?)</div>"
    body_term2 = r"<div.{0,20}[Pp]ost.{0,20}>([\w\W]*?)</div>"
    body_term3 = r"<div.{0,20}[Tt]ext.{0,20}>([\w\W]*?)</div>"
    body_term4 = r"<p>([\w\W]*?)</p>"

    title = re.search(title_term, text)
    body_lines = (
        re.findall(body_term, text)
        + re.findall(body_term2, text)
        + re.findall(body_term3, text)
        + re.findall(body_term4, text)
    )
    html_lines = [clean_input(line) for line in body_lines]

    if title and title.group(1):
        title = clean_input(title.group(1))
    else:
        title = "-1"

    html_text = "Title " + title.replace(".", " ").replace(":", " ") + " <|> "

    for line in html_lines:
        if line:
            html_text += " " + line + " <|> "

    if search_term:
        search_lines = [clean_input(line) for line in re.findall(search_term, text)]
        for line in search_lines:
            if line:
                html_text += (
                    " Search " + line.replace(".", " ").replace(":", " ") + " <|> "
                )

    html_text = fix_spacing(html_text)

    return html_text


class SymbolTable(object):
    """Wrapper for dict to encode unknown symbols
    :param starting_symbol: Starting index of symbol.
    :type starting_symbol: int
    :param unknown_symbol: Index of unknown symbol.
    :type unknown_symbol: int
    """

    def __init__(self, starting_symbol=2, unknown_symbol=1):
        self.s = starting_symbol
        self.unknown = unknown_symbol
        self.d = dict()

    def get(self, w):
        if w not in self.d:
            self.d[w] = self.s
            self.s += 1
        return self.d[w]

    def lookup(self, w):
        return self.d.get(w, self.unknown)

    def lookup_strict(self, w):
        return self.d.get(w)

    def len(self):
        return self.s

    def reverse(self):
        return {v: k for k, v in self.d.items()}

def load_data_from_db(postgres_db_name, postgres_db_location, label_dict, char_dict=None, clobber_label=True):
    """Load data from database.
    """

    print(f"Loading data from db {postgres_db_name}")
    # Start DB connection
    conn_string = os.path.join(postgres_db_location, postgres_db_name)
    session = Meta.init(conn_string).Session()

    # Printing number of docs/sentences
    print("==============================")
    print(f"DB contents for {postgres_db_name}:")
    print(f"Number of documents: {session.query(Document).count()}")
    print("==============================")

    docs = session.query(Document).all()

    uid_field = []
    text_field = []
    label_field = []
    missed_ids = 0

    term = r"([Ll]ocation:[\w\W]{1,200}</.{0,20}>|\W[cC]ity:[\w\W]{1,200}</.{0,20}>|\d\dyo\W|\d\d.{0,10}\Wyo\W|\d\d.{0,10}\Wold\W|\d\d.{0,10}\Wyoung\W|\Wage\W.{0,10}\d\d)"
    
    for doc in docs:
        if (doc.name in label_dict) or clobber_label:
            uid_field.append(doc.name)
            text_field.append(get_posting_html_fast(doc.text, term))
            if not clobber_label:
                label_field.append(label_dict[doc.name])
            else:
                label_field.append(-1)
        else:
            missed_ids += 1    

    # Printing data stats
    print("==============================")
    print(f"Loaded {len(uid_field)} ids")
    print(f"Loaded {len(text_field)} text")
    print(f"Loaded {len(label_field)} labels")
    print(f"Missed {missed_ids} samples")

    X_dict = {"text": text_field, "uid": uid_field}
    Y_dict = {"label": torch.from_numpy(np.array(label_field))}
    
    dataset = EmmentalDataset(name="HT", X_dict=X_dict, Y_dict=Y_dict, uid="uid")
        
    emb_field = []
    for i in range(len(dataset)):
        emb_field.append(torch.from_numpy(np.array(list(map(char_dict.lookup, dataset[i][0]['text'])))))
    dataset.add_features({"emb": emb_field})
    return dataset
