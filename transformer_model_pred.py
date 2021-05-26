# -*- coding: utf-8 -*-
# author: LZY
# TIME: 2021.3.9
import json
import numpy as np
from transformers import TFBertForSequenceClassification, BertTokenizer
from sklearn.metrics import classification_report
import time
import pandas as pd

max_length = 256
num_classes = 5

# model_path = "chinese-electra-180g-small-discriminator"
# tokenizer = BertTokenizer.from_pretrained(model_path)
# model_name = 'chinese-electra'

model_path = "bert-base-chinese"
tokenizer = BertTokenizer.from_pretrained(model_path)
model_name = 'bert-base-chinese'


model = TFBertForSequenceClassification.from_pretrained(model_path, num_labels=num_classes)
model.load_weights('{}-model.h5'.format(model_name))

with open("{}-label.json".format(model_name), "r", encoding="utf-8") as f:
    labels = json.loads(f.read())


def pre(text):
    w = time.time()
    bert_input = tokenizer.encode_plus(text, add_special_tokens=True, max_length=max_length,
                                       pad_to_max_length=True, return_attention_mask=True, truncation=True)
    input_ids_list = [bert_input['input_ids']]
    token_type_ids_list = [bert_input['token_type_ids']]
    attention_mask_list = [bert_input['attention_mask']]
    test_ds = [np.array(input_ids_list), np.array(attention_mask_list), np.array(token_type_ids_list)]
    int_label = np.argmax(model.predict(test_ds).logits)
    label = labels.get(str(int_label))
    e = time.time()
    print(e - w)
    print(label)


def evaluate_report(df_data):
    # 根据values找到key  list(d.keys())[list(d.values()).index("#你要索引的value")]  转成list后直接索引（value值必须唯一）
    true_y_list = [int(list(labels.keys())[list(labels.values()).index(i)]) for i in df_data["label"].tolist()]
    pred_y_list = []
    for index, row in df_data.iterrows():
        text = str(row["text"])
        bert_input = tokenizer.encode_plus(text, add_special_tokens=True, max_length=max_length,
                                           pad_to_max_length=True, return_attention_mask=True, truncation=True)
        input_ids_list = [bert_input['input_ids']]
        token_type_ids_list = [bert_input['token_type_ids']]
        attention_mask_list = [bert_input['attention_mask']]
        test_ds = [np.array(input_ids_list), np.array(attention_mask_list), np.array(token_type_ids_list)]
        pre_label = np.argmax(model.predict(test_ds).logits)
        pred_y_list.append(pre_label)

    target_name_list = list(labels.values())
    report = classification_report(true_y_list, pred_y_list, target_names=target_name_list, digits=4, output_dict=True)
    print(report)
    df = pd.DataFrame(report).transpose()
    df.to_csv("{}-report.csv".format(model_name), encoding='utf_8_sig', index=True)


if __name__ == '__main__':
    data = pd.read_excel(r'data/data.xlsx')
    evaluate_report(data)
