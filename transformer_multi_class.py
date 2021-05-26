# -*- coding: utf-8 -*-
from transformers import TFBertForSequenceClassification, BertTokenizer
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
import json

max_length = 256
batch_size = 32
learning_rate = 2e-5
number_of_epochs = 30
num_classes = 5

# model_path = "chinese-electra-180g-small-discriminator"
# tokenizer = BertTokenizer.from_pretrained(model_path)
# model_name = 'chinese-electra'

model_path = "bert-base-chinese"
tokenizer = BertTokenizer.from_pretrained(model_path)
model_name = 'bert-base-chinese'


def split_dataset(df):
    train_set, x = train_test_split(df, stratify=df['new_label'], test_size=0.3, random_state=42)
    eval_set, test_set = train_test_split(x, stratify=x['new_label'], test_size=0.5, random_state=43)
    return train_set, eval_set, test_set


def data_to_dict(input_ids, attention_masks, token_type_ids, label):
    return {
               "input_ids": input_ids,
               "token_type_ids": token_type_ids,
               "attention_mask": attention_masks,
           }, label


def data_generator(df_data):
    input_ids_list = []
    token_type_ids_list = []
    attention_mask_list = []
    label_list = []
    for index, row in df_data.iterrows():
        text = str(row["text"])
        label = row["new_label"]
        bert_input = tokenizer.encode_plus(text, add_special_tokens=True, max_length=max_length,
                                           pad_to_max_length=True, return_attention_mask=True, truncation=True)
        input_ids_list.append(bert_input.get('input_ids'))
        token_type_ids_list.append(bert_input.get('token_type_ids'))
        attention_mask_list.append(bert_input.get('attention_mask'))
        label_list.append([label])

    dataset = tf.data.Dataset.from_tensor_slices((input_ids_list, attention_mask_list, token_type_ids_list, label_list))
    dataset = dataset.map(data_to_dict)
    dataset = dataset.shuffle(10000).batch(batch_size)
    return dataset

def get_train_data(df_raw):
    train_data, eval_data, test_data = split_dataset(df_raw)
    train_encoded = data_generator(train_data)
    eval_encoded = data_generator(eval_data)
    test_encoded = data_generator(test_data)
    return train_encoded, eval_encoded, test_encoded


def standard_label(df=None):
    key_list = [i for i in df["new_label"].tolist()]
    value_list = df["label"].tolist()
    with open("{}-label.json".format(model_name), "w", encoding="utf-8") as f:
        f.write(json.dumps(dict(zip(key_list, value_list)), ensure_ascii=False, indent=2))


def train_model(df_data):
    train_dataset, eval_dataset, test_dataset = get_train_data(df_data)
    gpu_len = len(tf.config.experimental.list_physical_devices('GPU'))
    print("gpu_len:" + str(gpu_len))
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = TFBertForSequenceClassification.from_pretrained(model_path, num_labels=num_classes)
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
        model.compile(optimizer=optimizer, loss=loss, metrics=[metric])
        model.summary()

    model.fit(train_dataset, epochs=number_of_epochs, validation_data=eval_dataset)
    print("evaluate test_set:", model.evaluate(test_dataset))
    model.save_weights('{}-model.h5'.format(model_name))


if __name__ == '__main__':
    data = pd.read_excel(r'data/data.xlsx')
    label_list = list(set(data["label"].tolist()))
    df_label = pd.DataFrame({"label": label_list, "new_label": list(range(len(label_list)))})
    standard_label(df_label)
    df_data = pd.merge(data, df_label, on="label", how="left")
    df_data = df_data.sample(frac=1)
    train_model(df_data)
