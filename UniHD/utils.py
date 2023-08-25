import dataclasses
import logging
import math
import os
import io
import sys
import time
import json
from typing import Optional, Sequence, Union

import tqdm
import copy

def _make_w_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f_dirname = os.path.dirname(f)
        if f_dirname != "":
            os.makedirs(f_dirname, exist_ok=True)
        f = open(f, mode=mode)
    return f


def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f


def jdump(obj, f, mode="w", indent=4, default=str):
    """Dump a str or dictionary to a file in json format.

    Args:
        obj: An object to be written.
        f: A string path to the location on disk.
        mode: Mode for opening the file.
        indent: Indent for storing json dictionaries.
        default: A function to handle non-serializable entries; defaults to `str`.
    """
    f = _make_w_io_base(f, mode)
    if isinstance(obj, (dict, list)):
        json.dump(obj, f, indent=indent, default=default)
    elif isinstance(obj, str):
        f.write(obj)
    else:
        raise ValueError(f"Unexpected type: {type(obj)}")
    f.close()


def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict


def read_tsar_dataset(data_path, train=True, is_label=True):
    '''
    Format:  
    '''
    sentences=[]
    mask_words = []
    mask_labels = []
    id = 0

    with open(data_path, "r", encoding='utf-8') as reader:
        while True:
            line = reader.readline()
            if is_label:
                id += 1
                if not line:
                    break
                sentence,words = line.strip().split('\t',1)
                mask_word,labels = words.strip().split('\t',1)
                label = labels.split('\t')
                
                sentences.append(sentence)
                mask_words.append(mask_word)
                
                one_labels = []
                for la in label:
                    if la.startswith(" "):
                        la=la[1:]
                    if la.startswith(" "):
                        la=la[1:]            
                    if la.endswith(" "):
                        la=la[:-1]
                    if la.endswith(" "):
                        la=la[:-1]
                    if la not in one_labels and la!="" and la!=mask_word:
                        one_labels.append(la)
                    
                mask_labels.append(one_labels)
            else:
                if not line:
                    break
                sentence,mask_word = line.strip().split('\t')
                sentences.append(sentence)
                mask_words.append(mask_word)

        list_data_dict = []
        for index in range(len(sentences)):
            list_data_dict.append(
                {'sentence': sentences[index], 'mask_word': mask_words[index], 'mask_label': mask_labels[index]}
            )
            
    if train:
        return list_data_dict
    else:
        return sentences, mask_words, mask_labels


def read_lexmturk_dataset(data_path, is_label=True):
    sentences=[]
    mask_words = []
    mask_labels = []
    id = 0

    with open(data_path, "r", encoding='ISO-8859-1') as reader:
        while True:
            line = reader.readline()
            if is_label:
                id += 1
                if id==1:
                    continue
                if not line:
                    break
                sentence,words = line.strip().split('\t',1)
                mask_word,labels = words.strip().split('\t',1)
                label = labels.split('\t')
                
                sentences.append(sentence)
                mask_words.append(mask_word)
                
                freq_dict = {}
                for word in label:
                    if word not in freq_dict:
                        freq_dict[word] = 1
                    else:
                        freq_dict[word] += 1
                        
                sorted_freq_dict = dict(sorted(freq_dict.items(), key=lambda item: item[1], reverse=True))

                mask_labels.append(list(sorted_freq_dict.keys()))
                
            else:
                if not line:
                    break
                sentence,mask_word = line.strip().split('\t')
                sentences.append(sentence)
                mask_words.append(mask_word)

    return sentences, mask_words, mask_labels


def read_NNSeval_BenchLS_dataset(data_path, is_label=True):
    sentences=[]
    mask_words = []
    mask_labels = []
    id = 0

    with open(data_path, "r", encoding='ISO-8859-1') as reader:
        while True:
            line = reader.readline()
            if is_label:
                id += 1
                if id==1:
                    continue
                if not line:
                    break
                sentence,words = line.strip().split('\t',1)
                mask_word,labels = words.strip().split('\t',1)
                label = labels.split('\t')
                
                sentences.append(sentence)
                mask_words.append(mask_word)
                
                for j in range(len(label)-1):
                    label[j] = label[1:][j].split(":")[1]
                    
                label = label[:-1]                

                mask_labels.append(label)

            else:
                if not line:
                    break
                sentence,mask_word = line.strip().split('\t')
                sentences.append(sentence)
                mask_words.append(mask_word)
            

    return sentences, mask_words, mask_labels



    
