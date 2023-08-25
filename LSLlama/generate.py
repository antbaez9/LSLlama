import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence
import ast

import torch
import transformers
import utils
from torch.utils.data import Dataset
from transformers import Trainer, GenerationConfig
from transformers.generation.utils import GreedySearchDecoderOnlyOutput
import textwrap

eval_data_path = "./eval/tsar2022_en_test_gold.tsv"
gold_sentences, gold_mask_words, gold_mask_labels = utils.read_tsar_dataset(eval_data_path, train=False)

# choose start and end indeces
start, end = 0, 500
sentences = gold_sentences[start:end]
words = gold_mask_words[start:end]
labels = gold_mask_labels[start:end]

# adjust inference hyperparameters
temperature = 0.1
top_p = 0.75
repetition_penalty = 1.11 # best = 1.11

print("temperature: ", temperature, "top_p: ", top_p, "repetition penalty: ", repetition_penalty)

tokenizer = transformers.AutoTokenizer.from_pretrained("./tuned-llama_7b", model_max_length=256, padding_side="right", use_fast=False)

model = transformers.AutoModelForCausalLM.from_pretrained("./tuned-llama-7b").to("cuda")

model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
model.config.bos_token_id = 1
model.config.eos_token_id = 2
    
model = model.eval()

def create_prompt(input_sentence: str, input_mask_word: str) -> str:
    sentence = input_sentence
    mask_word = input_mask_word
    prompt = f"""
    Respond with a list of ten different, simpler synonyms of the complex word in the given context.
    
    ### Complex Word: {mask_word}
    
    ### Sentence: {sentence}
    
    ### Response:
    """
    return prompt


def generate_response(prompt: str, model: transformers.AutoModelForCausalLM, temp: float, top_p_val: float, rep: float) -> GreedySearchDecoderOnlyOutput:
    encoding = tokenizer(prompt, return_tensors="pt")
    input_ids = encoding["input_ids"].to("cuda")
    
    generation_config = GenerationConfig(
        temperature=temp,
        top_p=top_p_val, 
        repetition_penalty=rep, 
    )
    
    with torch.inference_mode():
        return model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=256,
        )
    
def format_response(response: GreedySearchDecoderOnlyOutput) -> str:
    decoded_output = tokenizer.decode(response.sequences[0])
    print("prompt", decoded_output)
    response = decoded_output.split("### Response:")[1].strip()
    # remove end of sentence token and handle empty strings
    if response == "</s>" or response == "":
        response = "[None]"
    return response

def ask_LSLlama(sentence: str, mask_word: str, model: transformers.AutoModelForCausalLM) -> str:
    prompt = create_prompt(sentence, mask_word)
    response = generate_response(prompt, model, temperature, top_p, repetition_penalty)
    formatted_response = format_response(response)
    # correct malformed strings formed with ' and "
    final_response = formatted_response.replace('"', "'")
    final_response = final_response.replace("''", "'")

    # handle special cases
    if "'coup d'etat', " in final_response:
        final_response = final_response.replace("'coup d'etat', ", "")
    if "'coup d'etat'" in final_response:
        final_response = final_response.replace("'coup d'etat'", "")            
    if "[" != final_response[0]:
        final_response = "[" + final_response

    # convert string into list
    return ast.literal_eval(final_response.replace("</s>", ""))

# go through eval dataset doing model inference
model_labels = []
for i in range(len(sentences)):
    print("index: ", i + start)
    model_label = ask_LSLlama(sentences[i], words[i], model)
    print("gold labels: ", labels[i])
    model_labels.append(model_label)
        

# ACC@1

def calc_acc_at_1(model_labels, gold_mask_labels):

    total_instances = len(gold_mask_labels)
    
    matches = 0
    for instance_index in range(total_instances):
        top_model_label = model_labels[instance_index][0]
        if top_model_label in gold_mask_labels[instance_index]:
            matches += 1
                
    return round(matches/total_instances, 4)

# POTENTIAL@k

def calc_potential_at_k(model_labels, gold_mask_labels, k):

    total_instances = len(gold_mask_labels)
    
    matches = 0
    for instance_index in range(total_instances):
        for model_label in model_labels[instance_index][:k]:
            if model_label in gold_mask_labels[instance_index]:
                matches += 1
                break
                
    return round(matches/total_instances, 4)

# ACCURACY@k@top1

def calc_accuracy_at_k_at_top1(model_labels, gold_mask_labels, n):

    total_instances = len(gold_mask_labels)
    
    matches = 0
    for instance_index in range(total_instances):
        for model_label in model_labels[instance_index][:n]:
            most_freq_gold_label = gold_mask_labels[instance_index][0]
            if model_label == most_freq_gold_label:
                matches += 1
                break
            
    return round(matches/total_instances, 4)
   
# MAP@k

def calc_map_at_k(model_labels, gold_mask_labels, k):
    
    total_instances = len(gold_mask_labels)
    
    adjusted_ap_list = []
    for instance_index in range(total_instances):
        model_labels_instance = model_labels[instance_index]
        gold_mask_labels_instance = gold_mask_labels[instance_index]
        
        relevance_list = []
        for label in model_labels_instance[:k]:
            if label in gold_mask_labels_instance:
                relevance_list.append(True)
            else:
                relevance_list.append(False)
    
        AP = 0
        true_positives = 0
        for i in range(len(relevance_list)):
            if relevance_list[i]:
                true_positives += 1
                precision = true_positives/(i+1)
                AP += precision
        adjusted_ap_list.append(AP/k)
    
    MAP = sum(adjusted_ap_list)/len(adjusted_ap_list)

    return round(MAP, 4)


print("METRICS:")
print("ACC@1:        ", calc_acc_at_1(model_labels, labels))
print("ACC@1@top1:   ", calc_accuracy_at_k_at_top1(model_labels, labels, 1))
print("ACC@2@top1:   ", calc_accuracy_at_k_at_top1(model_labels, labels, 2))
print("ACC@3@top1:   ", calc_accuracy_at_k_at_top1(model_labels, labels, 3))
print("MAP@3:        ", calc_map_at_k(model_labels, labels, 3))
print("MAP@5:        ", calc_map_at_k(model_labels, labels, 5))
print("MAP@10:       ", calc_map_at_k(model_labels, labels, 10))
print("Potential@3:  ", calc_potential_at_k(model_labels, labels, 3))
print("Potential@5:  ", calc_potential_at_k(model_labels, labels, 5))
print("Potential@10: ", calc_potential_at_k(model_labels, labels, 10))


        
        
        
        
        





