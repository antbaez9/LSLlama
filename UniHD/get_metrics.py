import copy
import utils

# select dataset
# read_lexmturk_dataset can be used to read results files

data_path = "lexmturk_results.txt"
model_labels = utils.read_lexmturk_dataset(data_path)[2]
labels = utils.read_lexmturk_dataset("./datasets/lex.mturk.txt")[2]

# data_path = "NNSeval_results.txt"
# model_labels = utils.read_lexmturk_dataset(data_path)[2]
# labels = utils.read_NNSeval_BenchLS_dataset("./datasets/NNSeval.txt")[2]

# data_path = "BenchLS_results.txt"
# model_labels = utils.read_lexmturk_dataset(data_path)[2]
# labels = utils.read_NNSeval_BenchLS_dataset("./datasets/BenchLS.txt")[2]

round_to = 4

# ACC@1

def calc_acc_at_1(model_labels, gold_mask_labels):

    total_instances = len(gold_mask_labels)
    
    matches = 0
    for instance_index in range(total_instances):
        top_model_label = model_labels[instance_index][0]
        if top_model_label in gold_mask_labels[instance_index]:
            matches += 1
    
    print(matches, total_instances)
    return round(matches/total_instances, round_to)

# POTENTIAL@k

def calc_potential_at_k(model_labels, gold_mask_labels, k):

    total_instances = len(gold_mask_labels)
    
    matches = 0
    for instance_index in range(total_instances):
        for model_label in model_labels[instance_index][:k]:
            if model_label in gold_mask_labels[instance_index]:
                matches += 1
                break

    return round(matches/total_instances, round_to)

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
    # print(matches, total_instances)

    return round(matches/total_instances, round_to)
   
# MAP@k

def calc_map_at_k(model_labels, gold_mask_labels, k):
    
    total_instances = len(gold_mask_labels)
    
    adjusted_ap_list = []
    for instance_index in range(total_instances):
        match_list = []
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

    return round(MAP, round_to)

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
    
    