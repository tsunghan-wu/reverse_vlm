# We subsample the data to 100k samples for Qwen's finetuning.

import json
import random
fname = "./final_dataset.json"
# fname = "./llava_v1_5_mix665k.json"
with open(fname, "r") as f:
    data = json.load(f)

length = len(data)
print(f"Original length: {length}")
sampled_length = 100000
# subsample the data
sampled_data = random.sample(data, sampled_length)
output_fname = "./final_dataset_subsample.json"
# output_fname = "./llava_v1_5_mix665k_subsample.json"
with open(output_fname, "w") as f:
    json.dump(sampled_data, f)
print(f"Subsampled length: {len(sampled_data)}")