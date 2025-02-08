
models=['gemma2-2b','llama2-7b','llama2-13b']
model_template_dict = {
    'gemma2-2b': 'gemma',
    'llama2-7b': 'llama2',
    'llama2-13b': 'llama2'
}
datasets=['wizard','dolly']
model=models[2]
dataset=datasets[0]
merge_path=f"../lora_ckp/merged/{model}_{dataset}"
model_paths=[merge_path+"1epo",merge_path+"2epo",merge_path]

data_path=f"../datasets/sft/{dataset}.json"
houzhui=["1epo","2epo",""]

import pandas as pd
import json
import os

score_files = [f"../score_res/{model}{h}_{dataset}.pkl" for h in houzhui]
# score_files = [f"../score_res/fft/{model}{h}_{dataset}.pkl" for h in houzhui]
# score_files = [f"../score_res/lora/rmbert/{model}{h}_{dataset}.pkl" for h in houzhui]
print(f"score_files:{score_files}")
data = {}
for i, file in enumerate(score_files):
    if os.path.exists(file):
        df = pd.read_pickle(file)
        if isinstance(df, list):
            df = pd.DataFrame(df)
        for index, row in df.iterrows():
            idx = row['Index']
            score = row['res得分']
            if idx not in data:
                data[idx] = {}
            data[idx][houzhui[i]] = score
            print(f"data[{idx}][{houzhui[i]}] = {score}")

score_diff = []
for idx, scores in data.items():
    score_1epo = scores.get("1epo", -float('inf'))
    score_2epo = scores.get("2epo", -float('inf'))
    diff = score_2epo - score_1epo
    score_diff.append([idx, diff])

score_diff_df = pd.DataFrame(score_diff, columns=["Index", "score_diff"])

score_diff_df = score_diff_df.sort_values(by="score_diff", ascending=True)
print(f"score_diff_df:{score_diff_df}")

delete_counts = [8000, 12000, 20000,24000,32000,40000,48000,60000] 

with open(f"../datasets/sft/{dataset}.json", "r") as f:
    original_data = json.load(f)

for delete_count in delete_counts:
    remaining_indices = score_diff_df["Index"].iloc[delete_count:].tolist()
    # print("//////////////////////////////////////////")
    remain_diff=score_diff_df["score_diff"].iloc[delete_count:].tolist()
    # print(f"remain len:{len(remain_diff)}")
    # print(f"remain diffs:{remain_diff}")
    
    remaining_data = [original_data[int(idx)] for idx in remaining_indices]
    
    output_file = f"../datasets/sft/{dataset}_{model}del{delete_count}.json"
    # output_file = f"../datasets/sft/fft_{dataset}_{model}del{delete_count}.json"
    # output_file = f"../datasets/sft/rmbert_{dataset}_{model}del{delete_count}.json"
    with open(output_file, "w") as f:
        json.dump(remaining_data, f, indent=4)
    
    print(f"{output_file} line count: {len(remaining_data)}")

# 打印结果
print("Inspire data has been generated!")
