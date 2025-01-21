# 1. Data Preparation
```
cd datasets
python setup_datasets.py
python wizard_handle.py # we use alpaca data format so you should run transform script first.
python dolly_handle.py
```
# 2. train
We use LLaMA_Factory to perform finetuning process. Go to [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory.git) for details.
```
llamafactory-cli train train_yamls/example.yaml 
```
# 3. infer
Infer sft datasets using 1st and 2nd epoch checkpoints.
```
python inf1.py
```
# 4. RM scoring
Score the inference result generated in step 3.
```
python score1.py
```
# 5. get inspire data
```
python calc_inspire_data.py
```