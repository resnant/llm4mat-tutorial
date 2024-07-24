import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

# set hugging face token to access gated model
with open('/workspace/api_keys/hf_token.txt', mode="r") as f:
    os.environ["HF_TOKEN"] = f.read()
    os.environ["HUGGING_FACE_HUB_TOKEN"] = f.read()

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import glob
import pickle
from tqdm import tqdm
from multiprocessing import Pool
import transformers
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
from trl import SFTTrainer
from peft import LoraConfig


def load_mp_pickles(root_dir):
    pkl_files = glob.glob(f"{root_dir}/*.pkl")
    print(f"found files: {pkl_files}")
    merged_list = []

    for file in tqdm(pkl_files, desc="Merging pickle files"):
        with open(file, 'rb') as f:
            data = pickle.load(f)
            merged_list += data
    print(f"loaded entries: {len(merged_list)}")

    return merged_list

def get_structure_info_text(structure):
    structure_info_text = [line for line in str(structure).split("\n")[1:] if not line.startswith('-')]
    structure_info_text.insert(4, f"space group: {structure.get_space_group_info()}")
    
    return "\n".join(structure_info_text)

def prepare_datasets(mp_materials, target_prop="band_gap"):
    structures = [mat["structure"] for mat in tqdm(mp_materials)]
    targets_values =[mat[target_prop] for mat in tqdm(mp_materials)]

    # pre-processing structure data in parallel to speedup
    with Pool(processes=None) as pool:  # use all cpu cores
        structure_text = list(tqdm(pool.imap(get_structure_info_text, structures, chunksize=1000), total=len(structures)))

    dataset_dict = {"target": targets_values, "structure": structure_text}
    dataset = Dataset.from_dict(dataset_dict)

    train_test_split_ratio = 0.1
    test_size = int(len(dataset) * train_test_split_ratio)
    return dataset.train_test_split(test_size=test_size, seed=42)

def setup_tokenizer_and_model(model_id:str):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    model = AutoModelForCausalLM.from_pretrained(model_id,
                                                 device_map="auto",
                                                 torch_dtype=torch.bfloat16,
                                                 )
    return tokenizer, model
    
def formatting_prompts_func(example):
    output_texts = []
    for target_value, structure_text in zip(example['target'], example['structure']):
        text = f"Instruction: What is the bandgap value of following material?:\n{structure_text}\n\nOutput:\n{target_value}{tokenizer.eos_token}"
        output_texts.append(text)
    return output_texts

def setup_peft_config(use_lora=False):
    if use_lora:
        peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        )
    else:
        peft_config = None
    return peft_config


if __name__ == "__main__":
    ### 
    # modify here if needed
    mp_data = load_mp_pickles("./mp_download")    
    
    # target property choices: [band_gap, energy_per_atom, formation_energy_per_atom, energy_above_hull, total_magnetization_normalized_vol]
    dataset_split = prepare_datasets(mp_data, "band_gap")
    # model_id = "meta-llama/Meta-Llama-3-8B"
    # model_id = "mistralai/Mistral-7B-v0.3"
    model_id = "mistralai/Mistral-Nemo-Base-2407"
    ###

    tokenizer, model = setup_tokenizer_and_model(model_id = model_id)

    training_args = transformers.TrainingArguments(
        num_train_epochs=1,
        learning_rate=1.0e-5,
        per_device_train_batch_size=1,
        bf16=True,
        logging_steps=50,
        save_total_limit=3, 
        save_steps=2000,
        output_dir=f"outputs/{model_id.replace('/', '_').replace('.', '_')}_full",
    )

    trainer = SFTTrainer(
        args=training_args,
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset_split["train"],
        max_seq_length=2048,
        packing=False,
        num_of_sequences=1024,
        formatting_func=formatting_prompts_func,
        peft_config=setup_peft_config(use_lora=False)
    )

    trainer.train()
    trainer.save_model()
