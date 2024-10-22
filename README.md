# llm4mat-tutorial

このリポジトリは[DxMT AIMHack 2024](https://dxmt.mext.go.jp/news/1105) での講演資料・ハンズオン用コードです。  

この講演では、LLM（大規模言語モデル）の基礎と材料科学における応用例について、ハンズオン形式で学ぶことを目的としています。
これらの資料が、LLMを使うための現代的なソフトウェアスタックへの入門と、材料科学におけるLLMの可能性を探るための助けとなれば幸いです。

## リポジトリの概要
- [lecture_slide/DxMT20240725_LLM.pdf](lecture_slide/DxMT20240725_LLM.pdf)
    - 講義スライド
- [notebooks/paper_keyword_generation.ipynb](notebooks/paper_keyword_generation.ipynb)
    - Gemini (Google製LLM)を使って論文タイトルや要旨からキーワードを生成する例
- [notebooks/run_inference_and_eval.ipynb](notebooks/run_inference_and_eval.ipynb)
    - 結晶構造からバンドギャップを予測するためにファインチューニングしたLLMを動かして推論と評価を行う例
- [download_mp_data.py](download_mp_data.py)
    - Materials Project（MP）から学習用データをダウンロードするコード
- [train_structure2property.py](train_structure2property.py)
    - MPのデータでLLMをファインチューニングするためのコード

## Model Inference
- fine-tuned model for bandgap prediction:
    - https://huggingface.co/ysuz/Mistral-Nemo-Base-2407-bandgap
    - The result of `train_structure2property.py`
- Please refer example:[notebooks/run_inference_and_eval.ipynb](https://github.com/resnant/llm4mat-tutorial/blob/main/notebooks/run_inference_and_eval.ipynb)

- Minimum code:
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


model_id = "ysuz/Mistral-Nemo-Base-2407-bandgap"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id,
                                             device_map="auto",
                                             torch_dtype=torch.float16,
                                            )

# example of input context
structure_text = """
Reduced Formula: BaSrI4
abc   :   5.807091   5.807091   8.251028
angles:  90.000000  90.000000  90.000000
pbc   :       True       True       True
space group: ('P4/mmm', 123)
Sites (6)
  #  SP      a    b         c    magmom
  0  Ba    0.5  0.5  0               -0
  1  Sr    0    0    0.5             -0
  2  I     0    0.5  0.257945         0
  3  I     0.5  0    0.257945         0
  4  I     0    0.5  0.742055         0
  5  I     0.5  0    0.742055         0

Output:
"""

prompt = f"Instruction: What is the bandgap value of following material?:\n{structure_text}\n\nOutput:\n"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    tokens = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.5,
        top_p=0.9,
        repetition_penalty=1.05,
    )
generated_text = tokenizer.decode(tokens[0], skip_special_tokens=True)
print(f"Generated raw text:\n{generated_text}\n\n")
```


## Environment Setup

### Docker build
```bash
docker build -t mi_llm:v0.3 ./docker/
```

## Running Experiments
### Docker startup
```bash
mkdir hf_cache # for hugging face model cache dir
docker run --rm --gpus all -it --shm-size=200g -v $PWD:/workspace -v hf_cache:/root/.cache/huggingface/ mi_llm:v0.3 bash
```

### Preparing the Materials Project (MP) Dataset
- Download data from the Materials Project
- Obtain an API key from MP and save it in `api_keys/MP_API_KEY.txt`
    - Refer to https://next-gen.materialsproject.org/api to generate an API key

- Running the following script will create sequentially numbered pkl files like`mp_download/20240718_0000.pkl`
```bash
python download_mp_data.py
```
- dataset preparation completed


### Fine-tuning LLMs
- Generate a token from Hugging Face and place it in `api_keys/hf_token.txt`
    - The token can be obtained from https://huggingface.co/settings/tokens

- Example of predicting physical properties (bandgap, etc.) from crystal structure:
```bash
python train_structure2property.py
```

### Start JupyterLab for analysis

```bash
jupyter lab --allow-root --ip=0.0.0.0
```