# llm4mat-tutorial

## 環境構築

### Dockerの場合（ローカル）
```bash
docker build -t mi_llm:v0.3 ./docker/
```

## 実験実行
### Dockerの起動
```bash
mkdir hf_cache # for hugging face model cache dir
docker run --rm --gpus all -it --shm-size=200g -v $PWD:/workspace -v hf_cache:/root/.cache/huggingface/ mi_llm:v0.3 bash
```

### Materials Project（MP）データセット準備
- Materials Projectからデータをダウンロードする
- MPからAPI keyを取得し、`api_keys/MP_API_KEY.txt` に保存
    - API key生成は https://next-gen.materialsproject.org/api を参照

- 以下のスクリプトを実行すると、`mp_download/20240718_0000.pkl` のように連番のpklファイルができる
```bash
python download_mp_data.py
```
- これでデータ準備は完了


### LLMのファインチューニング
- Hugging Faceのtokenを生成し、`api_keys/hf_token.txt` に配置
    - tokenは https://huggingface.co/settings/tokens からダウンロードできる

- 結晶構造から物性値（bandgap他）を予測する例：
```bash
python train_structure2property.py
```

### 分析用にJupyterLabを起動する場合

```bash
jupyter lab --allow-root --ip=0.0.0.0
```

## モデルの推論
- バンドギャップ予測用にファインチューニングしたモデル：
    - `https://huggingface.co/ysuz/Mistral-Nemo-Base-2407-bandgap`

- 使い方
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