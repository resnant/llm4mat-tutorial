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
