FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-devel

# Set environment variables
ENV HF_HOME="/root/.cache/huggingface/"
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# Install required libraries
RUN pip install PyYAML>=6.0 datasets>=2.12.0 einops>=0.6.1 evaluate>=0.4.0 protobuf>=4.23.3 scikit-learn>=1.2.2 scipy>=1.10.1 sentencepiece>=0.1.99 fire>=0.5.0 jupyterlab tensorboard tensorflow accelerate transformers peft deepspeed bitsandbytes trl matplotlib
RUN pip install mp_api tf-keras pymatgen
COPY jupyter_notebook_config.py /root/.jupyter/jupyter_notebook_config.py
COPY jupyter_notebook_config.py /root/.jupyter/jupyter_lab_config.py