# training/pretraining related to mm model



# layout

- `scripts`
    - should have 2 versions of train
        - 1 for single-gpu where i can ensure dataset/model/etc is working as expected and for easier debugging
        - 1 for fsdp that is simple and based mostly off of finetune llama
- `src/config`
    - configs/model-init/etc for local/eng dev vs borah
- `src/pretrain_mm`
    - `datasets`
        - not clear to me how much data i need yet and how much data from others is actually all that useful
        - from most useful to least is probably mind2web, silatus, common_scenes,
        - mind2web i can make the most similar to what eventual mm agent will be like
    - `distributed`
        - fsdp related, keep as much out of the main scripts as possible as makes debugging much harder
    - `processor`
        - need to have more customizable way to do image+text tokenization/vectorizing for later incontext tasks


# data

- keeping all the data on eng402001 as have plenty of space there i know it wont get auto-deleted like the cluster


- mind2web
    - need to use globus, it was a bit of an ordeal to get the transfer
    - once have it somewhere i can stage/dev, to transfer to borah:
        - `scp -r /data/graham/datasets/mind2web/data borah:/bsuhome/gannett/scratch/datasets/mind2web`



# tests

each dataset should have a test written to make sure they are uniform.  use fixtures/stubs from config


# transfo
Name: transformers
Version: 4.40.1
Summary: State-of-the-art Machine Learning for JAX, PyTorch and TensorFlow
Home-page: https://github.com/huggingface/transformers
Author: The Hugging Face team (past and future) with the help of all our contributors (https://github.com/huggingface/transformers/graphs/contributors)
Author-email: transformers@huggingface.co
License: Apache 2.0 License
Location: /home/graham/mambaforge/envs/tr/lib/python3.11/site-packages
Requires: filelock, huggingface-hub, numpy, packaging, pyyaml, regex, requests, safetensors, tokenizers, tqdm
Required-by: collie-lm, llava, mosaicml-streaming, peft, pretrain_mm, repeng, surya-ocr, trl
