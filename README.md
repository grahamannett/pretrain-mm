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




# tests

each dataset should have a test written to make sure they are uniform.  use fixtures/stubs from config



