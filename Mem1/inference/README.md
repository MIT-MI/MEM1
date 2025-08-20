## What does this directory do
- Distill SFT data
- Generate Rollout data for evaluation of different kinds of models

### Distill SFT data
- We use GPT-4o to distill trajectories fitting that of Mem1 style
- First navigate to `Mem1` directory
- First, build the tasks by running the following scripts (here we build for 1-obj, 2-obj, and 3-obj tasks)
    - `python gen_data/data_process/nq_search_train_merge.py`
    - `python gen_data/data_process/nq_search_train_merge_multi.py --batch_size 2`
    - `python gen_data/data_process/nq_search_train_merge_multi.py --batch_size 3`
- Then, we can generate the rollout data for each dataset by running
    - `python inference/generate_rollout.py --model gpt-4o --use_mem1 --use_litellm --data_file data/xxx` for each of the task dataset generated
- Lastly, run `python distill/build_sft_dataset.py` to consolidate the generated rollout paths to a SFT dataset. (depending on the path of the jsonls you generated, you might need to modify the inputs of build_sft_dataset.py abit)
- You will obtain a `RAG_train_sft.json` file. You can then just SFT using that json. In our initial experiment, we used the `swift` library.

### Evaluate trained models

For evaluation, the process is similar to that of distillation. If you use an opensource model, you can first host it on your local machine by running `start_vllm.sh` (remember to first modify the model path and cuda visible devices in the bash file).


Then, run the rollout generation file
- `python inference/generate_rollout.py --model [VLLM MODEL PATH] [--use_mem1] [--use_amem] --data_file train/data/xxx`


After that, you can evaluate the model performance by running
- `python inference/eval.py --eval_file_path xxx`
