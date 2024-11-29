# PRED

<a href="https://openreview.net/pdf?id=cq2uB30uBM"> <b> Pre-emptive Action Revision by Environmental Feedback for Embodied Instruction Following Agents </b> </a>
<br>
<a href="http://jinyeonkim.notion.site">Jinyeon Kim*</a>,
<a href="https://mch0916.github.io/">Cheolhong Min*</a>,
<a href="https://bhkim94.github.io/">Byeonghwi Kim</a>,
<a href="http://ppolon.github.io/"> Jonghyun Choi </a>
<br>
<a href="https://www.corl.org/"> CoRL 2024 </a>

**PRED** (**P**re-emptive action **R**evision by **E**nvironmental fee**D**back) allows embodied agents to revise their action in response to perceived environmental status “before they make mistakes.”

We note that our code is largely built upon <a href="https://bhkim94.github.io/projects/CAPEAM/">CAPEAM</a>.

## Code

## Downloading the dataset
Download as in the original [teach repo](https://github.com/alexa/teach#downloading-the-dataset)
- Move "teach-dataset" inside "PRED/EDH/teach-dataset"
- Move "teach-dataset" inside "PRED/TfD/teach-dataset"
- Move "teach-dataset" inside "PRED+/teach-dataset"

or

- ln -s teach-dataset PRED/EDH/teach-dataset
- ln -s teach-dataset PRED/TfD/teach-dataset
- ln -s teach-dataset PRED+/teach-dataset


## Setting up this repository


#### Step 1 - git clone this repo
#### Step 2 - Copy files from google drive 
Download and unzip from this link: [https://drive.google.com/file/d/1UOBNhuaKRcG3HxT14aRM_Fud5YH5tVCo/view?usp=share_link](https://drive.google.com/file/d/18jeotqa6dvSE94ysSh_qRHtDo-xqXAxu/view?usp=sharing)

- Move "BERT_models" inside "FILM_model/models/instructions_processed_LP"
- Move "depth_models" inside "FILM_model/models/depth/depth_models"
- Move "maskrcnn_alfworld" inside "FILM_model/models/segmentation"
- Move "best_model_multi.pt" inside "FILM_model/models/semantic_policy"
- Move "new_best_model.pt" inside "FILM_model/models/semantic_policy"
- Move "weight_maskrcnn.pt" inside "PRED/EDH"
- Move "weight_maskrcnn.pt" inside "PRED/TfD"
- Move "weight_maskrcnn.pt" inside "PRED+"


#### Step 3 - Installations
```
conda create --name pred python=3.7 
conda activate pred

pip install transformers==4.9.2 && pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html && python -m pip install -U detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.8/index.html

pip install -r requirement.txt
```

#### Step 4 - Run command 
Set the variables as in the original (if you want to run pred model in TfD)

##### Run TfD in PRED
  ```
  cd /home/pred/TfD/
  pip install -e .
  export DATA_DIR=/home/pred/TfD/teach-dataset
  export OUTPUT_DIR=/home/pred/TfD/output
  export IMAGE_DIR=/home/pred/TfD/img_dir
  export METRICS_FILE=/home/pred/TfD/output/metics
  ```
  
  Run command 
  
  ```
  CUDA_VISIBLE_DEVICES=0 python src/teach/cli/inference.py -- tfd --data_dir $DATA_DIR   --output_dir $OUTPUT_DIR   --split valid_seen  --metrics_file $METRICS_FILE  --model_module teach.inference.FILM_teach_model --model_class FILMModel  --images_dir $IMAGE_DIR --set_dn  edh_vs_0_304 --map_pred_threshold 40 --max_episode_length 1000 --cat_pred_threshold 10  --use_bert --start_idx 0 --end_idx 304
  ```

Flags:
- set_dn: The name of the saved pickle (in results/analysis_recs)
- start_idx: start of the task index
- end_idx: end of the task index


#### Step 5 - Check results 

Check results in "results/analyze_recs". Pickles are generated for each command. 

Calculate success rate:
![Screen Shot 2023-04-04 at 3 57 10 PM](https://user-images.githubusercontent.com/77866067/229905790-dc4b2b11-48bf-4478-8bbc-035cfe5f38e1.png)


## More explanations about the commands
This code can run both "edh" and "tfd" tasks.
Default is EDH. To run TfD, put a "--tfd" flag.
You can chose split among "valid_seen" and "valid_unseen"

##### Run EDH in PRED
  ```
  cd /home/pred/EDH/
  pip install -e .
  export DATA_DIR=/home/pred/EDH/teach-dataset
  export OUTPUT_DIR=/home/pred/EDH/output
  export IMAGE_DIR=/home/pred/EDH/img_dir
  export METRICS_FILE=/home/pred/EDH/output/metics
  ```
  
  Run command 
  
  ```
  CUDA_VISIBLE_DEVICES=0 python src/teach/cli/inference.py --data_dir $DATA_DIR   --output_dir $OUTPUT_DIR   --split valid_seen  --metrics_file $METRICS_FILE  --model_module teach.inference.FILM_teach_model --model_class FILMModel  --images_dir $IMAGE_DIR --set_dn  edh_vs_0_304 --map_pred_threshold 40 --max_episode_length 1000 --cat_pred_threshold 10  --use_bert --start_idx 0 --end_idx 304
  ```

##### Run TfD in PRED+
  Add personal "openai.api_key" inside "PRED+/FILM_model/ask_w_example.py"

  and 

  ```
  cd /home/pred+
  pip install -e .
  export DATA_DIR=/home/pred+/teach-dataset
  export OUTPUT_DIR=/home/pred+/output
  export IMAGE_DIR=/home/pred+/img_dir
  export METRICS_FILE=/home/pred+/output/metics
  ```
  
  Run command 
  
  ```
  CUDA_VISIBLE_DEVICES=0 python src/teach/cli/inference.py --data_dir $DATA_DIR   --output_dir $OUTPUT_DIR   --split valid_seen  --metrics_file $METRICS_FILE  --model_module teach.inference.FILM_teach_model --model_class FILMModel  --images_dir $IMAGE_DIR --set_dn  edh_vs_0_304 --map_pred_threshold 40 --max_episode_length 1000 --cat_pred_threshold 10  --use_bert --start_idx 0 --end_idx 304
  ```


## License

The code is licensed under the MIT License (see SOFTWARELICENSE), images are licensed under Apache 2.0 
(see IMAGESLICENSE) and other data files are licensed under CDLA-Sharing 1.0 (see DATALICENSE).

