### evaluation
python main.py --save_dir ./eval/CUFED/TTSR \
               --reset True \
               --log_file_name eval.log \
               --eval True \
               --eval_save_results True \
               --num_workers 1 \
               --dataset CUFED \
               --dataset_dir ./CUFED/ \
               --model_path ./model_Motion.pt