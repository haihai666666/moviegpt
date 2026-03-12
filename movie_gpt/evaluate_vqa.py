import csv
import json
import os
import cv2
import os
import argparse
import argparse
from tqdm import tqdm
import threading
import jsonlines
from transformers import AutoTokenizer
from mplug_owl.modeling_mplug_owl import MplugOwlForConditionalGeneration
from mplug_owl.processing_mplug_owl import MplugOwlImageProcessor, MplugOwlProcessor
from PIL import Image
from pipeline.interface import get_model
from pipeline.interface import do_generate
import argparse
import os
import random
import sys
import json
from tqdm import tqdm
import torch
from peft import LoraConfig, get_peft_config, get_peft_model
class mPLUG_Owl_Server:
    def __init__(
        self,
        base_model='MAGAer13/mplug-owl-bloomz-7b-multilingual',
        log_dir='./',
        load_in_8bit=False,
        bf16=True,
        device="cuda",
        io=None
    ):
        self.log_dir = log_dir
        self.image_processor = MplugOwlImageProcessor.from_pretrained(base_model)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.processor = MplugOwlProcessor(self.image_processor, self.tokenizer)
        self.model = MplugOwlForConditionalGeneration.from_pretrained(
            base_model,
            load_in_8bit=load_in_8bit,
            torch_dtype=torch.bfloat16 if bf16 else torch.half,
            device_map="auto"
        )
        self.tokenizer = self.processor.tokenizer
        peft_config = LoraConfig(
            target_modules=r'.*language_model.*\.query_key_value', inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.05
        )
        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()
        lora_path = '/iris/u/huaxiu/ch/yiyang/mPLUG-Owl/output_vqa/sft_v0.1_ft_grad_ckpt/checkpoint-12600/pytorch_model.bin'
        print('load lora from {}'.format(lora_path))
        prefix_state_dict = torch.load(lora_path, map_location='cpu')
        self.model.load_state_dict(prefix_state_dict)
def parse_args():
    # parser.add_argument('folder_path', type=str, help='Path to the folder containing CSV files')
    parser = argparse.ArgumentParser(description="Fintuning")
    parser.add_argument("--image_dir", type=str, default="")
    args = parser.parse_args()
    return args

torch.cuda.set_device(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# input_dir = " "
output_file = "output_vqa_12600.jsonl"
time_seg = []
# model, tokenizer, processor = get_model(
#     pretrained_ckpt='/iris/u/huaxiu/ch/yiyang/mPLUG-Owl/output_new/sft_v0.1_ft_grad_ckpt/checkpoint-500/',
#     use_bf16=True)
# peft_config = LoraConfig(target_modules=r'.*language_model.*\.(q_proj|v_proj)', inference_mode=False, r=8,
#                          lora_alpha=32, lora_dropout=0.05)
# model = get_peft_model(model, peft_config)
# lora_path = '/iris/u/huaxiu/ch/yiyang/mPLUG-Owl/output_new/sft_v0.1_ft_grad_ckpt/checkpoint-9000/pytorch_model.bin'
# prefix_state_dict = torch.load(lora_path, map_location='cpu')
# model.load_state_dict(prefix_state_dict)
# model = model.to(device)
instruction = "The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. \nHuman: <image>\nHuman: "
se = mPLUG_Owl_Server()
model = (se.model).to(device)
objs = []
count = 0
with open('clean_data_vqa_mplug.jsonl', 'r') as file:
    with open(output_file, "a+") as f:
        for line in file:
            # if count%50!=0:
            #     count+=1
            #     continue
            count+=1
            obj = json.loads(line)
            qs = obj["text"].split('\nAI: ')[0]
            if ('原因' not in qs or count%50==0):
                count += 1
                continue
            image_list = obj["image"]
            num_images = len(image_list)

            # 在文本中找到 "<image>\nHuman:" 替换为 num_images个 "<image>\nHuman:"
            # new_human_line = "<image>\nHuman: " * num_images
            # qs = qs.replace("<image>\nHuman:", new_human_line)
            qs += '\nAI: '
            response = do_generate([qs], image_list, model, se.tokenizer, se.processor, use_bf16=True, max_length=2048, top_k=2,
                                   do_sample=False)
            #response = response.split("\nAI: ")[1].split("\n")[0]

            result = {"id": image_list, "answer": response, "question": qs, "gold_ans": obj["text"].split('\nAI: ')[1]}
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
            f.flush()
    f.close()
    file.close()
#  "model": "Mplug_owl"
