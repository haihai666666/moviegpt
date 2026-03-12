import csv
import json
import cv2
import os
import argparse
import re
import datetime
from tqdm import tqdm
import jsonlines
def cv_imwrite(savePath,tem):
    cv2.imencode('.jpg',tem)[1].tofile(savePath)
# Create an ArgumentParser object
# Add an argument for the folder path
def parse_args():
    # parser.add_argument('folder_path', type=str, help='Path to the folder containing CSV files')
    parser = argparse.ArgumentParser(description="Fintuning")
    parser.add_argument("--image_dir", type=str, default="")
    args = parser.parse_args()
    return args


# ./minigpt4_utils/minigpt4_eval.yaml
# llava /iris/u/huaxiu/ch/LLaVA/llava/eval/LLaVA-13B-v1
# llama_adp /iris/u/huaxiu/ch/base_model/LLaMA-7B  /iris/u/huaxiu/ch/models/LLaMA-Adapter/llama_adapter_v2_multimodal7b/ckpts/7fa55208379faf2dd862565284101b0e4a2a72114d6490a95e432cf9d9b6c813_BIAS-7B.pth
args = parse_args()

# Parse the command-line arguments
# Get the folder path from the command-line arguments
# folder_path = args.folder_path
image_dir = args.image_dir


def extract_frames(video_p, start_time, end_time, output_path, num_word, frame_ratio=8):
    # 打开视频文件

    file_list = list()
    # 获取视频的帧速率
    video = cv2.VideoCapture(video_p)
    fps = video.get(cv2.CAP_PROP_FPS)
    if start_time-end_time<2:
        end_time = end_time + num_word/4
    end_time = end_time + 1
    # 计算开始和结束时间对应的帧索引
    start_frame = int(start_time * fps)
    # if start_frame - int(fps/2) < 0:
    #     start_frame = 0
    # else:
    #     start_frame -= int(fps/2)
    end_frame = int(end_time * fps)
    if start_frame > video.get(cv2.CAP_PROP_FRAME_COUNT)-1:
        print('out_of_range')
        return []
    if end_frame  > video.get(cv2.CAP_PROP_FRAME_COUNT)-1:
        end_frame = video.get(cv2.CAP_PROP_FRAME_COUNT)-1
    else:
        end_frame  = end_frame

    frame_num = int((end_frame - start_frame)/frame_ratio)
    if frame_num == 0:
        return []
    # 设置视频的当前帧索引
    video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_show = 0
    frame_count = 0
    while True:
        # 读取当前帧
        ret, frame = video.read()

        # 如果无法读取帧，则退出循环
        if not ret:
            break

        # 如果超过结束帧，则退出循环
        if frame_count + start_frame > end_frame:
            break

        # 保存当前帧为图片
        if frame_count % frame_num == 0:
            frame_path = f"{output_path}/start_{start_time}_end{end_time}_frame_{frame_show}.jpg"
            file_list.append(frame_path)
            cv_imwrite(frame_path, frame)

            frame_show += 1
        frame_count += 1
    # 释放视频对象
    video.release()

    return file_list

    # print(f"提取了 {frame_count} 帧，并保存到 {output_path} 目录下。")


def read_csv_movfile(csv_file):
    movies = []

    with open(csv_file, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        # 跳过CSV文件的标题行
        next(reader)

        for row in reader:
            movie_name = row[1]
            json_data = row[2]

            # 解析JSON数据
            data = json.loads(json_data)

            # 将电影名和JSON数据添加到列表中
            movies.append((movie_name, data))

    return movies


def obtain_dial_list(file_path):
    # file_path = "咱们结婚吧.txt"  # 替换为你的文件路径
    #pattern_1 = r"(\d{2}:\d{2}:\d{2})\n+(.+?)：(.+)"
    pattern_2 = r"(\d{2}:\d{2}:\d{2})\n+(.+?):(.+)"

    #combined_pattern = f"{pattern_1}|{pattern_2}"
    # pattern_1 = r"(\d{2}:\d{2}:\d{2})\n+(.+?)：(.+)"
    # pattern_2 = r"(\d{2}:\d{2}:\d{2})\n+(.+?):(.+)"
    # pattern = f"""{pattern_1})|{pattern_2}"""
    with open(file_path, "r", encoding='utf-8') as file:
        lines = file.read()
    lines = lines.replace("：", ":")
    dialogues = []
    characters = []
    seconds = []
    rounds = []
    matches = re.findall(pattern_2, lines)
    #matches2 = re.findall(pattern_2, lines)
    for match in matches:
        time = match[0]
        if time=='':continue
        character = match[1]
        dialogue = match[2]
        time_obj = datetime.datetime.strptime(time, "%H:%M:%S")
        second = time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second
        dialogues.append(dialogue)
        characters.append(character)
        seconds.append(second)
    i = 0
    while i < len(dialogues) - 1:

        tim_seg = []
        round_cha = []
        round_dia = []
        if seconds[i + 1] - seconds[i] > 30 or characters[i] == characters[i + 1]:
            i = i + 1
            continue
        tim_seg.append(seconds[i])
        tim_seg.append(seconds[i + 1])
        round_cha.append(characters[i])
        round_cha.append(characters[i + 1])
        round_dia.append(dialogues[i])
        round_dia.append(dialogues[i + 1])

        for j in range(i + 2, len(dialogues)):
            if characters[j] in round_cha and characters[j] != characters[j - 1] \
                    and (seconds[j] - seconds[j - 1]) < 30:
                round_dia.append(dialogues[j])
                tim_seg[1] = seconds[j]
                # i = j
            else:
                break
        rounds.append((tim_seg, round_cha, round_dia))

        i = i + 1
    return rounds
    # for round in rounds:
    #     #print(round)
    #
    #
    #     file = open('dial_test.txt', 'a')
    #     # 删除单引号并用字符空格代替逗号
    #
    #     file.write(str(round))
    #     file.write('\n')
    #     file.close()


# 设置CSV文件路径
# folder_path = '/path/to/folder'  # Replace with the actual folder path
objs = []
titles = []
with open('data_dialogue_mplug.jsonl', 'r') as file:
    for line in file:
        obj = json.loads(line)
        objs.append(obj["text"])

        titles.append(obj["image"][0].split("/")[-2])
    file.close()
txt_files = []

for file_name in os.listdir(os.getcwd()):
    if file_name.endswith('.txt'):
        txt_files.append(file_name)
movie_indentity = read_csv_movfile('xghx.csv')
# Print the list of CSV files
count = 0
for txt_file in tqdm(txt_files, total=len(txt_files), desc='Processing dia'):
    flag = 0
    count+=1
    if count <=87:
        count += 1
        continue
    # print(csv_file)
    # csv_file = '情感判断.csv'

    # 调用函数读取CSV文件并提取电影名和JSON数据
    #movies = read_csv_movfile(csv_file)
    # 'data_vqa_' + csv_file.split(".")[0] + '.jsonl'
    dial_list = obtain_dial_list(txt_file)
    for movie in movie_indentity:
        movie_name, data = movie
        if movie_name == txt_file.split(".")[0]:
            flag = 1
            break
    if flag==0:
        print(f"""miss {txt_file.split(".")[0]}""")
        continue

    print(movie_name)

    # if  movie_name in titles and movie_name!='风声' and movie_name!='老炮儿' :
    #     continue

    ch_portrait = {}
    for item in data["items"]:
        item_l = item["labels"]
        ch_portrait[item_l["Role"]] = list()
        ch_portrait[item_l["Role"]].append(item_l["Individuality"])
        ch_portrait[item_l["Role"]].append(item_l["Identity"])


    with open('data_dialogue_mplug.jsonl', 'a+', encoding='utf-8') as file:
        for round in dial_list:
            seg, round_cha, round_dia = round
            movie_path = movie_name + ".mp4"

            if os.path.isfile(os.path.join(os.getcwd(), movie_path)):
                home = os.getcwd()
                video = cv2.VideoCapture(os.path.join(os.getcwd(), movie_path))
                if not video.isOpened():
                    print("无法打开视频文件")
                    exit()

                if image_dir != '':
                    movie_directory = os.path.join(image_dir, movie_name)
                else:
                    movie_directory = os.path.join('movies', movie_name)
                #movie_directory = 'movies'
                if not os.path.isdir(movie_directory):
                    os.makedirs(movie_directory)
                output_path = os.path.join(os.getcwd(),movie_directory)
            else:
                video = -1

            instance = {}
            start_time = int(seg[0]) - 1
            end_time = int(seg[1]) + 1
            instance["image"] = []
            #prompt = item["labels"]["prompt"]

            # instance["image"] = file_list
            if round_cha[1] in ch_portrait.keys():
                instruction = f"""接下来你将根据用户提供的电影片段扮演其中一个角色跟用户对话. 在这个片段中你将扮演{round_cha[1]}, 你的经历是:{ch_portrait[round_cha[1]][0]}, 你的性格是:{ch_portrait[round_cha[1]][1]}. \nHuman: <image>\nHuman:"""
            else:
                instruction = f"""接下来你将根据用户提供的电影片段扮演其中一个角色跟用户对话. 在这个片段中你将扮演{round_cha[1]}. \nHuman: <image>"""

            round_dialogue = ""
            num_w = 0
            for i in range(len(round_dia)):
                if i % 2 == 0:
                    round_dialogue = round_dialogue + "\nHuman: " + round_dia[i]
                    num_w += len(round_dia[i])
                else:
                    round_dialogue = round_dialogue + "\nAI: " + round_dia[i]
                    num_w += len(round_dia[i])

            #instance["text"] = "The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. \nHuman: <image>\nHuman:"
            # instance["text"] = instance["text"] + prompt + '\nAI:' + ans
            instance["text"] = instruction + round_dialogue
            instance["task_type"] = "llava_sft"

            if video != -1 and instance["text"] not in objs:
                file_list = extract_frames(os.path.join(os.getcwd(), movie_path), start_time, end_time, output_path, num_w)
                instance["image"] = file_list
                if len(file_list) != 0:
                    file.write(json.dumps(instance, ensure_ascii=False) + '\n')
                    file.flush()

        file.close()




