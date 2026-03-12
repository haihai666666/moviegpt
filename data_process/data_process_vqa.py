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



# def process_jsonl_file(file_path):
#     # 读取原始JSONL文件
#     with open(file_path, 'r', encoding='utf-8') as file:
#         lines = file.readlines()
#
#     new_lines = []
#     for line in lines:
#         data = json.loads(line.strip())
#         if len(data.get('image', [])) > 0:
#             new_lines.append(line)
#
#     # 将结果写回原始JSONL文件
#     with open(file_path, 'w', encoding='utf-8') as file:
#         file.writelines(new_lines)

# 指定JSONL文件路径
# jsonl_file_path = 'data_vqa_mplug.jsonl'
#
# # 处理JSONL文件
# process_jsonl_file(jsonl_file_path)
# # ./minigpt4_utils/minigpt4_eval.yaml
# # llava /iris/u/huaxiu/ch/LLaVA/llava/eval/LLaVA-13B-v1
# # llama_adp /iris/u/huaxiu/ch/base_model/LLaMA-7B  /iris/u/huaxiu/ch/models/LLaMA-Adapter/llama_adapter_v2_multimodal7b/ckpts/7fa55208379faf2dd862565284101b0e4a2a72114d6490a95e432cf9d9b6c813_BIAS-7B.pth
args = parse_args()

# Parse the command-line arguments
# Get the folder path from the command-line arguments
# folder_path = args.folder_path
image_dir = args.image_dir


def timeout_handler():
    global timeout_flag
    timeout_flag = True
    print("Time out")

def extract_frames(video_p, start_time, end_time, output_path, frame_ratio=8):
    # 打开视频文件

    file_list = list()
    # 获取视频的帧速率
    video = cv2.VideoCapture(video_p)
    fps = video.get(cv2.CAP_PROP_FPS)

    # 计算开始和结束时间对应的帧索引

    start_frame = int(start_time * fps)
    if start_frame > video.get(cv2.CAP_PROP_FRAME_COUNT)-1:
        print('out_of_range')
        return []
    end_frame = int(end_time * fps)
    frame_num = int((end_frame - start_frame)/frame_ratio)
    # 设置视频的当前帧索引
    video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    # 设置超时时间（秒）

    # 定义函数来执行 video.set 操作
    # def set_frame_position():
    #     global operation_done
    #     video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    #     operation_done = True
    #
    # thread = threading.Thread(target=set_frame_position)
    # thread.start()

    # 等待操作完成或超时
    # thread.join(timeout)
    # if ~operation_done:
    #     print('time out')
    #     return []

    # 启动线程
    # thread.start()
    # # 设置视频的当前帧索引
    # video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # 检查是否超时
    # if timeout_flag:
    #     timer.cancel()
    #     print('time out')
    #     return []
    #
    # # 取消定时器
    # timer.cancel()

    frame_show = 0
    frame_count = 0
    if frame_num == 0:
        # start_frame = start_frame - 4
        # end_frame = end_frame + 4
        # frame_num = int((end_frame - start_frame) / frame_ratio)
        return []

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


# 设置CSV文件路径
# folder_path = '/path/to/folder'  # Replace with the actual folder path

csv_files = []
objs = []
with open('data_vqa_mplug.jsonl', 'r') as file:
    for line in file:
        obj = json.loads(line)
        objs.append(obj["text"])
    file.close()

for file_name in os.listdir(os.getcwd()):
    if file_name.endswith('.csv') and file_name != 'xghx.csv':
        csv_files.append(file_name)
count = 0
# Print the list of CSV files
for csv_file in csv_files:
    # print(csv_file)
    # csv_file = '情感判断.csv'
    # if count <14:
    #     count += 1
    #     continue

    # 调用函数读取CSV文件并提取电影名和JSON数据
    movies = read_csv_movfile(csv_file)
    # 'data_vqa_' + csv_file.split(".")[0] + '.jsonl'
    with open('data_vqa_mplug.jsonl', 'a+', encoding='utf-8') as file:
        ct = 0
        for movie in tqdm(movies, total= len(movies), desc='Processing movies'):
            # if ct<=31 :
            #     ct += 1
            #     continue
            ct+=1
            movie_name, data = movie
            movie_path = movie_name.split(".")[0] + ".mp4"
            print(movie_name)
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
                print('no video')
            for item in data["items"]:
                file_list = []
                instance = {}
                seg = item["meta"]["segment_range"]
                start_time = int(seg[0])
                end_time = int(seg[1])
                ans = item["labels"]["answer"]
                prompt = item["labels"]["prompt"]

                instance["image"] = []
                instance["text"] = "The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. \nHuman: <image>\nHuman: "
                instance["text"] = instance["text"] + prompt + '\nAI: ' + ans
                instance["task_type"] = "llava_sft"



                if instance["text"] not in objs:
                    if video != -1:
                        file_list = extract_frames(os.path.join(os.getcwd(), movie_path), start_time, end_time,
                                                   output_path)
                        instance["image"] = file_list
                        if len(file_list)!=0:
                            file.write(json.dumps(instance,  ensure_ascii=False) + '\n')
                            file.flush()
        count += 1
        file.close()
        #         break
        #     break
        # break
                



