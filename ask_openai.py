import json
import openai

# Initialize the OpenAI library with your API key.
openai.api_key = 'YOUR_API_KEY'


def get_gpt_response(prompt):
    try:
        response = openai.Completion.create(engine="gpt-3.5-turbo", prompt=prompt, max_tokens=150, n=1)
        # Parse the response to extract the generated text.
        generated_text = response.choices[0].text.strip()

        return generated_text
    except Exception as e:
        print(f"Error calling GPT API: {e}")
        return []




# 读取jsonl文件
with open('clean_data_vqa_mplug.jsonl', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 初始化输出列表
output_data = []

for line in lines:
    data = json.loads(line)


    first_image_path = data["image"][0]
    parts = first_image_path.split("/")
    tem = data['image'][0].split('_')
    start_time = (tem[-4])
    end_time = (tem[-3].split('.')[0])[3:]
    seg = [start_time, end_time]


    text_parts = data["text"].split("\n")
    question = text_parts[-2].split("Human:  ")[-1]
    goldanswer = text_parts[-1].split("AI: ")[-1]


    prompt = f"Input question: {question} Generate three diverse questions with the same meaning as the input sentence, Only output the sentence, without any extra words:"
    gpt_response = get_gpt_response(prompt)

    movie_name = parts[-2]

    output_data.append({
        "seg": seg,
        "question": question,
        "gold_answer": goldanswer,
        "movie_name": movie_name,
        "GPT_response": gpt_response
    })

# 将提取的数据保存为新的jsonl文件
with open('output.jsonl', 'w', encoding='utf-8') as f:
    for item in output_data:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')
