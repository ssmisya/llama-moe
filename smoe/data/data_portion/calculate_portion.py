import json

input_file_name = "/mnt/petrelfs/songmingyang/code/llama-moe/smoe/data/data_portion/the_stack_eval_language_portion.json"
output_file_name = "./eval_language_portion.json"
with open(input_file_name, "r") as f:
    data = json.load(f)

    sum_num = sum([data[i] for i in data.keys()])

    portion = {i: data[i] / sum_num for i in data.keys()}
    print(portion)

    scaled_portion = {i: portion[i] * 0.85 for i in portion.keys()}
    print("scaled:")
    print(scaled_portion)
    with open(output_file_name, "w") as fw:
        fw.write(json.dumps(scaled_portion, ensure_ascii=False) + "\n")
