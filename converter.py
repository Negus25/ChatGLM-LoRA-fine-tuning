import argparse
import json
import datetime
import os

'''
orig_data: {"instruction": "题目：小明买了一支钢笔，花费了5元，又买了一本书，花费8元，现在他手里还有10元钱，他手上原来有多少钱？", "input": "", "output": "\n令小明手上原来有的钱为X元。根据题目描述，得出以下方程式：\nX - 5 - 8 = 10\n化简可得：\nX = 23\n因此，小明手上原来有23元钱。"}
convert: {
        "instruction": "题目：小明买了一支钢笔，花费了5元，又买了一本书，花费8元，现在他手里还有10元钱，他手上原来有多少钱？",
        "output": "\n令小明手上原来有的钱为X元。根据题目描述，得出以下方程式：\nX - 5 - 8 = 10\n化简可得：\nX = 23\n因此，小明手上原来有23元钱。"
        }
'''

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--orig_data",
    )
    parser.add_argument(
        "--write_data",
    )
    parser.add_argument(
        "--dataset_name",
    )
    args = parser.parse_args()
    f_write = open(args.write_data,"w")
    with open(args.orig_data) as f:
        lines = f.readlines()
        num_id = 1
        for line in lines:
            data = json.loads(line)
            conversations = {"instruction": data['instruction']+data['input'],"output": data['output']}
            f_write.write(json.dumps(conversations, ensure_ascii=False)+"\n")
            num_id += 1
    f_write.close()


if __name__ == "__main__":
    main()
