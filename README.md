# MindCon2023_LLM_Task4-4_ChatGLM


## 基于MindSpore的ChatGLM模型LoRA微调

ChatGLM-6B 是一个开源的、支持中英双语的对话语言模型，基于 General Language Model (GLM) 架构，具有 62 亿参数。ChatGLM-6B 使用了和 ChatGPT 相似的技术，针对中文问答和对话进行了优化。经过约 1T 标识符的中英双语训练, 辅以监督微调、反馈自助、人类反馈强化学习等技术的加持，62 亿参数的 ChatGLM-6B 已经能生成相当符合人类偏好的回答。

可以使用AutoClass接口, 通过模型名称获取相应的模型/tokenizer实例, 并自动下载并加载权重
from_pretrained() 接口会自动从云上下载预训练的模型，存储路径：./checkpoint_download/glm
```
import mindspore; mindspore.set_context(mode=0, device_id=0)
from mindformers import AutoModel, AutoTokenizer, TextGenerationPipeline
model = AutoModel.from_pretrained("glm_6b")
tokenizer = AutoTokenizer.from_pretrained("glm_6b")
```
## School Math 0.25M Dataset
包含约25万条由BELLE项目生成的中文数学题数据, 包含解题过程。
注意: 此数据集是由ChatGPT产生的, 未经过严格校验, 题目或解题过程可能包含错误。使用过程中请注意这一点
```
{
  "instruction": "题目：小华手里有一个装满糖果的袋子，共有12个，小明想知道里面有多少颗糖果，于是他问小华：“你手里的糖果袋子里有偶数个糖果吗？”小华回答：“有，而且多于10颗。”请问小华手里的糖果袋子里最少有几颗糖果？",
  "input": "",
  "output": "\n由题目可知：小华手里的糖果袋子里有偶数个糖果；\n又知道小华手里的糖果袋子里有多于10颗糖果。\n因为糖果数为偶数，多于10颗，所以糖果数最小必须是12。\n所以小华手里的糖果袋子里最少有12颗糖果。"
}
```
## 下载数据
```
wget https://huggingface.co/datasets/BelleGroup/school_math_0.25M/resolve/main/school_math_0.25M.json
```
该数据是数学应用题求解任务，包含解题过程，共计约25万条。
### 转换数据格式

demo选取前1000条作为验证集，其余数据作为训练集。
```
!python converter.py --orig_data ./data/school_math_0.25M.json --write_data school_math_0.25M_conv.json --dataset_name bellemath
!head -n 1000 school_math_0.25M_conv.json > belleMat-test1K.json
!tail -n +1001 school_math_0.25M_conv.json > belleMath-train.json
```
We just add instruction and output and assign it to instruction.
```
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
```
After processing,
```
conversations = {"instruction": data['instruction']+data['input'],"output": data['output']}
{
        "instruction": "题目：小明买了一支钢笔，花费了5元，又买了一本书，花费8元，现在他手里还有10元钱，他手上原来有多少钱？",
        "output": "\n令小明手上原来有的钱为X元。根据题目描述，得出以下方程式：\nX - 5 - 8 = 10\n化简可得：\nX = 23\n因此，小明手上原来有23元钱。"
        }
```
## 数据处理

使用 school-math.py 脚本将数据集处理成mindrecord格式。
执行命令生成训练数据集：

```
!python school-math.py \
    --input_file belleMath-train.json \
    --vocab_file ice_text.model\
    --output_file belle-train.mindrecord \
    --max_source_length 64 \
    --max_target_length 64 \
    --mode train
    
```
/home/ma-user/anaconda3/envs/MindSpore/lib/python3.9/site-packages/numpy/core/getlimits.py:549: UserWarning: The value of the smallest subnormal for <class 'numpy.float64'> type is zero.
  setattr(self, word, getattr(machar, word).flat[0])
/home/ma-user/anaconda3/envs/MindSpore/lib/python3.9/site-packages/numpy/core/getlimits.py:549: UserWarning: The value of the smallest subnormal for <class 'numpy.float32'> type is zero.
  setattr(self, word, getattr(machar, word).flat[0])
/home/ma-user/anaconda3/envs/MindSpore/lib/python3.9/site-packages/jieba/_compat.py:18: DeprecationWarning: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html
  import pkg_resources
/home/ma-user/anaconda3/envs/MindSpore/lib/python3.9/site-packages/pkg_resources/__init__.py:2871: DeprecationWarning: Deprecated call to `pkg_resources.declare_namespace('google')`.
Implementing implicit namespace packages (as specified in PEP 420) is preferred to `pkg_resources.declare_namespace`. See https://setuptools.pypa.io/en/latest/references/keywords.html#keyword-namespace-packages
  declare_namespace(pkg)
100%|##########################################################################| 247481/247481 [29:41<00:00, 138.92it/s]
2024-01-22 12:39:43,297 - mindformers[/dataset/school-math.py:191] - INFO - Wrote 247480 total instances
/dataset/school-math.py:226: ResourceWarning: unclosed file <_io.TextIOWrapper name='belleMath-train.json' mode='r' encoding='UTF-8'>
  preprocess_function(

执行命令生成评估数据集：
```
!python school-math.py \
    --input_file belleMat-test1K.json \
    --vocab_file ice_text.model \
    --output_file belle-eval.mindrecord \
    --max_source_length 256 \
    --max_target_length 256 \
    --mode eval
```
**将任务配置文件 run_glm_6b_lora.yaml 中的 ==== dataset config ==== 部分替换成：**
```
train_dataset: &train_dataset
  data_loader:
    type: MindDataset
    dataset_dir: ""
    shuffle: True
  input_columns: ["input_ids", "labels", "position_ids", "attention_mask"]
  num_parallel_workers: 8
  python_multiprocessing: False
  drop_remainder: True
  batch_size: 1
  repeat: 1
  numa_enable: False
  prefetch_size: 1
  seed: 0

train_dataset_task:
  type: CausalLanguageModelDataset
  dataset_config: *train_dataset

eval_dataset: &eval_dataset
  data_loader:
    type: MindDataset
    dataset_dir: ""
    shuffle: True
  input_columns: ["input_ids", "labels"]
  num_parallel_workers: 8
  python_multiprocessing: False
  drop_remainder: True
  batch_size: 1
  repeat: 1
  numa_enable: False
  prefetch_size: 1
  seed: 0

eval_dataset_task:
  type: CausalLanguageModelDataset
  dataset_config: *eval_dataset
```

## LoRA低参微调
```
use_parallel=False
task='text_generation'
model_type='glm_6b_lora'
checkpoint_path='./checkpoint_download/glm/glm_6b.ckpt'
train_dataset='belle-train.mindrecord'
eval_dataset='belle-eval.mindrecord'
predict_data='你好'
dp=1 
mp=1
pp=1
micro_size=1
op=False


import argparse

from mindformers import Trainer, TrainingArguments
from mindformers import init_context, ContextConfig, ParallelContextConfig

def context_init(use_parallel=False, optimizer_parallel=False):
    """init context for mindspore."""
    context_config = ContextConfig(mode=0, device_target="Ascend", device_id=0)
    parallel_config = None
    if use_parallel:
        parallel_config = ParallelContextConfig(parallel_mode='SEMI_AUTO_PARALLEL',
                                                gradients_mean=False,
                                                enable_parallel_optimizer=optimizer_parallel,
                                                full_batch=True)
    rank_id, device_num = init_context(use_parallel=use_parallel,
                                       context_config=context_config,
                                       parallel_config=parallel_config)
# 环境初始化
context_init(use_parallel, op)
# 训练超参数定义
yaml_path = 'run_glm_6b_lora.yaml' #we edit the yaml file to set the hyperparameters, check the run_glm_6b_lora.yaml file for more info.
#training_args = TrainingArguments(num_train_epochs=1, batch_size=batch_size, learning_rate=5e-5, warmup_steps=100, sink_mode=True, sink_size=4)
# 定义任务，预先准备好相应数据集
task = Trainer(task=task, model=model_type, args=yaml_path, train_dataset=train_dataset, eval_dataset=eval_dataset)
task.set_parallel_config(data_parallel=dp,
                         model_parallel=mp,
                         pipeline_stage=pp,
                         micro_batch_num=micro_size)
```
## start fine-tuning
```
task.finetune(checkpoint_path)
```

## 推理
```
import time
import mindspore as ms
import numpy as np
import argparse
from mindformers.models.glm import GLMConfig, GLMChatModel
from mindformers.models.glm.chatglm_6b_tokenizer import ChatGLMTokenizer
from mindformers.models.glm.glm_processor import process_response
from mindformers.pet.pet_config import LoraConfig
from mindformers.pet import get_pet_model


ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend", device_id=0)

pet_config = LoraConfig(
    lora_rank=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules = ".*query_key_value*"
)
config = GLMConfig(
    position_encoding_2d=True,
    use_past=True,
    is_sample_acceleration=True,
)

checkpoint_path = './output/checkpoint/rank_0/glm-6b-lora_rank_0-1625_4.ckpt'
vocab_path = 'ice_text.model'


model = GLMChatModel(config)
#config.pet_config = pet_config
model = get_pet_model(model, pet_config)

ms.load_checkpoint(checkpoint_path, model)
tokenizer = ChatGLMTokenizer(vocab_path)

inputs = ["你好",
          "今天小明骑自行车从家到学校用了20分钟，回家用了25分钟。如果小明在上学和回家的路上的速度一样，那么他从家到学校的距离是学校到家的距离的百分之几？"]

for query in inputs:
    input_ids = tokenizer(query)['input_ids']

    start_time = time.time()
    outputs = model.generate(input_ids, max_length=config.max_decode_length, do_sample=False)
    end_time = time.time()
    print(f'generate speed: {outputs[0].shape[0]/(end_time-start_time):.2f} tokens/s')

    response = tokenizer.decode(outputs)
    response = process_response(response[0])
    print(response)

```
2024-01-22 20:23:37,419 - mindformers[mindformers/generation/text_generator.py:1105] - WARNING - When do_sample is set to False, top_k will be set to 1 and top_p will be set to 0, making them inactive.
2024-01-22 20:23:37,424 - mindformers[mindformers/generation/text_generator.py:1109] - INFO - Generation Config is: {'max_length': 2048, 'max_new_tokens': None, 'num_beams': 1, 'do_sample': False, 'use_past': True, 'temperature': 1.0, 'top_k': 0, 'top_p': 1.0, 'repetition_penalty': 1.0, 'encoder_repetition_penalty': 1.0, 'renormalize_logits': False, 'pad_token_id': 3, 'bos_token_id': 130004, 'eos_token_id': 130005, '_from_model_config': True}
2024-01-22 20:23:37,426 - mindformers[mindformers/generation/text_generator.py:176] - INFO - The generation mode will be **GREEDY_SEARCH**.
2024-01-22 20:23:37,430 - mindformers[mindformers/generation/text_generator.py:309] - WARNING - max_length 2048 can not exceeds model seq_length 512, set max_length = seq_length.
2024-01-22 20:23:38,075 - mindformers[mindformers/generation/text_generator.py:478] - INFO - total time: 0.6454319953918457 s; generated tokens: 9 tokens; generate speed: 13.944149134621139 tokens/s
generate speed: 19.37 tokens/s
你好 你好！请问有什么需要帮助的吗？
2024-01-22 20:23:38,092 - mindformers[mindformers/generation/text_generator.py:1105] - WARNING - When do_sample is set to False, top_k will be set to 1 and top_p will be set to 0, making them inactive.
2024-01-22 20:23:38,094 - mindformers[mindformers/generation/text_generator.py:1109] - INFO - Generation Config is: {'max_length': 2048, 'max_new_tokens': None, 'num_beams': 1, 'do_sample': False, 'use_past': True, 'temperature': 1.0, 'top_k': 0, 'top_p': 1.0, 'repetition_penalty': 1.0, 'encoder_repetition_penalty': 1.0, 'renormalize_logits': False, 'pad_token_id': 3, 'bos_token_id': 130004, 'eos_token_id': 130005, '_from_model_config': True}
2024-01-22 20:23:38,096 - mindformers[mindformers/generation/text_generator.py:176] - INFO - The generation mode will be **GREEDY_SEARCH**.
2024-01-22 20:23:38,100 - mindformers[mindformers/generation/text_generator.py:309] - WARNING - max_length 2048 can not exceeds model seq_length 512, set max_length = seq_length.
2024-01-22 20:23:41,319 - mindformers[mindformers/generation/text_generator.py:478] - INFO - total time: 3.219697952270508 s; generated tokens: 63 tokens; generate speed: 19.567052852138772 tokens/s
generate speed: 32.12 tokens/s
今天小明骑自行车从家到学校用了20分钟，回家用了25分钟。如果小明在上学和回家的路上的速度一样，那么他从家到学校的距离是学校到家的距离的百分之几？ 
小明从家到学校的距离为：
d = 20 × 60 + 25 × 60 = 1200 米
小明从家到学校的速度为：
v = d / t = 1200 / 20




