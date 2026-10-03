import argparse
import json
import time
import csv
import sys
from pathlib import Path

import torch

from transformers import AutoTokenizer,AutoModelForCausalLM
DATA_DIR=Path("data/alpaca_eval/alpaca_eval_gpt4_turbo.json")
ALPACA_PROMPT_PATH=Path("cs336_alignment/prompts_safety/alpaca_eval_zero_shot.prompt")
SYSTEM_PROMPT_PATH=Path("cs336_alignment/prompts_safety/zero_shot_system_prompt.prompt")
BASE_MODEL_PATH=()
RESULTS_VOLUME_MOUNT_PATH=()

def build_task_prompt(template,
                      instruction,
                      dataset):
    return template.format(instruction=instruction,
                           dataset=dataset)

def build_full_prompt(system_template,
                      instructions):
    return system_template.format(instruction=instructions)

def build_prompt(example:dict,
                 alpaca_template:str,
                 system_template:str):
    task_prompt=build_task_prompt(alpaca_template,example["instruction"],example["dataset"])
    return build_full_prompt(system_template,task_prompt)
def load_alpaca()->list[dict]:
    examples=[]
    with open(DATA_DIR,"r",encoding="utf-8") as f:
        reader=json.load(f)
        for data in reader:
            examples.append({"instruction":data["instruction"],
                                })

    """ with open()as f:
     reader=json.load(f)  (json file is a big list)
     for data in reader:
     examples.append("question":data["question])
       """
    return examples

@torch.inference_mode()
def evaluate(limit=None,batch_size=8):
    #load mmlu
    examples=load_alpaca()
    if limit is not None:
        examples=examples[:limit]
    print(f"loaded{len(examples)}examples")

    #load prompt template
    alpaca_template=ALPACA_PROMPT_PATH.read_text(encoding="utf-8")
    system_template=SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    prompts=[build_prompt(example,alpaca_template,system_template) for example in examples]
    
    #load tokenizer
    tokenizer=AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
    tokenizer.padding_side="left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token=tokenizer.eos_token

    #load model
    model=AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH,torch_dtype=torch.bfloat16).to("cuda")
    model.eval()
    results=[]#list[dict]
    torch.cuda.synchronize()#等待之前GPU上的工作完成
    start_time=time.perf_counter()

    #building based on batches
    for start in range(0,
                       len(examples),
                       batch_size):
        end=min(start+batch_size,len(examples))
        batch_examples=examples[start:end]
        batch_prompts=prompts[start:end]
        inputs=tokenizer(batch_prompts,return_tensors="pt",padding=True).to("cuda")
        input_length=inputs["input_ids"].shape[1]#input通常是一个dict，形如{"input_ids":tensor()，"attention_mask":tensor()}
        outputs=model.generate(**inputs,
                               max_new_tokens=512,
                               do_sample=False,#贪心生成
                               pad_token_id=tokenizer.pad_token_id,
                               eos_token_id=tokenizer.eos_token_id)
        generated_tokens=outputs[:,input_length:]
        generations=tokenizer.batch_decode(generated_tokens,skip_special_tokens=True)
        for example,model_output in zip(batch_examples,generations):
            if "# Query:" in model_output:
                model_output=model_output.split("# Query:",1)[0]
            model_output=model_output.strip()
            results.append({"instruction":example["instruction"],
                            "output":model_output,
                            "generator": "llama-3.1-8b-base",
                            "dataset": example["dataset"]})
        print(f"{end}/{len(examples)}")
    torch.cuda.synchronize()
    elapsed=time.perf_counter()-start_time

    #metrics
    total=len(results)
    throughput=total/elapsed
    print()
    print(f"throughput{throughput:.3f}examples/s")

    #save results
    output_dir=(Path(RESULTS_VOLUME_MOUNT_PATH)/"alpaca_zero_shot")
    output_dir.mkdir(parents=True,exist_ok=True)
    with open(output_dir/"results.json","w",encoding="utf-8")  as f:
        for result in results:
            json.dump(result,f,ensure_ascii=False,indent=2)
    print(f"saved to{RESULTS_VOLUME_MOUNT_PATH}")

def make_parser():
    parser=argparse.ArgumentParser()
    parser.add_argument("--batch_size",type=int,default=8)
    parser.add_argument("--limit",type=int,default=None)
    return parser

if __name__=="__main__":
    args=make_parser().parse_args()
    evaluate(limit=args.limit,batch_size=args.batch_size)

