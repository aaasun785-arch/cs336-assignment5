import argparse
import json
import random
import time
import csv
import sys
from pathlib import Path

import torch

from cs336_alignment.MMLU import parse_mmlu_response 
from cs336_alignment.modal_utils_safety import app,BASE_MODEL_PATH,RESULTS_VOLUME_MOUNT_PATH,submit_commands
from transformers import AutoTokenizer,AutoModelForCausalLM
from cs336_alignment.MMLU import parse_mmlu_response
MMLU_DIR=Path("data/mmlu/test")
MMLU_PROMPT_PATH=Path("cs336_alignment/prompts_safety/mmlu_zero_shot.prompt")
SYSTEM_PROMPT_PATH=Path("cs336_alignment/prompts_safety/zero_shot_system_prompt.prompt")
LETTERS=["A","B","C","D"]

def build_task_prompt(template,
                      subject,
                      question,
                      options):
    return template.format(subject=subject.replace("_"," "),
                           question=question,
                           options=options)

def build_full_prompt(system_template,
                      instructions):
    return system_template.format(instructions=instructions)

def build_prompt(example:dict,
                 mmlu_template:str,
                 system_template:str):
    task_prompt=build_task_prompt(mmlu_template,example["subject"],example["question"],example["options"])
    return build_full_prompt(system_template,task_prompt)
def load_mmlu()->list[dict]:
    examples=[]

    for file_path in sorted(MMLU_DIR.glob("*_test.csv")):
        subject=(file_path.stem.removesuffix("_test").replace("_"," "))
        with open(file_path,"r",encoding="utf-8") as f:
            reader=csv.reader(f)
            for row in reader:
                examples.append({
                    "subject":subject,
                    "question":row[0],
                    "options":row[1:5],
                    "answer":row[5]
                })
    return examples

@torch.inference_mode()
def evaluate(limit=None,batch_size=8):
    #load mmlu
    examples=load_mmlu()
    if limit is not None:
        examples=examples[:limit]
    print(f"loaded{len(examples)}examples")

    #load prompt template
    mmlu_template=MMLU_PROMPT_PATH.read_text(encoding="utf-8")
    system_template=SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    prompts=[build_prompt(example,mmlu_template,system_template) for example in examples]
    
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
                               max_new_tokens=64,
                               do_sample=False,
                               pad_token_id=tokenizer.pad_token_id,
                               eos_token_id=tokenizer.eos_token_id)
        generated_tokens=outputs[:,input_length:]
        generations=tokenizer.batch_decode(generated_tokens,skip_special_tokens=True)
        for example,model_output in zip(batch_examples,generations):
            if "# Query:" in model_output:
                model_output=model_output.split("# Query:",1)[0]
                model_output=model_output.strip()
                prediction=parse_mmlu_response(example,model_output)
                results.append({**example,
                                "model_output":model_output,
                                "prediction":prediction,
                                "correct":prediction==example["answer"]})
            print(f"{end}/{len(examples)}")
        torch.cuda.synchronize()
        elapsed=time.perf_counter()-start_time

    #metrics
    total=len(results)
    correct=sum(result["correct"] for result in results)
    failed_parses=sum(result["prediction"] is None for result in results)
    accuracy=correct/total
    throughput=total/elapsed
    print()
    print(f"accuracy:{accuracy:.4f}")
    print(f"throughput{throughput:.3f}examples/s")

    #some parse-failures instances
    failures=[result for result in results if result["prediction"] is None]
    for result in failures[:5]:
        print()
        print("parse failures:")
        print(result["question"])
        print(result["model_output"])

    #save results
    output_dir=(Path(RESULTS_VOLUME_MOUNT_PATH)/"mmlu_zero_shot")
    output_dir.mkdir(parents=True,exist_ok=True)
    with open(output_dir/"results.jsonl","w",encoding="utf-8")  as f:
        for result in results:
            f.write(json.dumps(result,ensure_ascii=False)+"\n")
    summary={"total":total,
                "correct":correct,
                "accuracy":accuracy,
                "failed_parses":failed_parses,
                "generation_seconds":elapsed,
                "examples_per_second":throughput}
    with open(output_dir/"summary.json","w",encoding="utf-8")as f:
        json.dump(summary,f,indent=2)

def make_parser():
    parser=argparse.ArgumentParser()
    parser.add_argument("--batch_size",type=int,default=8)
    parser.add_argument("--limit",type=int,default=None)
    return parser

if __name__=="__main__":
    args=make_parser().parse_args()
    evaluate(limit=args.limit,batch_size=args.batch_size)

