import re

def parse_mmlu_response(mmlu_example:dict,
                        mmlu_output:str)->str | None:
    match=re.search(r"The correct answer is\s*([ABCD])\b",mmlu_output,flags=re.IGNORECASE)
    #r表示是原始字符串，不涉及转义
    if match is not None:
        return match.group(1).upr()#.group(0)是整ge匹配的对象,括号内是捕获的对象，(1)表示第一个捕获的对象


    