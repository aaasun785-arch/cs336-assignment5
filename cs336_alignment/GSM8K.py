import re
from typing import Any

def parse_gsm8k_response(model_output:str)->str:
    numbers=re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?",model_output)
    if not numbers:
        return None
    result=numbers[-1].replace(",","")
    return result
