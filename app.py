import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

class InferlessPythonModel:
    def initialize(self):
        quantization_config = BitsAndBytesConfig(load_in_8bit=True,llm_int8_skip_modules=["mamba"])
        model = AutoModelForCausalLM.from_pretrained("ai21labs/Jamba-v0.1",
                                             trust_remote_code=True,
                                             torch_dtype=torch.bfloat16,
                                             attn_implementation="flash_attention_2",
                                             quantization_config=quantization_config)
        tokenizer = AutoTokenizer.from_pretrained("ai21labs/Jamba-v0.1")
        

        
    def infer(self,inputs):
        prompts = inputs["prompt"]  # Extract the prompt from the input
        input_ids = tokenizer(prompts, return_tensors='pt').to(model.device)["input_ids"]
        outputs = model.generate(input_ids, max_new_tokens=216)
        generated_text = tokenizer.batch_decode(outputs)
        # Return a dictionary containing the result
        return {'generated_result': generated_text}

    def finalize(self):
        pass
