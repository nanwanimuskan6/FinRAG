"""
FinRAG grounded LLM generator.

The model is instructed to answer only from retrieved evidence.
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


class LLMGenerator:

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    ):

        print("Loading language model...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32
        )

        self.model.eval()

        print("Language model loaded successfully.")

    def generate(self, prompt: str) -> str:

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a financial document question-answering "
                    "assistant. Answer ONLY using the supplied evidence. "
                    "Never use outside knowledge. "
                    "If the answer is not explicitly present in the evidence, "
                    "say that it cannot be found."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096
        )

        with torch.no_grad():

            output = self.model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=False,
                temperature=None,
                top_p=None,
                repetition_penalty=1.05,
                pad_token_id=self.tokenizer.eos_token_id
            )

        generated_tokens = output[
            0,
            inputs["input_ids"].shape[1]:
        ]

        answer = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True
        )

        return answer.strip()
    