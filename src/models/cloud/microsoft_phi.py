import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

class MicrosoftPhiMiniInstruct:
    def __init__(self, model_name="microsoft/Phi-3.5-mini-instruct"):
        # Set seed for reproducibility
        torch.random.manual_seed(0)
        
        # Load model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            device_map="auto", 
            torch_dtype=torch.float32,  
            trust_remote_code=True
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Initialize pipeline
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )
        
    def generate_text_response(self, messages, max_new_tokens=500, return_full_text=False, temperature=0.1, do_sample=True):
        # Set generation arguments
        generation_args = {
            "max_new_tokens": max_new_tokens,
            "return_full_text": return_full_text,
            "temperature": temperature,
            "do_sample": do_sample,
        }
        
        # Generate text response
        output = self.pipe(messages, **generation_args)
        return output[0]['generated_text']


# Example usage
if __name__ == "__main__":
    assistant = MicrosoftPhiMiniInstruct()    
    messages = [
        {"role": "system", "content": "You are a question-answering helper, designed to assist users with their questions by providing accurate and helpful response"},
        {"role": "user", "content": "Who is Socrates"}
    ]    
    response = assistant.generate_text_response(messages)
    print(response)