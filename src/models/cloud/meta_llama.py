import transformers
import torch
from huggingface_hub import login

class MetaLlama:
        
    def __init__(self, model_id="meta-llama/Meta-Llama-3.1-8B-Instruct", device_map="auto"):
        
        # Private attributes
        self._model_id = model_id
        self._device_map = device_map
        # Initialize the pipeline in the constructor
        self._pipeline = transformers.pipeline(
            "text-generation",
            model=self._model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map=self._device_map,
        )
    
    @property
    def model_id(self):
        return self._model_id
    
    @property
    def device_map(self):
        return self._device_map
    
    @property
    def pipeline(self):
        return self._pipeline

    def generate_text_response(self, message):
        # Set up the initial conversation context
        messages = [
            {"role": "system", "content": "You are a question-answering helper, designed to assist users with their questions by providing accurate and helpful response"},
            {"role": "user", "content": message}
        ]

        # Generate the response using the pipeline
        outputs = self.pipeline(
            message,  # Change messages to message as the input should be text, not conversation history
            max_new_tokens=256,
        )
        return outputs
    
if __name__ == "__main__":
    import transformers
    import torch

    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )

    messages = [
        {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
        {"role": "user", "content": "Who are you?"},
    ]

    outputs = pipeline(
        messages,
        max_new_tokens=256,
    )
    print(outputs[0]["generated_text"][-1])
        
    #metaLlama = MetaLlama()        
    #metaLlama.generate_text_response("Who is Socrates?")    