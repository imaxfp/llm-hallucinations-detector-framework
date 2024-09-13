from typing import List
import ollama

# RUN the next models wiht OLLAMA - https://ollama.com/library
LLM_MODELS = {"llama3.1", "gemma2", "phi3"}

class OllamaChat:
    def __init__(self, models=LLM_MODELS):        
        self.models = models

    def chat_with_ollama(self, model_name, msg):
        stream = ollama.chat(
            model=model_name,            
            messages=[
                {'role': 'system', 'content': 'You are a question-answer helper. Analyze the question and provide a concise answer.'},
                {'role': 'system', 'content': 'Response have to be about 50 or 60 words.'},
                {'role': 'system', 'content': 'PAY ATTENTION YOUR Response have to be about 50 or 60 words.'},
                {'role': 'user', 'content': msg}
            ],
            stream=True
        )
        res = []
        for chunk in stream:
            res.append(chunk['message']['content'])
        return ''.join(res)
    
    def text_to_embedding(seld, model_name, txt):
        embed = ollama.embed(model=model_name, input=txt)
        return embed        

    def get_text_response_with_cascade_of_models(self, msgEntities: List):                      
        for msgEntity in msgEntities:
            res_dir = {}
            for mod_name in self.models:                                  
                res_dir[mod_name] = self.chat_with_ollama(mod_name, msgEntity.question)
            msgEntity.llm_answers = res_dir
        return msgEntities         
            
    def get_embeddings_with_cascade_of_models(self, msgEntities: List):                      
        for msgEntity in msgEntities:
            embed_dir = {}            
            true_answer_embed_dir = {}
            for mod_name, llm_answ in msgEntity.llm_answers.items():                                           
                res = self.text_to_embedding(mod_name, llm_answ)                                                
                embed_dir[res.get('model')] = res.get('embeddings')
                
                true_embed = self.text_to_embedding(mod_name, msgEntity.true_answer)    
                true_answer_embed_dir[true_embed.get('model')] = true_embed.get('embeddings')
                
            msgEntity.llm_embeddings = embed_dir 
            msgEntity.true_answer_embeddings = true_answer_embed_dir 
                                        
        return msgEntities 