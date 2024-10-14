import os
import sys
import time
from src.services.natural_questions_data_service import NaturalQuestionsParser
import logging

from src.services.ollama_service import OllamaChat

# Add the project root directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def log_time_taken(step_name, start_time):
    
    elapsed_time = time.time() - start_time
    if elapsed_time > 60:
        minutes = int(elapsed_time // 60)
        seconds = elapsed_time % 60
        logging.info(f"Time taken for {step_name}: {minutes} minutes and {seconds:.2f} seconds")
    else:
        logging.info(f"Time taken for {step_name}: {elapsed_time:.2f} seconds")
        
        
def generate(file_path: str, destination_file: str, role_prompt, start_line, stop_line):
    # Start measuring time
    total_start_time = time.time()

    # 1. Parse initial file
    parser = NaturalQuestionsParser(file_path)
    logging.info("Read entities")
    step_start_time = time.time()
    qa_entities = parser.parse_entities(start_line=start_line, stop_line=stop_line)
    log_time_taken("parsing entities", step_start_time)

    # 2. Send questions to the LLM models and collect responses from the LLMs    
    logging.info("Chat with Ollama models")
    step_start_time = time.time()
    ollamaChat = OllamaChat()
    msgEntities = ollamaChat.get_text_response_with_cascade_of_models(role_prompt, qa_entities)
    log_time_taken("chatting with models", step_start_time)

    # 3. Create embeddings with Ollama and add them to the dataset
    logging.info("Create embeddings with Ollama")
    step_start_time = time.time()
    msgEntities = ollamaChat.get_embeddings_with_cascade_of_models(msgEntities)
    log_time_taken("creating embeddings", step_start_time)

    # 4. Save results
    logging.info("Store results")
    step_start_time = time.time()
    parser.store_llms_responses(msgEntities, destination_file)
    log_time_taken("storing results", step_start_time)

    # Total time
    log_time_taken("total execution", total_start_time)


START_LINE = 0
STOP_LINE = 15             

if __name__ == "__main__":
    logging.info("====> Generate Correct Responses")
    generate("./data/Natural-Questions-Base-2k.csv", "./data/NQ-LLM-responses.csv", "", START_LINE, STOP_LINE)
    
    logging.info("====> Generate Factual Inaccuracy")
    prompt_rule = "Your have to generate 'Factual Inaccuracy' hallucination in your response. Factual Inaccuracy: The model presents incorrect facts, such as wrong numbers, dates, or other factual information, but doesn't invent completely new data."    
    generate("./data/Natural-Questions-Base-2k.csv", "./data/Factual-Inaccuracy-NQ-LLM-responses.csv", prompt_rule, START_LINE, STOP_LINE)

    logging.info("====> Generate Misinterpretation")
    prompt_rule = "Your have to generate 'Misinterpretation' hallucination in your response. Misinterpretation: The model misunderstands the context or the user's query, generating a response that is not factually incorrect but doesn't address the core of the question."    
    generate("./data/Natural-Questions-Base-2k.csv", "./data/Misinterpretation-NQ-LLM-responses.csv", prompt_rule, START_LINE, STOP_LINE)
    
    logging.info("====> Generate Needle in a Haystack")
    prompt_rule = "Your have to generate 'Needle in a Haystack' hallucination in your response. 'Needle in a Haystack:' The model struggles to extract relevant details from a large body of information, leading to incomplete or overly general responses."    
    generate("./data/Natural-Questions-Base-2k.csv", "./data/Needle-Haystack-NQ-LLM-responses.csv", prompt_rule, START_LINE, STOP_LINE)
    
    logging.info("====> Generate Fabrication")
    prompt_rule = "Your have to generate 'Fabrication' hallucination in your response. Fabrication: The model generates entirely false information that has no basis in reality or the training data."    
    generate("./data/Natural-Questions-Base-2k.csv", "./data/Fabrication-NQ-LLM-responses.csv", prompt_rule, START_LINE, STOP_LINE)
    
    logging.info("====> Generate Structural Hallucination")
    prompt_rule = "Your have to generate 'Structural Hallucination' hallucination in your response. Structural Hallucination: The model produces erroneous information based on flawed structural reasoning or logic, which results in plausible-sounding but incorrect conclusions"
    generate("./data/Natural-Questions-Base-2k.csv", "./data/Structural-Hallucination-NQ-LLM-responses.csv", prompt_rule, START_LINE, STOP_LINE)