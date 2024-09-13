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

if __name__ == "__main__":

    # Start measuring time
    total_start_time = time.time()

    # 1. Parse initial file
    parser = NaturalQuestionsParser("./data/Natural-Questions-Base-2k.csv")
    logging.info("Read entities")
    step_start_time = time.time()
    qa_entities = parser.parse_entities(start_line=3, stop_line=10)
    log_time_taken("parsing entities", step_start_time)

    # 2. Send questions to the LLM models and collect responses from the LLMs    
    logging.info("Chat with Ollama models")
    step_start_time = time.time()
    ollamaChat = OllamaChat()
    msgEntities = ollamaChat.get_text_response_with_cascade_of_models(qa_entities)
    log_time_taken("chatting with models", step_start_time)

    # 3. Create embeddings with Ollama and add them to the dataset
    logging.info("Create embeddings with Ollama")
    step_start_time = time.time()
    msgEntities = ollamaChat.get_embeddings_with_cascade_of_models(msgEntities)
    log_time_taken("creating embeddings", step_start_time)

    # 4. Save results
    logging.info("Store results")
    step_start_time = time.time()
    parser.store_llms_responses(msgEntities, "./data/Natural-Questions-LLM-responses.csv")
    log_time_taken("storing results", step_start_time)

    # Total time
    log_time_taken("total execution", total_start_time)