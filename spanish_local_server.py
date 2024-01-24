import os
import zmq
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login, logout

"""
Initialize custom prompting to resolve LLAMA2 issues
Chat -> initial question to ask client
Stylize -> ability to change the output question of model to different formatting/scenarios
Forward -> non-pipeline forward through LLAMA2


TODO
-Build a better stylize function
-Consider different memory methods to work with sylization
-Spanish system prompts
-Question bank 
"""

def get_tokens_as_list(word_list):
    "Converts a sequence of words into a list of tokens"
    tokens_list = []
    for word in word_list:
        tokenized_word = tokenizer_with_prefix_space([word], add_special_tokens=False).input_ids[0]
        tokens_list.append(tokenized_word)
    return tokens_list

# Load the model and tokenizer
print("\n\t**Loading models...")

model_desired = "meta-llama/Llama-2-7b-chat-hf" # "garrachonr/LlamaDos" # "meta-llama/Llama-2-13b-chat-hf" # "brockwilson12/Llama2-8bit" # "clibrain/Llama-2-7b-ft-instruct-es"

tokenizer = AutoTokenizer.from_pretrained(model_desired)

model = AutoModelForCausalLM.from_pretrained(model_desired,
                                            device_map='auto',
                                            load_in_4bit=True
                                            )

tokenizer_with_prefix_space = AutoTokenizer.from_pretrained(model_desired, add_prefix_space=True)

bad_words_ids = get_tokens_as_list(["¡", "Perfecto", "Entendido", "!", "(", ")", "Haha"])

print("\n\t**Models loaded successfully")

def forward_pass(msgs, bad_words_ids=bad_words_ids):
    inputs = tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,  return_tensors='pt')
    inputs = inputs.to(0)
    output = model.generate(inputs, bad_words_ids=bad_words_ids, do_sample=True, top_k=30, max_new_tokens=512)
    print(tokenizer.decode(output[0]))
    response = tokenizer.decode(output[0][len(inputs[0]):-1])
    return response


# def form_template(response, user_in):
#     prompt = response[4:]
#     prompt += "<s>[INST]" + user_in + "[/INST]"
#     return prompt


# def style(last_output):
#     instruction = f"""Hello. Can you tell me about your travel itinerary? => Can you tell me about your travel itinerary? 
#     Hello. Can you please tell me where you are traveling from and where you will be traveling to? => Can you tell me where you are traveling from and where you will be traveling to?
#     AI: Has anyone approached you and asked you to carry anything for them during your trip? => Has anyone approached you and asked you to carry anything for them during your trip?
#     {last_output} =>"""
#     system_prompt = """Change the last sentence to fit the correct format. Only respond with the last formatted sentence."""

#     template = custom_get_prompt(instruction, system_prompt)
    
#     return forward(template)


if __name__ == "__main__":
    
    # Login to model
    login("")

    port = 8888

    # Set up connection
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://*:{port}")

    # Display memory 
    os.system("nvidia-smi")

    print(f"\n\t**Loaded model and configuration, starting Server on port {port}...")

    # Create initial messages list
    messages = [{"role": "system", "content": "You are an airport customs agent. Interview the User in Spanish relating to travel security. Always speak in Spanish. Never speak in English. Ask only a single question."}]

    print("\n\t>>Waiting for input")
    server_active = True
    first_input = True
    while server_active:

        # Receive input
        socket_msg = socket.recv()
        response = socket_msg.decode("utf-8")
        print(f"\n\t>>Received: {response}")
        
        if response == "EXIT":
            break

        # Add user input
        messages.append({"role": "user", "content": response})

        # Compute question from user input
        question = forward_pass(messages)

        # Add computed question
        messages.append({"role": "assistant", "content": question})

        # Send question
        socket.send_string(question)
        print(f"\n\t>>Sent: '{question}'\n\t>>Current history: \n{messages}")


    logout()
    print("\n\t**All responses have been processed.")
    print("\n\t**Server shutting down...")
