import os
import zmq
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import login, logout
import torch
from sentence_transformers import SentenceTransformer, util


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

# Login to model
login("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")


def get_tokens_as_list(word_list):
    "Converts a sequence of words into a list of tokens"
    tokens_list = []
    for word in word_list:
        tokenized_word = tokenizer_with_prefix_space([word], add_special_tokens=False).input_ids[0]
        tokens_list.append(tokenized_word)
    return tokens_list

# Load the model and tokenizer
print("\n[START-UP] Loading models...")

similarity_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

model_desired = "meta-llama/Llama-2-7b-chat-hf" # "garrachonr/LlamaDos" # "meta-llama/Llama-2-13b-chat-hf" # "brockwilson12/Llama2-8bit" # "clibrain/Llama-2-7b-ft-instruct-es"

tokenizer = AutoTokenizer.from_pretrained(model_desired)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(model_desired,
                                            device_map='auto',
                                            quantization_config = bnb_config
                                            )

tokenizer_with_prefix_space = AutoTokenizer.from_pretrained(model_desired, add_prefix_space=True)

bad_words_ids = get_tokens_as_list(["¡", "Great", "Perfecto", "Entendido", "!", "(", ")", "Haha"])

print("\n[START-UP] Models loaded successfully")

def forward_pass(messages, bad_words_ids=bad_words_ids):
    inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
    inputs = inputs.to(0)
    output = model.generate(inputs, bad_words_ids=bad_words_ids, do_sample=True, top_k=30, max_new_tokens=512)
    response = tokenizer.decode(output[0][len(inputs[0]):-1])
    return response

def detect_intent(last_output):
    """
    0: Response
    1: Question
    2: Refusal
    """
    
    system_prompt = "Determine if the given sentence is a Response, Question, or Refusal."
    
    intent_frame = [{"role": "system", "content": system_prompt}, 
                    {"role": "user", "content": "Voy a enseñar nuevos reclutas"},
                    {"role": "assistant", "content": "Response"},
                    {"role": "user", "content": "Sí"},
                    {"role": "assistant", "content": "Response"},
                    
                    {"role": "user", "content": "Por qué estoy aquí"},
                    {"role": "assistant", "content": "Question"},
                    {"role": "user", "content": "Por qué quieres saber eso"},
                    {"role": "assistant", "content": "Question"},
                    
                    {"role": "user", "content": "Hola"},
                    {"role": "assistant", "content": "Response"},
                   
                    {"role": "user", "content": "No tengo libertad para decir"},
                    {"role": "assistant", "content": "Refusal"},
                    {"role": "user", "content": "No deseo responder eso"},
                    {"role": "assistant", "content": "Refusal"},
                    
                    {"role": "user", "content": f"{last_output}"},
                   ]
    
    return forward_pass(intent_frame)

def detect_similarity(unparsed):
    # print("\t[SIM]")
    parsed = []
    for x in unparsed:
        if x["role"] == "assistant":
            parsed.append(x["content"])
    target = parsed[-1]
    sentences = parsed[:-1]

    # print(target, sentences)
    if not sentences:
        return  0.0

    t_embeddings = similarity_model.encode(target, convert_to_tensor=True)
    s_embeddings = similarity_model.encode(sentences, convert_to_tensor=True)

    return util.cos_sim(t_embeddings, s_embeddings)


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

    print(f"\n[START-UP] Loaded model and configuration, starting Server on port {port}...")

    # Initial Question Bank
    question_bank = ["¿Cómo le ha ido en su viaje?",
                    "¿Ha notado algo sospechoso durante sus viajes hasta este momento?",
                    "¿Quién empaco sus maletas para el viaje?",
                    "¿Estará trabajando en la embajada de Estados Unidos, verdad?",
                    "¿Tiene con usted documentos oficiales que lo comprueben?",
                    "¿Cuánto tiempo lleva usted como parte de la Fuerza Aérea?",
                    "¿Cuándo fue que compro usted los boletos del vuelo?",
                    "¿Cómo se transportara usted hacia su destino?",
                    ]

    # Create initial messages list
    messages = [{"role": "system", "content": "Always speak in Spanish. You are an airport customs agent. Interview the User in Spanish relating to travel security. Never speak in English. Ask only a single question."}]

    print("\n\t[INFO] Waiting for client input")
    server_active = True
    first_input = True
    while server_active:

        for _ in range(2):
            socket_msg = socket.recv()
            response = socket_msg.decode("utf-8")
            print(f"\n\t[INFO] Received: {response}")

            messages.append({"role": "user", "content": response})

            # Intent detection
            print(f"\n\t[INFO] Detected intent: {detect_intent(response)}")

            # Compute question from user input
            question = forward_pass(messages)

            # Add computed question
            messages.append({"role": "assistant", "content": question})

            # Get Similarity
            print(f"\n\t[INFO] Simularity Score to other sentences: {detect_similarity(messages)}")

            # Send question
            socket.send_string(question)
            print(f"\n\t[INFO] Sent: '{question}'") # \n\t>>Current history: \n{messages}

        print("\n\t[INFO] Exiting loop.")

        # Receive input
        socket_msg = socket.recv()
        response = socket_msg.decode("utf-8")
        print(f"\n\t[INFO] Received: {response}")
        
        if response == "EXIT":
            print("\n\t[EXIT]")
            break

        # Add user input
        messages.append({"role": "user", "content": response})

        # Intent detection
        print(f"\n\t[INFO] Detected intent: {detect_intent(response)}")


        # New question from bank
        question = question_bank.pop(0)
        messages.append({"role": "assistant", "content": question})

        # Send popped question
        socket.send_string(question)
        print(f"\n\t[INFO] Sent: '{question}'") # \n\t>>Current history: \n{messages}
        print("\n\t[INFO] Entering loop.")


    logout()
    print("\n[INFO] All responses have been processed.")
    print("\n[EXIT] Server shutting down...")
    print("\n\n\n", messages)
