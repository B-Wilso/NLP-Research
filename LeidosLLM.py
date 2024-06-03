from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import login, logout
import torch
from sentence_transformers import SentenceTransformer, util


class LLM:
    def __init__(self, 
                 llm_model : str, 
                 bad_words : list = None,
                 similar_model : str = None, 
                 bnb_config : BitsAndBytesConfig = BitsAndBytesConfig(
                     load_in_4bit=True,
                     bnb_4bit_use_double_quant=True,
                     bnb_4bit_quant_type="nf4",
                     bnb_4bit_compute_dtype=torch.bfloat16
                     )) -> None:
        """
        On initialization:
        - Login to huggingface
        - Load specific LLM and its tokenizer
        - Load bad words or similarity model if they are input
        - Load new random context window
        """
        login("")

        self.llm = AutoModelForCausalLM.from_pretrained(llm_model,
                                            device_map='auto',
                                            quantization_config = bnb_config
                                            )
        
        self.llm_tokenizer =  AutoTokenizer.from_pretrained(llm_model)

        self.tokenizer_with_prefix_space = AutoTokenizer.from_pretrained(llm_model, add_prefix_space=True)

        if bad_words:
            self.bad_words_ids = self.__get_tokens_as_list(bad_words)

        if similar_model:
            self.similar = SentenceTransformer(similar_model)

        self.__init_agent()
    
    def __del__(self):
        """On deletion; logout of huggingface"""
        print("Deleting OBJ")
        logout()
    
    def __init_agent(self) -> None:
        """Use demo generator to return new random system prompt"""
        self.context_window = [{"role": "system", "content": "Always speak in Spanish. You are an airport customs agent. Interview the User in Spanish relating to travel security. Never speak in English. Ask only a single question."}]
    
    def __get_tokens_as_list(self, word_list: list) -> list:
        "Converts a sequence of words into a list of tokens"
        tokens_list = []
        for word in word_list:
            tokenized_word = self.tokenizer_with_prefix_space([word], add_special_tokens=False).input_ids[0]
            tokens_list.append(tokenized_word)
        return tokens_list

    def __detect_similarity(self, unparsed : list):
        """Simluarity of tokens based on model and its cos similarity encoding"""
        parsed = []
        for x in unparsed:
            if x["role"] == "assistant":
                parsed.append(x["content"])
        target = parsed[-1]
        sentences = parsed[:-1]

        # print(target, sentences)
        if not sentences:
            return  0.0

        t_embeddings = self.similar.encode(target, convert_to_tensor=True)
        s_embeddings = self.similar.encode(sentences, convert_to_tensor=True)

        return util.cos_sim(t_embeddings, s_embeddings)
    
    def __style(self, last_output : str):
        """Eventual stylization of of outputs from LLM"""
        pass
        # instruction = f"""Hello. Can you tell me about your travel itinerary? => Can you tell me about your travel itinerary? 
        # Hello. Can you please tell me where you are traveling from and where you will be traveling to? => Can you tell me where you are traveling from and where you will be traveling to?
        # AI: Has anyone approached you and asked you to carry anything for them during your trip? => Has anyone approached you and asked you to carry anything for them during your trip?
        # {last_output} =>"""
        # system_prompt = """Change the last sentence to fit the correct format. Only respond with the last formatted sentence."""

        # template = custom_get_prompt(instruction, system_prompt)
        
        # return forward(template)

    def detect_intent(self, last_output : str):
        """
        Eventual Intent Detection from user inputs 
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
        
        return self.forward_pass(intent_frame)
    
    def generate_question(self) -> str:
        """Forward pass current context window and append output"""
        question = self.__forward_pass(self.context_window)
        self.context_window.append({"role": "assistant", "content": question})
        return question
    
    def __forward_pass(self, messages : list) -> str:
        """Generic forward pass through LLM"""
        inputs = self.llm_tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
        inputs = inputs.to(0)
        output = self.llm.generate(inputs, bad_words_ids=self.bad_words_ids, do_sample=True, top_k=30, max_new_tokens=512)
        response = self.llm_tokenizer.decode(output[0][len(inputs[0]):-1])
        return response
