import os 
import zmq
import time
from LeidosLLM import LLM

if __name__ == "__main__":

    class config: 
        port = 8888
        model_desired = "meta-llama/Llama-2-7b-chat-hf" # "garrachonr/LlamaDos" # "meta-llama/Llama-2-13b-chat-hf" # "brockwilson12/Llama2-8bit" # "clibrain/Llama-2-7b-ft-instruct-es"
        bad_words = None # ["¡", "Great", "Perfecto", "Entendido", "!", "(", ")", "Haha"]
        similarity_model = None # 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'

    # Set up connection
    context = zmq.Context()

    # Context manager
    with context.socket(zmq.REP) as socket:
        # Bind
        socket.bind(f"tcp://*:{config.port}")

        # Display memory 
        os.system("nvidia-smi")

        # Model
        llm = LLM(llm_model=config.model_desired,
                  access_token="",
                  bad_words=config.bad_words, 
                  similar_model=config.similarity_model,
                )

        print(f"\n[START-UP] Loaded model and configuration, starting Server on port {config.port}...")
        print("\n\t[INFO] Waiting for client input")

        server_active = True
        try:
            while server_active:
                # Wait for user input 
                socket_msg = socket.recv()
                response = socket_msg.decode("utf-8")
                print(f"\n\t[INFO] Received: {response}")

                # Add user input to LLM context window
                llm.context_window.append({"role": "user", "content": response})

                # Compute question from user input
                question = llm.generate_question()

                # Send question
                socket.send_string(question)
                print(f"\n\t[INFO] Sent: '{question}'")
        except KeyboardInterrupt as e:
            print(f"\n[ERR] Detected user interrupt {e}, shutting server down")
        
        finally:
            print("\n[EXIT] Server shutting down...")
            socket.close()
