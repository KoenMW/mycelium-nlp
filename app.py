import os
from src.chat import Chat, MODEL
import src.utils as utils

def ensure_directories():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "./data"))
    raw_dir = os.path.join(base_dir, "raw")
    vector_store_dir = os.path.join(base_dir, "vector_store")

    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(vector_store_dir, exist_ok=True)

    if not os.listdir(raw_dir):
        print("Warning: './data/raw' is empty. No raw data will be used.")




def main():
    if not utils.server_ping():
        print("""
==============================
  Ollama Server Not Available
==============================

Please run the following command to start the server:

    ollama serve

Then, restart this script.
""")
        return
    
    if not utils.model_check():
        print(f"""
==============================
  {MODEL} not availlable 
==============================

Please run the following command to get the model:

    ollama pull {MODEL}

Then, restart this script.
""")
        return


    ensure_directories()

    chat = Chat()
    print("Start chatting! Type 'stop', 'exit', or 'quit' to end.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"stop", "exit", "quit"}:
            print("Goodbye!")
            break

        try:
            reply = chat.send(user_input)
            print("Assistant:", reply)
        except Exception as e:
            print("Error:", e)
            break

if __name__ == "__main__":
    main()
