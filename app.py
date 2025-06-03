import os
from src.chat import Chat

def ensure_directories():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "./data"))
    raw_dir = os.path.join(base_dir, "raw")
    vector_store_dir = os.path.join(base_dir, "vector_store")

    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(vector_store_dir, exist_ok=True)

    if not os.listdir(raw_dir):
        print("Warning: './data/raw' is empty. No raw data will be used.")

def main():
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
