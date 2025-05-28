from src.chat import Chat

def main():
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
