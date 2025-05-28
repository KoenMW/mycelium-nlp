from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from src.vector_db import VectorDB
import httpx


# Global instance
vector_db = VectorDB()

class Chat:
    def __init__(self):
        self.messages = []
        self.client = OpenAI(
            base_url = 'http://localhost:11434/v1',
            api_key='ollama', # required, but unused
            http_client=httpx.Client(trust_env=False)
        )

    def send(self, user_message: str):
        # Get similar context from vector DB
        context_results = vector_db.similarity_search(user_message, 10)

        # Format the context data (ignoring the score for now)
        context_texts = [text for text, score in context_results]
        context = "\n\n".join(context_texts)


        # Create temporary message list with system context
        temp_messages: list[ChatCompletionMessageParam] = [{"role": "system", "content": f"Context:\n{context}"}]
        temp_messages.extend(self.messages)
        temp_messages.append({"role": "user", "content": user_message})

        # Call LLM
        response = self.client.chat.completions.create(
            model="llama3.2",
            messages=temp_messages
        )

        # Save only user and assistant messages
        self.messages.append({"role": "user", "content": user_message})
        assistant_reply = response.choices[0].message.content
        self.messages.append({"role": "assistant", "content": assistant_reply})

        return assistant_reply

# Example usage:
# chat = Chat(model="local-llama-model")
# reply = chat.send("What is vector search?")
# print(reply)
