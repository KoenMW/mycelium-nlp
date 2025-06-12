from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from src.vector_db import VectorDB
import httpx
import json

MODEL="llama3.2"

# Global instance
vector_db = VectorDB()

class Chat:
    def __init__(self):
        self.messages: list[ChatCompletionMessageParam] = [
            {
                "role": "system",
                "content": (
                    "You are an assistant that helps users understand and explore the Mycelium Project. "
                    "This is a scientific and applied research project carried out by students at InHolland University of Applied Sciences in Haarlem. "
                    "The project is part of the Data Driven Smart Society program and focuses on using fungal mycelium for applications like bioremediation, sustainable materials, and environmental monitoring. "
                    "You provide detailed and factual answers in Dutch or English, depending on how the user asks the question. "
                    "Use the context provided from the vector database to inform your answers. If context is insufficient, you may ask for clarification or suggest how to refine the question."
                    "You will always respond"
                    "You will always responde in the same language as the user"
                )
            }
        ]
        self.client = OpenAI(
            base_url = 'http://localhost:11434/v1',
            api_key='ollama',
            http_client=httpx.Client(trust_env=False)
        )

    def contextCheck(self, question: str, failed_queries: list[str] = []):
        if len(failed_queries) >= 5:
            return vector_db.similarity_search(failed_queries[0])
        
        context_results = vector_db.similarity_search(question, 3)

        system_prompt = """
    You are a relevance-checker whose job is to decide whether the retrieved “context_results” 
    contains enough information to answer the user's question.  

    Output exactly one JSON object (no extra prose) matching this schema:

    {
    "sufficient": boolean,  // true if context_results already answers the question (in whole or in part), 
                            // or if no additional context is needed.  
                            // Otherwise false.
    "query": string,        // if sufficient=false: a BRAND-NEW SEARCH QUERY that helps find the missing information.  
                            // You may generate this query in English or Dutch, depending on what makes sense:
                            //   • If the snippets in context_results are primarily in Dutch, you may write the new query in Dutch.
                            //   • Otherwise, keep it in the same language as the original question.
                            // Do NOT simply rephrase the question verbatim. Add synonyms or related terms 
                            // to improve recall.  
                            // If sufficient=true, this must be the empty string "".
    "question": string,     // the EXACT original question (character-for-character). NEVER modify or paraphrase.
    "data": string          // if sufficient=true: ONLY the snippet(s) from context_results that directly answer 
                            // the question (concatenate multiple snippets with “ — ” if needed).  
                            // If sufficient=false: this must be the empty string "".
    }

    Rules:
    1. Output ONLY the JSON object above—no HTML, no extra keys, no explanatory text.
    2. To set “sufficient”:
    • Scan each snippet in context_results (which may be in Dutch or English).  
    • If you find explicit information (a direct answer or fact) that clearly addresses the user's question, set sufficient=true.  
        Otherwise set sufficient=false.
    3. If sufficient=true:
    • “query” = ""  
    • “data” = the exact snippet(s) (in Dutch or English) that answer the question.  
        - If there are multiple matching lines, join them with “ — ”.  
        - Do NOT add any commentary beyond the literal (or minimally paraphrased) text.
    4. If sufficient=false:
    • “query” must be a BRAND-NEW search string (in English or Dutch as appropriate) 
        that targets whatever information is missing.  
        - Do NOT echo or lightly rephrase the original question.  
        - Add extra keywords, synonyms, or related concepts to broaden the search.
        - Sometimes trying it in another language can be helpfull
    • “data” = "".
    5. No additional keys—only “sufficient”, “query”, “question”, and “data” are allowed.
        """

        user_prompt = f"""
    User question (do NOT modify):\n\"\"\"\n{question}\n\"\"\"\n

    Context retrieved from vector DB (raw snippets; may be in Dutch or English):\n{context_results}
    """

        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user",   "content": user_prompt.strip()}
        ]

        response = self.client.chat.completions.create(
            model=MODEL,
            messages=messages,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "context_check",
                    "description": "decides if the retrieved context answers the question, or if a new query is needed",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "sufficient": {
                                "type": "boolean",
                                "description": "true if context contains enough information to answer the question"
                            },
                            "query": {
                                "type": "string",
                                "description": "new search query when context is insufficient, or empty string if sufficient=true"
                            },
                            "question": {
                                "type": "string",
                                "description": "the EXACT original question"
                            },
                            "data": {
                                "type": "string",
                                "description": "relevant snippet(s) when sufficient=true, or empty string if sufficient=false"
                            }
                        },
                        "additionalProperties": False,
                        "required": ["sufficient", "query", "question", "data"]
                    },
                    "strict": True
                }
            }
        )

        content = response.choices[0].message.content
        if content:
             json_content = json.loads(content)
             if not json_content['sufficient']:
                 failed_queries.append(question)
                 return self.contextCheck(json_content['query'], failed_queries)
        return context_results



    def send(self, user_message: str):
        # Get similar context from vector DB
        context_results = self.contextCheck(user_message)

        # Format the context data (ignoring the score for now)
        context_texts = [text for text, _ in context_results]
        context = "\n\n".join(context_texts)

        temp_messages = self.messages
        temp_messages.append({"role": "system", "content": f"Contex related to the question:\n{context}"})
        temp_messages.append({"role": "user", "content": user_message})
        
        # Call LLM
        response = self.client.chat.completions.create(
            model=MODEL,
            messages=temp_messages
        )


        while not response.choices[0].message.content:
            temp_messages = self.messages
            temp_messages.append({
                "role": "system",
                "content": "the assistant should always respond to the user with something"
            })
            temp_messages.append({"role": "system", "content": f"Contex related to the question:\n{context}"})
            temp_messages.append({"role": "user", "content": user_message})
            response = self.client.chat.completions.create(
                model="llama3.2",
                messages=self.messages
            )

        self.messages.append({"role": "user", "content": user_message})
        assistant_reply = response.choices[0].message.content
        self.messages.append({"role": "assistant", "content": assistant_reply})

        return assistant_reply
