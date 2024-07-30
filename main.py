import gradio as gr
import faiss
import numpy as np
import llm


# Initialize FAISS index
index = faiss.IndexFlatL2(1536)  # Assuming 1536-dimensional embeddings

# Function to handle document upload and embedding
def handle_upload(file):
    global index
    print(type(file), file)
    text = file.read().decode("utf-8")
    chunks = chunk_text(text)
    embeddings = embed_text(chunks)
    
    for embedding in embeddings:
        index.add(np.array([embedding], dtype=np.float32))

    return "Document processed and embedded successfully."

# Function to generate response using RAG
def generate_response(user_input, history, embeddings_index):
    # Embed user input
    user_embedding = embed_text([user_input])[0]

    # Search the index for similar embeddings
    D, I = embeddings_index.search(np.array([user_embedding], dtype=np.float32), k=3)
    
    # Generate the response using the top similar chunks
    context = ""
    for idx in I[0]:
        if idx != -1:  # Check if there are valid indices
            context += " " + chunks[idx]

    # Generate response with context
    messages = [{"role": "system", "content": context}, {"role": "user", "content": user_input}]
    print(messages)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )
    bot_reply = response.choices[0].message.content

    history.append(("user", user_input))
    history.append(("bot", bot_reply))

    return format_conversation(history), history

def format_conversation(conversation):
    formatted_conversation = ""
    for role, message in conversation:
        if role == "user":
            formatted_conversation += f"You: {message}\n"
        else:
            formatted_conversation += f"GPT: {message}\n"
    return formatted_conversation.strip()

# Create the chat interface
with gr.Blocks() as demo:
    chat_history = gr.State([])  # Initialize state for conversation history
    embeddings_index = gr.State(index)  # Initialize state for embeddings index
    chat_display = gr.Textbox(label="Conversation", placeholder="Conversation will appear here...", interactive=False, lines=20)
    user_input = gr.Textbox(label="Your Message", placeholder="Type your message here...", lines=1)
    send_button = gr.Button("Send")
    upload_button = gr.File(label="Upload Document", type="binary")

    def send_message(user_input, chat_history, embeddings_index):
        chat_display_value, updated_history = generate_response(user_input, chat_history, embeddings_index)
        return chat_display_value, updated_history

    send_button.click(send_message, inputs=[user_input, chat_history, embeddings_index], outputs=[chat_display, chat_history])
    user_input.submit(send_message, inputs=[user_input, chat_history, embeddings_index], outputs=[chat_display, chat_history])
    upload_button.upload(handle_upload)

if __name__ == "__main__":
    demo.launch(share=True)
