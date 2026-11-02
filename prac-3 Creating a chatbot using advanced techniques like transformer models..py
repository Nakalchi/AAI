
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Step 1: Load Pre-trained Model and Tokenizer
print("Loading the DialoGPT model...")

model_name = "microsoft/DialoGPT-medium"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Step 2: Initialize Chat History
chat_history_ids = None
step = 0

# Step 3: Chat with the User
print("\nChatbot is ready! Type 'exit' to end the chat.\n")

while True:

    # Get user input
    user_input = input("You: ")

    if user_input.lower() == "exit":
        print("Chatbot: Goodbye!")
        break

    # Encode user input
    new_input_ids = tokenizer.encode(
        user_input + tokenizer.eos_token,
        return_tensors="pt"
    )

    # Add new input to chat history
    if step > 0:
        chat_history_ids = torch.cat(
            [chat_history_ids, new_input_ids],
            dim=-1
        )
    else:
        chat_history_ids = new_input_ids

    # Generate chatbot response
    response_ids = model.generate(
        chat_history_ids,
        max_length=1000,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_k=50,
        top_p=0.95
    )

    # Decode chatbot response
    response = tokenizer.decode(
        response_ids[:, chat_history_ids.shape[-1]:][0],
        skip_special_tokens=True
    )

    # Display response
    print("Chatbot:", response)

    # Update chat history
    chat_history_ids = response_ids

    step += 1
