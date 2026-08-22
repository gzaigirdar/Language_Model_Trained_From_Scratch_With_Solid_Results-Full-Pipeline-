"""
Test script to verify multi-turn dialogue support in Tiny_Chat_Inf.py
Simulates a conversation and prints the history at each turn.
"""
from Training_utils.buildModel import BuildModel
from transformers import AutoTokenizer
import torch
import re
import time

# ── Setup (same as Tiny_Chat_Inf.py) ──────────────────────────────
model_config = {
    "D_Model": 420,
    "Num_Heads": 6,
    "Num_Layers": 6,
    "Dropout": 0.05,
    "Vocab_size": 32105,
    "FeedForward_size": 2000,
    "Context_size": 200
}
tokenizer_path = 'Tokenizer/Saved_tokenizer/t5_Tokenizer'
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)

def clean_text(text):
    text = text.lower()
    tokens = re.findall(r"\w+(?:'\w+)*|[^\w\s]", text)
    return " ".join(tokens)

model_path = "Saved_Models/Tiny_Chat_41m_Pretrained/41m200T_pretrained_Tchat.pth"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
builder = BuildModel()
model = builder.createModel(model_config, Model_type='None')
model.to(device)
builder.load_weights(path=model_path)

# ── Generation function (same logic as Tiny_Chat_Inf.py) ──────────
def gen_text(prompt, model, tokenizer, max_tokens=80, pad_token=0,
             temperature=0.0, top_k=0, device=device):
    tokenized_text = tokenizer(prompt, return_tensors='pt',
                               add_special_tokens=False,
                               return_attention_mask=False,
                               padding=False, truncation=False)
    input_ids = tokenized_text['input_ids'].to(device)
    generated_tokens = []
    model.eval()

    for _ in range(max_tokens):
        with torch.no_grad():
            logits = model(input_ids, pad_token=pad_token)
            logits = logits[:, -1, :]

            if temperature > 0.0:
                logits = logits / temperature

            if top_k != 0:
                top_logits, _ = torch.topk(logits, top_k)
                min_val = top_logits[:, -1]
                logits = torch.where(logits < min_val,
                                     torch.tensor(float('-inf'), device=device),
                                     logits)

            if top_k != 0 or temperature > 0.0:
                probs = torch.softmax(logits, dim=-1)
                pred_index = torch.multinomial(probs, num_samples=1)
                pred_token = tokenizer.convert_ids_to_tokens(pred_index.tolist()[0])
                if pred_token[0] in ('<end>', '<user>', '<bot>'):
                    break
                generated_tokens.append(pred_index.squeeze(0).tolist()[0])
            else:
                pred_index = torch.argmax(logits, dim=-1)
                pred_token = tokenizer.convert_ids_to_tokens(pred_index.tolist())
                if pred_token[0] in ('<end>', '<user>', '<bot>'):
                    break
                generated_tokens.append(pred_index.tolist()[0])
                pred_index = pred_index.unsqueeze(0)

            input_ids = torch.cat([input_ids, pred_index], dim=-1)

    return tokenizer.decode(generated_tokens)

# ── Multi-turn test ───────────────────────────────────────────────
print("=" * 60)
print("MULTI-TURN DIALOGUE TEST")
print("=" * 60)

test_prompts = [
    "my name is alice",
    "what is my name?",
    "i like pizza",
    "what do i like?"
]

history = ""
for i, user_msg in enumerate(test_prompts):
    print(f"\n─── Turn {i+1} ───")
    print(f"User: {user_msg}")

    # Build the input for this turn
    user_msg_clean = clean_text(user_msg)
    history += f"{tokenizer.bos_token} <user> {user_msg_clean} <bot> "

    print(f"\n[DEBUG] History being sent to model:")
    print(f"  {repr(history)}\n")

    # Generate bot response
    bot_response = gen_text(history, model, tokenizer, top_k=0, temperature=0.0)
    print(f"\nBot: {bot_response}")

    # Append bot response to history
    history += bot_response + " "

    print(f"\n[DEBUG] History after appending bot response:")
    print(f"  {repr(history)}")
    print("-" * 60)

print("\n" + "=" * 60)
print("TEST COMPLETE")
print("If the model correctly recalled 'alice' in turn 2")
print("and 'pizza' in turn 4, multi-turn is working!")
print("=" * 60)