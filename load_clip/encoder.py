import torch
import json
import os
from clip_model import build_CLIP_from_openai_pretrained
from tokensizer import SimpleTokenizer
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
def tokenize(caption: str, tokenizer, text_length=77, truncate=True) -> torch.LongTensor:
    sot_token = tokenizer.encoder["<|startoftext|>"]
    eot_token = tokenizer.encoder["<|endoftext|>"]
    tokens = [sot_token] + tokenizer.encode(caption) + [eot_token]

    result = torch.zeros(text_length, dtype=torch.long)
    if len(tokens) > text_length:
        if truncate:
            tokens = tokens[:text_length]
            tokens[-1] = eot_token
        else:
            raise RuntimeError(
                f"Input {caption} is too long for context length {text_length}"
            )
    result[:len(tokens)] = torch.tensor(tokens)
    return result

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model, base_cfg = build_CLIP_from_openai_pretrained('../ViT-B-16.pt', (384, 128), 16)
tokenizers = SimpleTokenizer()
model.cuda().eval()
json_file_path = '../caption/interference.json'  
with open(json_file_path, 'r') as f:
    text_data = json.load(f)

texts = list(text_data.values())  
image_paths = list(text_data.keys())  
batch_size = 2048
output_file = '../caption/tt_int.pt'
features_with_paths = {}

for i in range(0, len(texts), batch_size):
    if batch_size>len(texts)-i:
        batch_size = len(texts)-i
    batch_texts = texts[i:i + batch_size]
    batch_image_paths = image_paths[i:i + batch_size]
    with torch.no_grad():
        inputs = []
        for j in range(batch_size):
            input = tokenize(batch_texts[j], tokenizer=tokenizers, text_length=77, truncate=True).to(device)  # 加入truncation=True
            inputs.append(input)
        inputs = torch.stack(inputs).to(device)
        batch_features = model.encode_text(inputs)  
        for j in range(batch_size):
            features_with_paths["tf"+batch_image_paths[j]]=batch_features[0][j].cpu()
            features_with_paths["af"+batch_image_paths[j]]=batch_features[1][j].cpu()
            features_with_paths["tt"+batch_image_paths[j]]=inputs[j].cpu()

    del inputs
    del batch_features
    torch.cuda.empty_cache() 
    torch.save(features_with_paths, output_file)
    if batch_size>len(texts)-i:
        batch_size = len(texts)
    print(f"Processed {i + batch_size} / {len(texts)} entries and saved to file.")

print(f"All text features have been processed and saved to {output_file}.")