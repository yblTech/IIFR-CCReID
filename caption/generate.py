import os
import json
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

model_dir = "../Qwen2-VL"  
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_dir, torch_dtype="auto", device_map="auto"
).eval()
# exit()
processor = AutoProcessor.from_pretrained(model_dir)

def load_existing_keys(filepath):
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            data = json.load(f)
        return set(data.keys())
    return set()

def save_to_json(data, output_path):
    if os.path.exists(output_path):
        with open(output_path, "r") as file:
            existing_data = json.load(file)
    else:
        existing_data = {}

    existing_data.update(data)

    with open(output_path, "w") as file:
        json.dump(existing_data, file, indent=4)

def gen5(dataset_dir, batch_size=100):
    # attributes = [
    #     "background description",  
    #     "clothing description",    
    #     "lighting description",    
    #     "occluded description",    
    #     "view description",        
    #     "pedestrian action",       
    #     "image quality"            
    # ]

    # template = (
    #     "The background features [background description], the person is wearing a [color] "
    #     "[upper body clothing type] with [upper body pattern] on the upper body and a [color] "
    #     "[lower body clothing type] with [lower body pattern] on the lower body, the brightness is "
    #     "[High/Medium/Low], the pedestrian is [occluded description] and facing "
    #     "[the camera/away from the camera/sideways to the camera], currently "
    #     "[standing/crouching/walking], and the overall image quality is [blurry/clear/poor]."
    # )
    attributes = [
        'body shape description',        # Description of shape
        'hair description',        # Description of hair
        'skin color description',  # Description of skin color
        'age description',         # Age of the person
        'gender description',      # Gender of the person
        'special features description'  # Any special features
    ]
    template = ("The person has a [body shape description], [hair description], their skin color "
                "is [skin color description], they appear to be [age description] years old, are identified "
                "as [gender description], and have the following special features: [special features description].")
    text = (
        f"Generate a description of the image using the following attributes: {', '.join(attributes)}. "
        f"Follow the template: \"{template}\". If some requirements in the template are not visible, "
        f"you can ignore them. Do not imagine any contents that are not in the image."
    )

    # Path initialization
    interference_json_path = "../int.json"
    existing_keys = load_existing_keys(interference_json_path)
    interference_info_dict = {}
    count = 0  # 当前批次计数
    total_count = 0  # 总计数

    
    identity_path = dataset_dir

    if os.path.isdir(identity_path):
            image_paths = [
                os.path.join(identity_path, img)
                for img in os.listdir(identity_path)
                if img.endswith(".png")
            ]
            history = None
            for image_path in image_paths:
                if image_path in existing_keys:
                    print(f"Skipping {image_path}, already processed.")
                    continue
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image_path},
                            {"type": "text", "text": text},
                        ],
                    }
                ]
                text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                image_inputs, _ = process_vision_info(messages)

                inputs = processor(
                    text=[text_input],
                    images=image_inputs,
                    padding=True,
                    return_tensors="pt",
                ).to("cuda")

                generated_ids = model.generate(**inputs, max_new_tokens=256)
                response = processor.batch_decode(
                    generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]

    
                interference_info_dict[image_path] = response
                count += 1
                total_count += 1

                if count == batch_size:
                    print(f"Processed {total_count} images, saving batch to JSON.")
                    save_to_json(interference_info_dict, interference_json_path)
                    interference_info_dict.clear()
                    count = 0

    # 保存剩余数据
    if interference_info_dict:
        print(f"Processed {total_count} images, saving the final batch to JSON.")
        save_to_json(interference_info_dict, interference_json_path)
def main():
    dataset_dir = '../dataset'  # Replace with your training dataset path
    if os.path.isdir(dataset_dir):
        gen5(dataset_dir)
    else:
        print("Invalid dataset directory.")

if __name__ == "__main__":
    main()