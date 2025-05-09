from datasets import load_dataset

dataset = load_dataset("adirik/fashion_image_caption-100")
print(dataset)


print(dataset["train"][0])

dataset["train"][0]["image"]

dataset = load_dataset("adirik/fashion_image_caption-100", split="train")
print(dataset)


from transformers import BlipProcessor, BlipForConditionalGeneration

preprocessor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

inputs = preprocessor(dataset[0]["image"], return_tensors="pt")


with torch.no_grad():
    outputs = model.generate(**inputs)

caption = preprocessor.decode(outputs[0], skip_special_tokens=True)

print(caption)



def replace_caption(data):
    inputs = preprocessor(data["image"], return_tensors="pt")

    with torch.no_grad():
        output = model.generate(**inputs)

    caption = preprocessor.decode(output[0], skip_special_tokens=True)
    
    data["text"] = caption
    return data

new_dataset = dataset.map(replace_caption)
