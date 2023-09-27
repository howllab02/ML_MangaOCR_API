from transformers import AutoTokenizer, VisionEncoderDecoderModel, AutoImageProcessor


class OCRModel:
    def __init__(self, weight_path, tokenizer, processor):
        self.weight = weight_path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.processor = AutoImageProcessor.from_pretrained(processor)

    def predict(self, img):
        model = VisionEncoderDecoderModel.from_pretrained(self.weight)
        pixel_values = self.processor(img, return_tensors="pt").pixel_values
        output = model.generate(pixel_values)
        generated_text = self.tokenizer.batch_decode(output, skip_special_tokens=True)[0]
        return generated_text
