import onnxruntime
from tokenizers import Tokenizer
import numpy as np
import mobileclip
from PIL import Image
import numpy as np
import torch
from io import BytesIO

class TextVectorizingModel():
    def __init__(self, filename="bge.quant.onnx"):
        self.model = onnxruntime.InferenceSession(filename)
        self.inputs = self.model.get_inputs()
        self.outputs = self.model.get_outputs()
        self.tokenizer = Tokenizer.from_pretrained("BAAI/bge-m3")


    def vectorize(self, sentence: str) -> np.ndarray: 
        encoded = self.tokenizer.encode(sentence)
        _, sentence_embedding = self.model.run(output_names=None, 
                                               input_feed={self.inputs[0].name:[encoded.ids], 
                                                           self.inputs[1].name:[encoded.attention_mask]})
        return sentence_embedding

class ImageVectorizingModel():
    def __init__(self, filename="./models/mobileclip_s0.pt"):
        self.model, _, self.preprocess = mobileclip.create_model_and_transforms('mobileclip_s0', pretrained=filename)        
        self.tokenizer = mobileclip.get_tokenizer('mobileclip_s0')
    
    def vectorize(self, image) -> np.ndarray: 
        if isinstance(image, str):
            im=Image.open(image)
        if isinstance(image, bytes):
            im=Image.open(BytesIO(image))
        features = self.preprocess(im.convert('RGB')).unsqueeze(0)
        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = self.model.encode_image(features).numpy()
        return image_features