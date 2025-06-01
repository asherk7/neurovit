# preprocess images
from torchvision import transforms

def preprocess(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
        # normalize
    ])

    input_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    return input_tensor
