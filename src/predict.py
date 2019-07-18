from train import config
from fcn import get_model
import torch
from PIL import Image
from torchvision import transforms
from train import config, get_device
from imgs_dir_reader import from_dir_get_imgs_list
import numbers


class Predict:
    def __init__(self, model, cuda=True):
        self.model = model
        self.model.eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() and cuda else 'cpu')
        self.model = self.model.to(self.device)

    def predict_one_img(self, img_path):
        img, img_tensor = self.from_PIL_path_to_Tensor(img_path)
        img_tensor = img_tensor.unsqueeze(dim=0)
        img_tensor = img_tensor.to(self.device)
        print(f'img_tensor:{img_tensor.size()}')

        self.model.eval()
        self.model = self.model.to(self.device)

        pred = self.model(img_tensor)  # shape[1, n_class, h, w]
        pred = pred.argmax(dim=1, keepdim=True)  # shape[1, 1, h, w]
        print(f'pred:{pred.size()}')

    @staticmethod
    def from_PIL_path_to_Tensor(img_path, resize=224):
        img = Image.open(img_path)
        if resize is not None:

            assert isinstance(resize, numbers.Number) or (isinstance(resize, tuple) and len(resize) == 2)

            if isinstance(resize, numbers.Number):
                resize = (resize, resize)

            img = img.resize(resize)
        img_tensor = transforms.ToTensor()(img)
        return img, img_tensor


if __name__ == '__main__':

    args = config()
    model = get_model(name=args.model_name, n_class=args.n_class)
    predictor = Predict(model)

    img_dir = '../test_pic'
    img_list = from_dir_get_imgs_list(test_img_dir=img_dir)
    for img_path in img_list:
        predictor.predict_one_img(img_path)
    pass
