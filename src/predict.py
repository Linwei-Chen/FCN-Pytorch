from train import config
from fcn import get_model
import torch
from PIL import Image
from torchvision import transforms
from train import config, get_device
from imgs_dir_reader import from_dir_get_imgs_list
import numbers
from dataset import CLASSES, COLORMAP


class Predict:
    def __init__(self, model, cuda=True):
        self.model = model
        self.model.eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() and cuda else 'cpu')
        self.model = self.model.to(self.device)
        self.color_map = torch.Tensor(COLORMAP)

    def predict_one_img(self, img_path):
        img, img_tensor = self.from_PIL_path_to_Tensor(img_path)
        img_tensor = img_tensor.unsqueeze(dim=0)
        img_tensor = img_tensor.to(self.device)
        print(f'img_tensor:{img_tensor.size()}')

        self.model.eval()
        self.model = self.model.to(self.device)

        pred = self.model(img_tensor)  # shape[1, n_class, h, w]
        pred = pred.argmax(dim=1, keepdim=True)  # shape[1, 1, h, w]
        pred = pred.squeeze()  # shape[ h, w]
        print(f'pred size:{pred.size()}\npred:{pred}')
        pred_img = self.from_pred_to_color_img(pred)
        pred_img.show()

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

    def from_pred_to_color_img(self, pred):
        h, w = pred.size()
        temp = torch.zeros((3, h, w))
        for i in range(h):
            for j in range(w):
                temp[:, i, j] = self.color_map[pred[i, j]]
        return transforms.ToPILImage()(temp)

    @staticmethod
    def merge_two_img(img1, img2):
        pass


if __name__ == '__main__':

    args = config()
    model = get_model(name=args.model_name, n_class=args.n_class)
    predictor = Predict(model)

    img_dir = '../test_pic'
    img_list = from_dir_get_imgs_list(test_img_dir=img_dir)
    for img_path in img_list:
        predictor.predict_one_img(img_path)
    pass
