from train import config
from fcn import get_model
import torch
from PIL import Image
from torchvision import transforms
from train import config, get_device
from imgs_dir_reader import from_dir_get_imgs_list
import numbers
from dataset import CLASSES
from logger import ModelSaver

# RGB color for each class.
COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
            [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0],
            [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128],
            [64, 128, 128], [192, 128, 128], [0, 64, 0], [128, 64, 0],
            [0, 192, 0], [128, 192, 0], [0, 64, 128]]


class Predict:
    def __init__(self, model, cuda=False):
        self.model = model
        self.model.eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() and cuda else 'cpu')
        self.model = self.model.to(self.device)
        self.color_map = torch.Tensor(COLORMAP)

    def predict_one_img(self, img_path, resize=None):
        img, img_tensor, (w, h) = self.from_PIL_path_to_Tensor(img_path, resize)
        img_tensor = img_tensor.unsqueeze(dim=0)
        img_tensor = img_tensor.to(self.device)
        print(f'img_tensor:{img_tensor.size()}')

        self.model.eval()
        self.model = self.model.to(self.device)

        pred = self.model(img_tensor)  # shape[1, n_class, h, w]
        pred = torch.sigmoid(pred)

        print(f'pred.size:{pred.size()}')
        print(f'pred:{pred}')
        pred = pred.argmax(dim=1, keepdim=True)  # shape[1, 1, h, w]
        pred = pred.squeeze()  # shape[ h, w]
        print(f'pred size:{pred.size()}\npred:{pred}')
        pred_img = self.from_pred_to_color_img(pred)
        pred_img = pred_img.resize((w, h))
        pred_img.show()
        merge_img = self.merge_two_img(img, pred_img)
        merge_img.show()

    @staticmethod
    def from_PIL_path_to_Tensor(img_path, resize):
        img = Image.open(img_path)
        w, h = img.size
        if resize is not None:
            assert isinstance(resize, numbers.Number) or (isinstance(resize, tuple) and len(resize) == 2)

            if isinstance(resize, numbers.Number):
                resize = (resize, resize)
        else:
            resize = (480, 480)

        img_resized = img.resize(resize)
        img_tensor = transforms.ToTensor()(img_resized)
        return img, img_tensor, (w, h)

    def from_pred_to_color_img(self, pred):
        h, w = pred.size()
        temp = torch.zeros((3, h, w))
        for i in range(h):
            for j in range(w):
                temp[:, i, j] = self.color_map[pred[i, j]]
        return transforms.ToPILImage()(temp / 255.0)

    @staticmethod
    def merge_two_img(img1, img2):
        return Image.blend(img1, img2, 0.5)


if __name__ == '__main__':

    args = config()
    model = get_model(name=args.model_name, n_class=args.n_class)
    model_saver = ModelSaver(save_path=args.save, name_list=[args.optimizer, args.model_name])
    # model_saver.load(args.model_name, model)
    model_saver.load(name=args.model_name, model=model)
    predictor = Predict(model)

    img_dir = '../test_pic'
    img_list = from_dir_get_imgs_list(test_img_dir=img_dir)
    for img_path in img_list:
        predictor.predict_one_img(img_path)
    pass
