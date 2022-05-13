import argparse
import os
import matplotlib.pyplot as plt
import torch
import tqdm
from torch.utils.data import Dataset, DataLoader
from model import RobustModel


class ImageDataset(Dataset):
    """ Image shape: 28x28x3 """

    def __init__(self, root_dir, fmt=':06d', extension='.png'):
        self.root_dir = root_dir
        self.fmtstr = '{' + fmt + '}' + extension

    def __len__(self):
        return len(os.listdir(self.root_dir))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.fmtstr.format(idx)
        img_path = os.path.join(self.root_dir, img_name)
        img = plt.imread(img_path)
        data = torch.from_numpy(img)
        return data

def inference(data_loader, model):
    """ model inference """

    model.eval()
    preds = []

    with torch.no_grad():
        for X in data_loader:
            y_hat = model(X)
            y_hat.argmax()

            _, predicted = torch.max(y_hat, 1)
            preds.extend(map(lambda t: t.item(), predicted))

    return preds


if __name__ == '__main__': #import나 위에서선언한것들이 자동으로 실행되지 않고 함수가 실행이 되었을때만 시작되는거
    parser = argparse.ArgumentParser(description='2022 DL Term Project #1')
    parser.add_argument('--load-model', default='model.pt', help="Model's state_dict")
    parser.add_argument('--dataset_test', default='./test/', help='image dataset directory') #이름을 dataset이라고 하고 생성되는 거가 default, help가 설명
    parser.add_argument('--dataset_train', default='./train/', help='image dataset directory_train')
    parser.add_argument('--batch-size', default=16, help='test loader batch size')

    args = parser.parse_args()

    # instantiate model
    model = RobustModel()
    model.load_state_dict(torch.load(args.load_model))

    # load dataset in test image folder
    train_data = ImageDataset(args.dataset_train)
    test_data = ImageDataset(args.dataset_test) #확인을 해볼 데이터를 args.dataset에서 가져온다.
    
    train_set, val_set = torch.utils.data.random_split(train_data, [35, 10])
    dev_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size) #testdata의 사진에 batchsize는 위에서 정한 16 
    #요기까지의 코드가 test, train, valid 데이터를 나눴으니 이제 위에서 이거를 이용해서 코드 작성
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)
    training_epochs = 5  
    for epoch in range(training_epochs):
        model.train()
        cost = 0
        n_batches = 0
        for X, Y in tqdm.tqdm(train_loader):
            optimizer.zero_grad()
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()

            cost += loss.item()
            n_batches += 1

        cost /= n_batches
        inference(dev_loader, model)
    
    # write model inference
    preds = inference(test_loader, model)

    with open('result.txt', 'w') as f:
        f.writelines('\n'.join(map(str, preds)))
