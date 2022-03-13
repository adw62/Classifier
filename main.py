import torch
import pandas as pd
import numpy as np
import copy
from sklearn import preprocessing
from torch.utils.data import Dataset, DataLoader, random_split

class MyDataset(Dataset):
    def __init__(self, df, df_a, labels):
        blank_ans = [0.0]*len(labels)
        answer_key = {}
        for i, label in enumerate(labels):
            ans = copy.deepcopy(blank_ans)
            ans[i] = 1.0
            answer_key[label] = ans
        data = [[float(x) for x in df.iloc[y]] for y in range(df.shape[0])]
        ans = torch.tensor([[float(x) for x in answer_key[df_a.iloc[y]]] for y in range(df.shape[0])], device=cuda)
        min_max_scaler = preprocessing.MinMaxScaler()
        data = torch.tensor(min_max_scaler.fit_transform(data), device=cuda).float()
        self.data = [[i, j] for i, j in zip(data, ans)]

    def __getitem__(self, i):
        return self.data[i]

    def __len__(self):
        return len(self.data)

def make_model(in_size, out_size):
    print('Building model {}x{}'.format(in_size, out_size))
    model = torch.nn.Sequential(
        torch.nn.Linear(in_size, in_size),
        torch.nn.Linear(in_size, out_size),
    ).cuda()
    return model

def train(model, train_loader):
    print('Training...')
    print('Iter: Loss')
    optimizer = torch.optim.Adam(model.parameters(), 0.1)
    loss_fn = torch.nn.CrossEntropyLoss()
    prev_loss = 100000
    t = 0
    training = True
    while training:
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            y_pred = model(inputs)
            loss = loss_fn(y_pred, labels)
            gain = abs(prev_loss-loss.item())
            if gain < 0.000001:
                training = False
            if t % 100 == 99:
                print(t, loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        t+=1
        prev_loss = loss.item()
        if t == 5000:
            training == False

def test(model, valid_loader):
    loss_fn = torch.nn.CrossEntropyLoss()

    for i, data in enumerate(valid_loader, 0):
        inputs, labels = data
        y_pred = model(inputs)
        loss = loss_fn(y_pred, labels)
        print('Test loss', loss.item())
        correct = 0
        wrong = 0
        for x, y in zip(y_pred, labels):
            pred = [int(i) for i in x]
            true = [int(j) for j in y]
            pa = pred.index(max(pred))
            ta = true.index(max(true))
            if pa != ta:
                wrong += 1
            else:
                correct += 1
        print('Accuracy = ', (correct/(correct+wrong))*100, '%')

def make_data_loaders(df, df_a, labels):
    train_ds = MyDataset(df, df_a, labels)
    # calculate size of train and validation sets
    train_size = int(0.8 * len(train_ds))
    valid_size = len(train_ds) - train_size
    partial_train_ds, valid_ds = random_split(train_ds, [train_size, valid_size])
    train_loader = DataLoader(partial_train_ds, batch_size=train_size)
    valid_loader = DataLoader(valid_ds, batch_size=valid_size)

    return train_loader, valid_loader

def extract_ans(df, class_key):
    df_a = df.pop(class_key)
    return df, df_a

def main():

    #file_name = 'Pistachio_28_Features_Dataset.xlsx'
    #labels = ['Kirmizi_Pistachio', 'Siirt_Pistachio']
    #sheet_name = 'Pistachio_28_Features_Dataset'
    #class_key = 'Class'
    #file_type = 'excel'

    #file_name = './Dry_Bean_Dataset.xlsx'
    #labels = ['SEKER', 'BARBUNYA', 'BOMBAY', 'CALI', 'HOROZ', 'SIRA', 'DERMASON']
    #sheet_name = 'Dry_Beans_Dataset'
    #class_key = 'Class'
    #file_type = 'excel'

    #file_name = 'Date_Fruit_Datasets.xlsx'
    #labels = ['BERHI', 'DEGLET', 'DOKOL', 'IRAQI', 'ROTANA', 'SAFAVI', 'SOGAY']
    #sheet_name = 'Date_Fruit_Datasets'
    #class_key = 'Class'
    #file_type = 'excel'

    file_name = 'star_classification.csv'
    labels = ['STAR', 'GALAXY', 'QSO']
    class_key = 'class'
    file_type = 'csv'

    if file_type == 'csv':
        #CSV
        df = pd.read_csv(file_name)
    elif file_type == 'excel':
        #EXCEL
        xl_file = pd.ExcelFile(file_name)
        dfs = {sheet_name: xl_file.parse(sheet_name)
               for sheet_name in xl_file.sheet_names}
        df = dfs[sheet_name]
    else:
        raise ValueError('Unknown file type')

    df, df_a = extract_ans(df, class_key)
    train_loader, valid_loader = make_data_loaders(df, df_a, labels)

    model = make_model(df.shape[1], len(labels))
    train(model, train_loader)
    test(model, valid_loader)

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Device:', device)
    cuda = torch.device(device)
    main()

