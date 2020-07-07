import torch.nn as nn
import torch
import torchvision.transforms as transforms
from PIL import Image
import torchvision
from matplotlib.pyplot import imshow
import Dataset as D_set
import VGGnet as VGG_n
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'




if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '1' 
    torch.manual_seed(777)
    if device == 'cuda':
        torch.cuda.manual_seed_all(777)

    path = './Magic_card/origin_data'
    # load data
    train_data = D_set.load_data(path)
    data_loader = D_set.custom_loader(train_data,8)
    #data_loader = torch.utils.data.DataLoader(dataset = train_data,batch_size = 8, shuffle =True, num_workers =2)
    #setting loss_func, net, optim
    conv =  VGG_n.make_layers(VGG_n.cfg['E'])
    net = VGG_n.VGG(conv,num_classes = 2,init_weights=True)
    model = net.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.00005)
    loss_func = nn.CrossEntropyLoss().to(device)
    total_batch = len(data_loader)
    


    # epochs = 10
    # model.train
    # for epoch in range(epochs):
    #     avg_cost = 0
    #     for X, Y in data_loader:
    #         X = X.to(device)
    #         Y = Y.to(device)

    #         optimizer.zero_grad()
    #         out = net(X)
    #         loss = loss_func(out,Y)
    #         loss.backward()
    #         optimizer.step()

    #         avg_cost += loss / total_batch
        
    #     print('Epoch:{} cost = {}'.format(epoch+1,avg_cost))
    # print('Learning Finished!')

    # if not os.path.exists('./model'):
    #     os.mkdir('./model')

    # torch.save(model.state_dict(),"./model/vgg11_model.pth")
    
    test_net = net.to(device)
    test_net.load_state_dict(torch.load('./model/vgg11_model.pth'))

    test_data = D_set.load_data(path)
 
    #test_data = D_set.load_data(path)
    #test_loader = D_set.custom_loader(test_data,len(test_data))
    test_loader = torch.utils.data.DataLoader(dataset = test_data, batch_size = len(test_data))

    with torch.no_grad():
        model.eval()
        for imgs, label in test_loader:
            imgs = imgs.to(device)

            label = label.to(device)

            prediction = model(imgs)

            correct_prediction = (torch.argmax(prediction, 1) == label)
            accuracy = correct_prediction.float().mean()
            print('Accuracy:', accuracy.item())


