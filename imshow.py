import matplotlib.pyplot as plt

def imshow(img):
    img=img/2 + 0.5
    img=img.detach()
    plt.imshow(img.permute(1,2,0).to('cpu'))
    plt.show()
        
def batchshow(batch):
    for i in range(len(batch)):
        features = batch[i]
        print("Feature batch shape:", {features.size()})
        print("Image shape",{features[0].size()})
        imshow(features)
