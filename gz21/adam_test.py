import torch
from gz21.models.models1 import FullyCNN
from gz21.models.transforms import SoftPlusTransform
def main():
    torch.manual_seed(1)
    cnn = FullyCNN()
    final_transformation = SoftPlusTransform()
    final_transformation.indices = slice(2,4)
    cnn.final_transformation = final_transformation
    
    optimizer = torch.optim.Adam(cnn.parameters(),lr = 5e-4)
    for ib in range(10): 
        x = torch.zeros((1,2,22,22))
        mean,prec = torch.split(cnn(x),2,dim = 1)
        loss = (torch.log(prec) - mean**2*prec).mean()
        
        sttdict = dict(
            input = x,
            output = torch.cat([mean,prec],dim = 1).detach(),
            loss = loss.detach(),
            **cnn.state_dict(),
        )
        torch.save(sttdict,f'sttdict_{ib}.pth')
        loss.backward()
        optimizer.step()
        print(f'loss = {loss.item()}')

if __name__ == '__main__':
    main()