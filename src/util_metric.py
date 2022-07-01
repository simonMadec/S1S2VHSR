from turtle import color
import torch
import numpy as np
import matplotlib.pyplot as plt 

def validateALL(net, loader):
    # compute all metrics
    y_pred = []
    y_true = []

    net = net.cuda()
    net.eval()

    with torch.no_grad():
        acc = []
        f1 = []
        ka = []
        for i, sample in enumerate(loader):
            # breakpoint()
            # col = {0: 'red', 1: 'blue' , 2: 'black', 3: 'yellow', 4: 'green', 5: 'orange', 6: 'grey', 7: 'brown'}
            # tlabel= sample['Target'].numpy()[0].astype('uint8')
            # plt.plot(range(62), sample['S1'][0, :, 4,4], label=tlabel, color=col[tlabel])
            for x in sample:
                sample[x] = sample[x].cuda()           
            
            outputs_is = net(sample)

            # generate targets
            if type(outputs_is) is dict: # fusion
                 outputs_is = outputs_is["fusion"]

            correct_pred = (outputs_is.argmax(-1) == sample["Target"]).float()
            val = correct_pred.sum() / len(correct_pred)

            #f1.append(f1_score(outputs_is.argmax(-1).cpu().numpy(), sample["Target"].cpu().numpy(),average='micro'))
            #ka.append(kappa(outputs_is.argmax(-1).cpu().numpy(), sample["Target"].cpu().numpy()))
            acc.append(val.cpu().numpy())

            y_pred = np.concatenate((y_pred, outputs_is.argmax(-1).cpu().numpy() ), axis=0)
            y_true = np.concatenate((y_true, sample["Target"].cpu().numpy() ), axis=0)
             #faire une accuracy  

    return np.mean(np.asarray(acc)), y_true, y_pred


def validate(net, loader,perclasse_acc = False,list_class=['Sugarcane', 'Pasture and fodder', 'Market gardening','Grenhouse and shaded crops', \
            'Orchards','Wooded areas','Moor and Savannah','Rocks and natural bare soil','Relief shadow','Water','Urbanized areas']):
    #compute just accuracy

    net = net.cuda()
    net.eval()
    with torch.no_grad():
        acc = []
        acc_classe = {k:[] for k in list_class}

        for i, sample in enumerate(loader):
            for x in sample:
                sample[x] = sample[x].cuda()
                
            outputs_is = net(sample)
            # generate targets
            if type(outputs_is) is dict: # fusion
                 outputs_is = outputs_is["fusion"]
            correct_pred = (outputs_is.argmax(-1) == sample["Target"]).float()

            if perclasse_acc == True:
                #debug not working now
                Target =Target.float().cpu().numpy()
                Pred = outputs_is.argmax(-1).float().cpu().numpy()

                for i,classe in enumerate(np.unique(Target)):
                    acc_classe[list_class[i]].append(np.sum((Pred== Target) & (Target == classe)) / np.sum(Target == classe))

            val = correct_pred.sum() / len(correct_pred)
            acc.append(val.cpu().numpy())
             #faire une accuracy

    if perclasse_acc == True:
        for key, value in acc_classe.items():
            acc_classe[key]= int(np.mean(value)*100)
        
        return np.mean(np.asarray(acc)), acc_classe
    else: 
         return np.mean(np.asarray(acc)), None