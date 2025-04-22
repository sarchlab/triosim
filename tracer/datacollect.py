import os
import time
import torch
import torchvision
import sys
import numpy as np
import torchvision.models as models
from torchvision import transforms
from tqdm import tqdm

import torch.optim as optim
import torch.nn as nn

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

from torch.profiler import profile, record_function, ProfilerActivity
from torch.profiler import ExecutionTraceObserver

bsinput=int(sys.argv[1])
listmodel=[]
num_iters = 43
vgg11 = models.vgg11(weights=None)
vgg13 = models.vgg13(weights=None)
vgg16 = models.vgg16(weights=None)
vgg19 = models.vgg19(weights=None)

resnet18 = models.resnet18(weights=None)
resnet34 = models.resnet34(weights=None)
resnet50 = models.resnet50(weights=None)
resnet101 = models.resnet101(weights=None)
resnet152 = models.resnet152(weights=None)

densenet161 = models.densenet161(weights=None)
densenet169 = models.densenet169(weights=None)
densenet121 = models.densenet121(weights=None)
densenet201 = models.densenet201(weights=None)

bslist=[bsinput]

data_transforms = {
    'predict': transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])}

device = torch.device("cuda:0")
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))
print(torch.cuda.current_device())
   
for bs in bslist:
    
    BATCH_SIZE = bs
    EPOCHS = 1
    WORKERS = 8
    IMG_DIMS = (336, 336)
    CLASSES = 10

    dataset = {'predict' : torchvision.datasets.ImageFolder("./ILSVRC2012_img_val", data_transforms['predict'])}
    dataset_subset=dataset['predict']
    dataset_subset = torch.utils.data.Subset(dataset['predict'],range(BATCH_SIZE))
    data_loader = {'predict': torch.utils.data.DataLoader(dataset_subset,
                                            batch_size=BATCH_SIZE,
                                            shuffle=False,
                                            num_workers=WORKERS)}

    listmodel=[vgg11,vgg13,vgg16,vgg19,resnet18,resnet34,resnet50,resnet101,resnet152,
              densenet161,densenet169,densenet121,densenet201]
    namelist = ['vgg11','vgg13','vgg16','vgg19','resnet18','resnet34','resnet50','resnet101','resnet152',
              'densenet161','densenet169','densenet121','densenet201']
    j = 0
    for model in listmodel:
        name = namelist[j]
        print(f"Batch Size: {BATCH_SIZE}, Epochs: {EPOCHS}, Workers: {WORKERS}, Network: {name}")
        j=j+1
        try: 
            model = model.to(device)
            loss_fn = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.01)
            for epoch in range(EPOCHS): # 1 epoch
                    model.train()
                    for batch in tqdm(data_loader['predict'], total=len(data_loader['predict'])):
                        features, labels = batch[0].to(device), batch[1].to(device)
                        print( "Batchsize:", features.size())
                        total_time1 = 0
                        total_time2 = 0
                        for i in range(num_iters):
                            if 40<i<=num_iters-1:
                                eg = ExecutionTraceObserver()
                                eg.register_callback("./graph_"+name+"-iter"+str(i)+".json")
                                eg.start()
                                with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA],record_shapes=True,with_stack=True,profile_memory=True) as prof: #,on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/resnet18')
                                    torch.cuda.synchronize()
                                    starter = torch.cuda.Event(enable_timing=True)
                                    ender = torch.cuda.Event(enable_timing=True)
                                    starter.record()
                    
                                    optimizer.zero_grad()

                                    preds = model(features)
                                    loss = loss_fn(preds, labels)

                                    loss.backward()
                                    optimizer.step()
                                    
                                    ender.record()
                                    torch.cuda.synchronize()
                                    curr_time = starter.elapsed_time(ender)
                                    total_time1 += curr_time                           
                                prof.export_chrome_trace("./profiler_"+name+"-iter"+str(i)+".json")
                                eg.stop()
                                eg.unregister_callback()
                                print(name, "gpu time", curr_time)
                            elif 30<i<=40:
                                torch.cuda.synchronize()
                                starter = torch.cuda.Event(enable_timing=True)
                                ender = torch.cuda.Event(enable_timing=True)
                                starter.record()
                                
                                optimizer.zero_grad()
                                preds = model(features)
                                loss = loss_fn(preds, labels)
                                loss.backward()
                                optimizer.step()
                                
                                ender.record()
                                torch.cuda.synchronize()
                                curr_time = starter.elapsed_time(ender)
                                total_time2 += curr_time
                                print(name, "gpu time 2", curr_time)
                            else:
                                optimizer.zero_grad()
                                preds = model(features)
                                loss = loss_fn(preds, labels)
                                loss.backward()
                                optimizer.step()
                        print("avg profiler Total time1:", total_time1/2)
                        print("avg Total time2:", total_time2/10)
                        
        except Exception as e:
            print(e)
            pass