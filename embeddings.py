import os, json
import pandas as pd
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm

import boto3
import s3fs
from io import BytesIO
import subprocess

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from torch.optim.lr_scheduler import OneCycleLR
from pytorch_lightning.callbacks import EarlyStopping
from torchmetrics.functional import accuracy
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models._utils import IntermediateLayerGetter
from pytorch_lightning.callbacks import ModelCheckpoint

seed_everything(333)


class Clustering_Module(nn.Module):

    def __init__(self, input_dim, num_clusters, use_bias=False):
        super(Clustering_Module, self).__init__()
        self.input = input_dim
        self.num_clusters = num_clusters
        self.Id = torch.eye( num_clusters )

        self.gamma = nn.Sequential(
            nn.Linear(input_dim, num_clusters, bias=use_bias),
            nn.Softmax()
        )
        self.mu = nn.Sequential(
            nn.Linear(num_clusters, input_dim, bias=use_bias),
        )

    def _mu(self):
        return self.mu( self.Id )

    def predict(self, x):
        return self.gamma(x).argmax(-1)

    def predict_proba(self, x):
        return self.gamma(x)

    def forward(self, x):
        g  = self.gamma(x)
        tx = self.mu(g)
        u  = self.mu[0].weight.T
        return (x, g, tx, u)


class Clustering_Module_Loss(nn.Module):
    def __init__(self, num_clusters, alpha=1, lbd=1, orth=False, normalize=True, device='cpu'): #device='cuda'
        super(Clustering_Module_Loss, self).__init__()

        if hasattr( alpha, '__iter__'):
            self.alpha = torch.tensor(alpha, device=device)
            # Use this if alpha are not all the same
            # self.alpha = torch.sort(self.alpha).values
        else:
            self.alpha = torch.ones( num_clusters, device=device ) * alpha

        self.lbd   = lbd
        self.orth = orth
        self.normalize = normalize
        self.Id = torch.eye(num_clusters, device=device)
        self.mask = 1-torch.eye(num_clusters, device=device)

    def forward(self, inputs, targets=None, split=False):
        x,g,tx,u = inputs
        n,d = x.shape
        k = g.shape[1]

        nd = (n*d) if self.normalize else 1.

        loss_E1 = torch.sum( torch.square( x - tx ) ) / nd

        if self.orth:
            loss_E2 = torch.sum( g*(1-g) ) / nd

            uu = torch.matmul( u, u.T )
            loss_E3 = torch.sum( torch.square( uu - self.Id.to(uu.device) ) ) * self.lbd
        else:
            loss_E2 = torch.sum( torch.sum( g*(1-g),0 ) * torch.sum( torch.square(u), 1) ) / nd

            gg = torch.matmul( g.T, g)
            uu = torch.matmul( u, u.T )
            gu = gg * uu
            gu = gu * self.mask
            loss_E3 = - torch.sum( gu ) / nd

        lmg = torch.log( torch.mean(g,0) +1e-10 )
        # Use this if alpha are not all the same
        # lmg = torch.sort(lmg).values
        loss_E4 = lmg

        if split:
            nd  = 1. if self.normalize else n*d
            return torch.stack( (loss_E1/nd, loss_E2/nd, loss_E3/(1 if self.orth else nd) , torch.sum(loss_E4* (1-self.alpha)) ) )
        else:
            return loss_E1 + loss_E2 + loss_E3 + torch.sum( loss_E4 * (1-self.alpha) )


def prepare_backbone(backbone='mobilenet', use_pretrained=True):

    if backbone == "mobilenet":
        model = torchvision.models.mobilenet_v2(pretrained=use_pretrained)
    elif backbone == "resnet18":
        model = torchvision.models.resnet18(pretrained=use_pretrained)
    elif backbone == "vgg16":
        model = torchvision.models.vgg16(pretrained=use_pretrained)
    elif backbone == "effnet0":
        model = torchvision.models.efficientnet_b0(pretrained=use_pretrained)

    model = create_feature_extractor(model, return_nodes=["flatten"] )

    return model

def get_feature_dims(backbone):
    inp = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = backbone(inp)
    in_channels_list = [o.shape[1] for o in out.values()]
    return in_channels_list[0]

class CustomClassifier(LightningModule):
    def __init__(self, lr=0.05, num_clusters=5, num_classes = 5):
        super().__init__()

        self.save_hyperparameters()
        self.lr = lr
        self.model = prepare_backbone("mobilenet")
        latent_dim=10
        num_classes=num_classes
        feature_dim = get_feature_dims(self.model)
        self.BETA=1.5
        self.ALPHA=1.1
        self.LBD=0.1
        self.cm_loss=Clustering_Module_Loss(
                        num_clusters=num_clusters,
                        alpha=self.ALPHA,
                        lbd=self.LBD,
                        orth=True,
                        normalize=True)
        self.latent_layer = nn.Linear(feature_dim, latent_dim)
        self.classifier = nn.Sequential(nn.Linear(latent_dim, int(feature_dim/2)),
                                nn.ReLU(),
                                nn.Dropout(0.5),
                                nn.Linear(int(feature_dim/2), 64),
                                nn.ReLU(),
                                nn.Dropout(0.3),
                                nn.Linear(64, num_classes))
        self.cm=Clustering_Module(latent_dim, num_clusters, False)

    def forward(self, x):
        x=self.model(x)
        x=x["flatten"]
        x_latent = self.latent_layer(x)
        # x = self.classifier(x)
        return F.log_softmax(self.classifier(x_latent), dim=1),self.cm(x_latent),x_latent

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits,cm_out,_ = self(x)
        loss_clf = F.nll_loss(logits, y)
        loss_cm=self.cm_loss(cm_out)
        loss=loss_clf*self.BETA+loss_cm
        # print("classifier loss: {}, cm loss: {}".format(loss_clf,loss_cm))
        # self.log("train_loss", loss)
        return loss

    def evaluate(self, batch, stage=None):
        x, y = batch
        # logits = self(x)
        # loss = F.nll_loss(logits, y)
        logits,cm_out,_ = self(x)
        loss_clf = F.nll_loss(logits, y)
        loss_cm=self.cm_loss(cm_out)
        loss=loss_clf*self.BETA+loss_cm
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y, task="multiclass", num_classes=5)

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_acc", acc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.lr,
            momentum=0.9,
            weight_decay=5e-4,
        )
        steps_per_epoch = 45000 // BATCH_SIZE
        scheduler_dict = {
            "scheduler": OneCycleLR(
                optimizer,
                0.1,
                epochs=self.trainer.max_epochs,
                steps_per_epoch=steps_per_epoch,
            ),
            "interval": "step",
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = CustomClassifier.load_from_checkpoint("epoch=89-val_loss=0.90.ckpt", map_location=torch.device('cpu')).to(device)

class Hook():
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output
    def close(self):
        self.hook.remove()

tforms = transforms.Compose([transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                # transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                              ])

hook = Hook(model.latent_layer)

res = boto3.resource('s3')
my_bucket = res.Bucket('gcp-vm-data')
image_list = []
for obj in my_bucket.objects.filter(Prefix = 'light_metrics/sample'):
    if '.jpg' in obj.key:
        image_list.append(obj.key)

s3 = boto3.client('s3')

def get_embedding(img, Bucket='gcp-vm-data'):
    # img = Image.open(BytesIO(s3.get_object(Bucket=Bucket, Key=key)['Body'].read()))
    img1 = tforms(img)
    img1 = img1.unsqueeze(0)
    img1 = torch.autograd.Variable(img1, requires_grad=False).to(device)
    output= model(img1)
    np_out = hook.output.cpu().detach().numpy()[0]
    return np_out.tolist() #, output[1][1][0].tolist()

def dhash(image, hashSize=8):
        resized = cv2.resize(image, (hashSize + 1, hashSize))
        diff = resized[:, 1:] > resized[:, :-1]
        return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])

def convert_to_eight_digits(digit):
    digit_str = str(digit)
    while len(digit_str) < 8:
        digit_str = '0' + digit_str
    return digit_str

def cropped_image(image, bbox_):
  dw,dh,_=image.shape
  x,y,w,h=bbox_
  x=x+w/2
  y=y+h/2
  x=float(x)-float(w)/2
  y=float(y)-float(h)/2
  nx = max(0,int(float(x)*dh))
  ny = max(0,int(float(y)*dw))
  nw = max(0,int(float(w)*dh))
  nh = max(0,int(float(h)*dw))
  return image[ny:ny+nh, nx:nx+nw]

with open('json_files/hb.json', 'r') as f:
    hb = json.load(f)
with open('json_files/ss.json', 'r') as f:
    ss = json.load(f)
with open('json_files/tg.json', 'r') as f:
    tg = json.load(f)
with open('whole_dataset_filtered.json', 'r') as f:
    whole_dataset = json.load(f)

if not os.path.exists('videos_crop_dataset/'):
    os.mkdir('videos_crop_dataset/')
    os.mkdir('videos_crop_dataset/HarshBraking_f/')
    os.mkdir('videos_crop_dataset/TailGating_f/')
    os.mkdir('videos_crop_dataset/StopSign_f/')

if not os.path.exists('sample_videos_results'):
    os.mkdir('sample_videos_results')

base_key = 'a100/datasets/lightmetrics_data/videos_frame_dataset'
Bucket = 'gcp-vm-data'

for i in tqdm(range(len(whole_dataset))):
    vid_name = whole_dataset[i]['inputs'][0]
    if 'Harsh-Braking' in vid_name:
        for l in range(len(hb)):
            if hb[l]['inputs'][0] == vid_name:
                fr_res = hb[l]['outputs']
                cl_ext = 'HarshBraking_f/'
                break
    elif 'tail_gating' in vid_name:
        for l in range(len(tg)):
            if tg[l]['inputs'][0] == vid_name:
                fr_res = tg[l]['outputs']
                cl_ext = 'TailGating_f/'
                break
    else:
        for l in range(len(ss)):
            if ss[l]['inputs'][0] == vid_name:
                fr_res = ss[l]['outputs']
                cl_ext = 'StopSign_f/'
                break
    
    full_path_name='videos_crop_dataset/{}'.format(cl_ext)+"{}/".format(vid_name.split('.')[0])
    if not os.path.exists(full_path_name):   
        os.mkdir(full_path_name)


    for j in range(len(whole_dataset[i]['outputs'])):
        frame_wd = int(whole_dataset[i]['outputs'][j]['imageName'].split('/')[-1].split('.')[0].split('_')[-1])
        s3_img = vid_name.split('.')[0]+'_000'+str(frame_wd)+'.jpg'
        key = os.path.join(base_key, cl_ext, vid_name.split('.')[0], s3_img)
        image = np.array(Image.open(BytesIO(s3.get_object(Bucket=Bucket, Key=key)['Body'].read())))
        whole_dataset[i]['outputs'][j]['duplicate_hash'] = dhash(image)
        img_name = whole_dataset[i]['outputs'][j]['imageName']
        for k in range(len(fr_res)):
            if fr_res[k]['frame_id'] == frame_wd:
                for q in range(len(fr_res[k]['detections'])):
                    bbox = fr_res[k]['detections'][q]['bbox']
                    crop_img = cropped_image(image, bbox)
                    clas = fr_res[k]['detections'][q]['class']
                    if not os.path.exists(os.path.join('sample_videos_results', clas)):
                        os.mkdir(os.path.join('sample_videos_results', clas))
                    whole_dataset[i]['outputs'][j]['detections'].append({'class': clas,
                    'bbox': bbox, 'confidence': fr_res[k]['detections'][q]['confidence'],
        'bbox_vector': get_embedding(Image.fromarray(crop_img)),'duplicate_hash': dhash(crop_img), 'bbox_cluster': ' ', 'duplicate': 'false'})
                    crop_name =str(img_name.split('.')[0] + '_' + clas + '_' + convert_to_eight_digits(q+1) + '.jpg')
                    cv2.imwrite(os.path.join(full_path_name,crop_name), crop_img)
                    print(full_path_name, crop_name)           
                break
                 
    

with open('whole_dataset_filtered_final.json', 'w') as f:
    json.dump(whole_dataset, f, indent =4)



#videos_crop_dataset/HarshBraking_f/eventVideo_sensor_Harsh-Braking_trip_master_2022_02_28_23_00_58_606_2F1D5D62253768B3F025509CBB95444255B8716B_ahYlK_3_1646116055421_primary_2022_03_01_06_27_35_464/eventVideo_stop_sign_trip_master_2022_03_07_15_40_32_719_6A634925A65969E094C02EB06F5F40969AAFCD8E_uG3ou_4_2022_03_07_16_28_24_248_primary_0000_Car_9.jpg

