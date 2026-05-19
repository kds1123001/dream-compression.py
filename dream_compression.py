import os
import cv2
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
from torch.utils.data import Dataset, DataLoader
import math
#i love my colors
import colorama



VIDEO_PATH="video.mp4.webm"

FRAME_SIZE=64

SEQUENCE_LEN=5

#you can add your own latent sizes for experimentation and testing

LATENT_SIZE=32

LATENT_SIZE=16

LATENT_SIZE=8

LATENT_SIZE=4

LATENT_SIZE=2

BATCH_SIZE=8

EPOCHS=25

device="cuda" if torch.cuda.is_available() else "cpu"

print("="*40)

print(
    f"Running on: {device}"
)



class VideoDataset(Dataset):

    def __init__(self,path):

        print(
            f"Looking for: {path}"
        )

        if not os.path.exists(path):

            raise FileNotFoundError(
                f"Video not found: {path}"
            )

        cap=cv2.VideoCapture(path)

        if not cap.isOpened():

            raise RuntimeError(
                "Could not open video"
            )

        frames=[]

        while True:

            ret,frame=cap.read()

            if not ret:
                break

            frame=cv2.resize(
                frame,
                (
                    FRAME_SIZE,
                    FRAME_SIZE
                )
            )

            frame=cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            frame=torch.tensor(
                frame,
                dtype=torch.float32
            )/255.0

            frame=frame.permute(
                2,
                0,
                1
            )

            frames.append(
                frame
            )

        cap.release()

        print(
            f"Loaded {len(frames)} frames"
        )

        if len(frames)<=SEQUENCE_LEN:

            raise ValueError(
                f"Need > {SEQUENCE_LEN} frames"
            )

        self.frames=frames


    def __len__(self):

        return (
            len(self.frames)
            -SEQUENCE_LEN
        )


    def __getitem__(self,idx):

        x=torch.stack(

            self.frames[
                idx:idx+SEQUENCE_LEN
            ]

        )

        y=self.frames[
            idx+SEQUENCE_LEN
        ]

        return x,y


dataset=VideoDataset(
    VIDEO_PATH
)

loader=DataLoader(

    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True

)


class KalmanFilter2D:

    def __init__(self):

        self.x=np.array([
            [0],
            [0],
            [0],
            [0]
        ],dtype=float)

        self.F=np.array([

            [1,0,1,0],
            [0,1,0,1],
            [0,0,1,0],
            [0,0,0,1]

        ])

        self.H=np.array([

            [1,0,0,0],
            [0,1,0,0]

        ])

        self.P=np.eye(4)*1000
        self.R=np.eye(2)*5
        self.Q=np.eye(4)*0.1


    def predict(self):

        self.x=self.F@self.x

        self.P=(
            self.F
            @self.P
            @self.F.T
            +self.Q
        )


    def update(self,z):

        z=np.array(
            z
        ).reshape(2,1)

        y=z-self.H@self.x

        S=(
            self.H
            @self.P
            @self.H.T
            +self.R
        )

        K=(
            self.P
            @self.H.T
            @np.linalg.inv(S)
        )

        self.x=self.x+K@y

        self.P=(
            np.eye(4)
            -K@self.H
        )@self.P




class DreamNet(nn.Module):

    def __init__(self):

        super().__init__()

        self.encoder=nn.Sequential(

            nn.Conv2d(
                3,
                32,
                4,
                2,
                1
            ),

            nn.ReLU(),

            nn.Conv2d(
                32,
                64,
                4,
                2,
                1
            ),

            nn.ReLU(),

            nn.Flatten(),

            nn.Linear(
                64*16*16,
                LATENT_SIZE
            )

        )

        self.lstm=nn.LSTM(

            LATENT_SIZE,
            LATENT_SIZE,
            batch_first=True

        )

        self.fc=nn.Linear(

            LATENT_SIZE,
            64*16*16

        )

        self.decoder=nn.Sequential(

            nn.ConvTranspose2d(
                64,
                32,
                4,
                2,
                1
            ),

            nn.ReLU(),

            nn.ConvTranspose2d(
                32,
                3,
                4,
                2,
                1
            ),

            nn.Sigmoid()

        )


    def forward(self,x):

        B,T,C,H,W=x.shape

        latent=[]

        for t in range(T):

            z=self.encoder(
                x[:,t]
            )

            latent.append(z)

        latent=torch.stack(
            latent,
            dim=1
        )

        output,_=self.lstm(
            latent
        )

        predicted=output[:,-1]

        x=self.fc(
            predicted
        )

        x=x.view(
            -1,
            64,
            16,
            16
        )

        return self.decoder(
            x
        )



model=DreamNet().to(
    device
)

kf=KalmanFilter2D()

optimizer=torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

fitness_history=[]
loss_history=[]



print("\nTraining...\n")

for epoch in range(EPOCHS):

    total_loss=0

    for sequence,target in loader:

        sequence=sequence.to(
            device
        )

        target=target.to(
            device
        )

        prediction=model(
            sequence
        )

        motion=torch.abs(
            target-
            sequence[:,-1]
        )

        weights=1+motion*10

        loss=(

            ((prediction-target)**2)
            *weights

        ).mean()

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss+=loss.item()

    avg_loss=(
        total_loss/
        len(loader)
    )

    fitness=(
        1/
        (avg_loss+1e-8)
    )

    loss_history.append(
        avg_loss
    )

    fitness_history.append(
        fitness
    )

    print(
        f"Epoch {epoch+1}/{EPOCHS}"
        f" | Loss:{avg_loss:.5f}"
        f" | Fitness:{fitness:.2f}"
    )




plt.figure()

plt.plot(
    loss_history
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Loss"
)

plt.title(
    "Training Loss"
)

plt.show()


plt.figure()

plt.plot(
    fitness_history
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Fitness"
)

plt.title(
    "Dream Fitness"
)

plt.show()




sample,target=dataset[0]

sample=sample.unsqueeze(
    0
).to(device)

dream=model(
    sample
)

dream=dream[0].permute(
    1,
    2,
    0
)

dream=dream.detach().cpu().numpy()

target=target.permute(
    1,
    2,
    0
).numpy()



gray=np.mean(
    dream,
    axis=2
)

y,x=np.unravel_index(
    np.argmax(gray),
    gray.shape
)

kf.predict()

kf.update(
    [x,y]
)

print()

print(
    "Filtered position:"
)

print(
    int(kf.x[0]),
    int(kf.x[1])
)




plt.figure(
    figsize=(10,5)
)

plt.subplot(
    1,
    2,
    1
)

plt.imshow(
    target
)

plt.title(
    "Real Frame"
)

plt.axis(
    "off"
)

plt.subplot(
    1,
    2,
    2
)

plt.imshow(
    dream
)

plt.title(
    "Dream Reconstruction"
)

plt.axis(
    "off"
)

plt.show()




original_bits=(
    FRAME_SIZE*
    FRAME_SIZE*
    3*
    8
)

compressed_bits=(
    LATENT_SIZE*
    32
)

ratio=(
    original_bits/
    compressed_bits
)

print("\n======")

print(
    f"Compression: {ratio:.2f}:1"
)
#you see i added these equal signs to make it fancy spancy
print("======")
#signed kds1123001 aka krishndev sen 
#also opencv gets a lil frisky with 3.14 if u wanna run it install the libraries and change the directory to where you saved this file and run it like py -3.12 dream_compression.p
#please keep it opensource and when sharing it remember my name bleh   :3
#also like change the code to your liking cuz i love you buh bye!!!