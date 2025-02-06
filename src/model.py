import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.7):
    #def __init__(self, num_classes, dropout):

        super(MyModel,self).__init__()
        
        self.features = nn.Sequential(
          nn.Conv2d(3,32,kernel_size = 3,stride=1,padding=1),  # 224, 224, 32
          nn.ReLU(),
          nn.BatchNorm2d(32),
          nn.MaxPool2d(kernel_size=2,stride=2), # 112, 112, 32

          nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), # 112, 112, 64
          nn.ReLU(),
          nn.BatchNorm2d(64),
          nn.MaxPool2d(kernel_size=2, stride=2),   # 56, 56, 64

          nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),   # 56, 56, 128 
          nn.ReLU(),
          nn.BatchNorm2d(128),
          nn.MaxPool2d(kernel_size=2, stride=2),   # 28, 28, 128
          
          nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),   # 28, 28, 256 
          nn.ReLU(),
          nn.BatchNorm2d(256),
          nn.MaxPool2d(kernel_size=2, stride=2),   # 14, 14, 256
          
          nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),   # 14, 14, 512 
          nn.ReLU(),
          nn.BatchNorm2d(512),
          nn.MaxPool2d(kernel_size=2, stride=2),   # 7, 7, 128
          
          nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),   # 7, 7, 1024 
          nn.ReLU(),
          nn.BatchNorm2d(1024),
          nn.MaxPool2d(kernel_size=2, stride=2),   # 3, 3, 1024
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((3, 3))

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(1024 * 3 * 3, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(512, num_classes),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
        
# class MyModel(nn.Module):
#     def __init__(self, num_classes: int = 1000, dropout: float = 0.7) -> None:
#         super(MyModel, self).__init__()

#         self.features = nn.Sequential(
#             # Adding the first convolution layer with 32 out channels
#             nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True),

#             nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),

#             nn.Conv2d(64, 128, kernel_size=3, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),

#             nn.Conv2d(128, 256, kernel_size=3, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),

#             nn.Conv2d(256, 1024, kernel_size=3, padding=1),
#             nn.BatchNorm2d(1024),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#         )

#         self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

#         # Modifying the classifier with just two layers and including suitable dropout layers
#         self.classifier = nn.Sequential(
#             nn.Dropout(p=dropout),
#             nn.Linear(1024 * 7 * 7, 4096),
#             nn.ReLU(inplace=True),
#             #nn.Dropout(p=dropout),
#             nn.Linear(4096,1024),
#             nn.ReLU(inplace=True),
#             nn.Dropout(p=dropout),            
#             nn.Linear(1024, num_classes)
#         )

#model = MyModel()


######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=2)


def test_model_construction(data_loaders):

    model = MyModel(num_classes=23, dropout=0.3)

    dataiter = iter(data_loaders["train"])
    images, labels = next(dataiter)

    out = model(images)

    assert isinstance(
        out, torch.Tensor
    )
    "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 23]
    ), f"Expected an output tensor of size (2, 23), got {out.shape}"
