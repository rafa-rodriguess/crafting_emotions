import torch
import torch.nn as nn
import pytorch_lightning as pl

class BaseCNN(pl.LightningModule):
    def __init__(self, num_classes=8, learning_rate=0.001):
        super().__init__()
        self.learning_rate = learning_rate
        self.criterion = nn.CrossEntropyLoss()
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        self.log("val_acc", acc, prog_bar=True, on_epoch=True)
        return {"val_loss": loss, "val_acc": acc}
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log("test_acc", acc, prog_bar=True, on_epoch=True)
        return {"test_acc": acc}

class CNNModel0(BaseCNN):
    def __init__(self, input_shape, num_classes=8, learning_rate=0.001):
        super().__init__(num_classes, learning_rate)
        in_channels = input_shape[0]  # Dynamically get the number of input channels
        
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten()
        )
        
        # Calculate the flattened size dynamically
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            dummy_output = self.model(dummy_input)
            flattened_size = dummy_output.shape[1]
        
        self.classifier = nn.Sequential(
            nn.Linear(flattened_size, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.model(x)
        x = self.classifier(x)
        return x

class CNNModel1(CNNModel0):
    def __init__(self, input_shape, num_classes=8, learning_rate=0.001):
        super().__init__(input_shape, num_classes, learning_rate)
        self.classifier.insert(1, nn.BatchNorm1d(256))

class CNNModel2(CNNModel0):
    def __init__(self, input_shape, num_classes=8, learning_rate=0.001):
        super().__init__(input_shape, num_classes, learning_rate)
        self.classifier[3] = nn.Linear(256, 512)
        self.classifier.insert(4, nn.ReLU())
        self.classifier.insert(5, nn.Dropout(0.6))

class CNNModel3(CNNModel0):
    def __init__(self, input_shape, num_classes=8, learning_rate=0.001):
        super().__init__(input_shape, num_classes, learning_rate)
        in_channels = input_shape[0]  # Dynamically get the number of input channels
        self.model[0] = nn.Conv2d(in_channels, 32, kernel_size=5, padding=2)
        self.model[3] = nn.Conv2d(32, 64, kernel_size=5, padding=2)

class CNNModel4(CNNModel0):
    def __init__(self, input_shape, num_classes=8, learning_rate=0.001):
        super().__init__(input_shape, num_classes, learning_rate)
        self.classifier[3] = nn.Linear(256, 128)









