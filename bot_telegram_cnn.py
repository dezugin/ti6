from telegram.ext.updater import Updater
from telegram.update import Update
from telegram.ext.callbackcontext import CallbackContext
from telegram.ext.commandhandler import CommandHandler
from telegram.ext.messagehandler import MessageHandler
from telegram.ext.filters import Filters
from io import BytesIO
import cv2
import numpy
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torchvision.transforms import functional as TF
from torchvision import transforms
from PIL import Image
from subprocess import Popen, PIPE


#paralelizador do codigo via cuda
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device

#paralelizador do codigo via openmp
if not torch.cuda.is_available():
    torch.set_num_threads(3)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()        
        self.conv1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=1,              
                out_channels=16,            
                kernel_size=5,              
                stride=1,                   
                padding=2,                  
            ),                              
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=2),    
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(16, 32, 5, 1, 2),     
            nn.ReLU(),                      
            nn.MaxPool2d(2),                
        )        # camada completamente conectada, saída 10 classes
        self.out = nn.Linear(32 * 7 * 7, 10)    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)        # operação de flatten da saída de conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)       
        output = self.out(x)
        return output, x    # retornar x para visualização



train_data = datasets.MNIST(
    root = 'data',
    train = True,                         
    transform = ToTensor(), 
    download = True,            
)
test_data = datasets.MNIST(
    root = 'data', 
    train = False, 
    transform = ToTensor()
)


loaders = {
    'train' : torch.utils.data.DataLoader(train_data, 
                                          batch_size=100, 
                                          shuffle=True, 
                                          num_workers=1),
    
    'test'  : torch.utils.data.DataLoader(test_data, 
                                          batch_size=100, 
                                          shuffle=True, 
                                          num_workers=1),
}
loaders



cnn = CNN()
loss_func = nn.CrossEntropyLoss()   
from torch import optim
optimizer = optim.Adam(cnn.parameters(), lr = 0.01)  
from torch.autograd import Variable
num_epochs = 10
def train(num_epochs, cnn, loaders):
    
    cnn.train()
        
    # treinar o modelo
    total_step = len(loaders['train'])
        
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(loaders['train']):
            
            # dá lote de dados, normaliza x quando itera train_loader
            b_x = Variable(images)   # lote x
            b_y = Variable(labels)   # lote y
            output = cnn(b_x)[0]               
            loss = loss_func(output, b_y)
            
            # limpar os gradientes para o passo de treinamento   
            optimizer.zero_grad()           
            
            # backpropagation, computar gradientes 
            loss.backward()                # aplicar gradientes             
            optimizer.step()                
            
            if (i+1) % 100 == 0:
                print ('Época [{}/{}], Passo [{}/{}], Perda: {:.4f}' 
                       .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))               
                pass
        
        pass
    
    
    pass
train(num_epochs, cnn, loaders)


# identificador do bot de telegram
updater = Updater("5608264386:AAHq9m4-c8EO9LNL9mK8UXEpk2qYTE06hZA",
                  use_context=True)
  
  
def start(update: Update, context: CallbackContext):
    update.message.reply_text(
        "Bem Vindo ao bot de processamento de digitos da aula de PAI de 2022/2. Me mande uma foto")
def help(update: Update, context: CallbackContext):
    update.message.reply_text("Mande uma foto")


def unknown_text(update: Update, context: CallbackContext):
	update.message.reply_text(
		"Sorry I can't recognize you , you said '%s'" % update.message.text)


def unknown(update: Update, context: CallbackContext):
	update.message.reply_text(
		"Sorry '%s' is not a valid command" % update.message.text)

def cnn_bot_function(input_stream):
    transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor()])

    cnn.eval()
    img = cv2.imdecode(numpy.fromstring(input_stream.read(), numpy.uint8), 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img_gray = TF.to_grayscale(img)
    img_tensor = transform(img_gray)
    img_tensor_array = img_tensor.unsqueeze(0)
    out = cnn(img_tensor_array)[0]
    result = torch.max(out, 1)[1].data.squeeze().numpy()

    return result


def photo(update: Update, context: CallbackContext):
    file = context.bot.get_file(update.message.photo[-1].file_id)
    f =  BytesIO(file.download_as_bytearray())

    # f se torna um objeto de arquivo para ser manipulado

    result = cnn_bot_function(f)

    response = 'O dígito enviado é %s' % (result,)

    context.bot.send_message(chat_id=update.message.chat_id, text=response)

updater.dispatcher.add_handler(CommandHandler('start', start))
updater.dispatcher.add_handler(CommandHandler('help', help))
photo_handler = MessageHandler(Filters.photo, photo)
updater.dispatcher.add_handler(photo_handler)
updater.dispatcher.add_handler(MessageHandler(Filters.text, unknown))
updater.dispatcher.add_handler(MessageHandler(
	# Filtra comandos desconhecidos
	Filters.command, unknown))
updater.start_polling()


# Filtra mensagens desconhecidas
updater.dispatcher.add_handler(MessageHandler(Filters.text, unknown_text))

