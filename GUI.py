from tkinter import *
import cv2
from PIL import Image, ImageGrab, ImageDraw
import numpy as np
import tensorflow as tf

last = [None, None]
model = tf.keras.models.load_model('MnistModel.h5')

#functions
def draw_point(event):
    global last
    x,y = event.x + 4, event.y + 4
    x1,y1 = event.x - 4, event.y - 4
    c.create_oval(x,y,x1,y1, fill = 'black')
    last = [event.x, event.y]

def draw_line(event):
    global last
    c.create_line(last[0], last[1],event.x, event.y, fill = 'black', width = 8, capstyle = 'round', smooth = TRUE)
    draw.line([(last[0], last[1]), (event.x, event.y)], fill = 'black', width = 15, joint = 'round')
    last = [event.x, event.y]

def predict():
    filename = "new_image.png"
    pil_image.save(filename)
    #pil_image.show()
    
    img = convert_img()
    pred=model.predict(img)

    pred_label['text'] = "Prediction: " + str(np.argmax(pred[0]))
    
def clear():
    c.delete('all')
    draw.rectangle([(0,0), (400,400)], fill = (255,255,255))
    pred_label['text'] = "Prediction:   "

def convert_img():
    img=cv2.imread('new_image.png', 0)
    
    img = cv2.bitwise_not(img)
    
    img = cv2.resize(img,(28,28))
    img = img.reshape(1,28,28,1)
    img = img.astype('float32')
    img = img/255.0

    return img

#tkinter
root = Tk()
root.resizable(0,0)
root.title('Draw a Digit 0-9')

#Frames
Frame1 = LabelFrame(root, padx = 5, pady = 5)
Frame1.config(highlightbackground = 'SteelBlue3', highlightcolor = 'SteelBlue3', highlightthickness = 2)
Frame1.pack(side = LEFT, padx = 10, pady = 10)

Frame2 = LabelFrame(root, padx = 25, pady= 25)
Frame2.config(highlightthickness = 0, borderwidth = 0)
Frame2.pack(side = RIGHT, padx = 10, pady = 10)

#Label
pred_label = Label(Frame2, height = 3, text = "Prediction:   ", bg = 'SteelBlue3')
pred_label.config(highlightbackground = 'blue', highlightcolor = 'blue', highlightthickness = 2)
pred_label.pack()

#canvas 
c = Canvas(Frame1, width=400, height = 400, bg = 'white')
c.pack()

#buttons
b_predict = Button(Frame2, width = 10, height = 5, fg='SteelBlue3', text = 'PREDICT', command = predict)
b_predict.pack()

b_clear = Button(Frame2, width = 10, height = 5, fg='SteelBlue3', text = 'CLEAR', command = clear)
b_clear.pack()

#binds
c.bind('<Button-1>', draw_point)
c.bind('<B1-Motion>', draw_line)

#PIL
pil_image = Image.new("RGB", (400,400), color = (255,255,255))
draw = ImageDraw.Draw(pil_image)

root.mainloop()