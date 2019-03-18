#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 11:07:31 2019

@author: giuseppelisi
"""


import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from parameters import directory
from utils import plot_multi_stroke
import tkinter

if len(sys.argv) != 2:
    raise Exception('Please provide the name of the digit in the command line')

if not os.path.exists(directory):
    os.makedirs(directory)

root = tkinter.Tk()

filename = os.path.join(directory,'{}_traj'.format(sys.argv[1]))
 
width = 500
height = 500
white = (255, 255, 255)

saved_traj = [[]]

def save():
    print("Saving")
    del saved_traj[-1]
    np.save(filename,saved_traj) 
    root.destroy()

def paint(event):
    # python_green = "#476042"
    x1, y1 = (event.x - 1), (event.y - 1)
    x2, y2 = (event.x + 1), (event.y + 1)
    cv.create_oval(x1, y1, x2, y2, fill="black",width=5)
    saved_traj[-1].append([event.x/width,1-event.y/height])
    
    
def release_callback(event):
    saved_traj.append([])
    
# Tkinter create a canvas to draw on
cv = tkinter.Canvas(root, width=width, height=height, bg='white')
cv.pack()

cv.pack(expand=tkinter.YES, fill=tkinter.BOTH)
cv.bind("<B1-Motion>", paint)
cv.bind("<ButtonRelease-1>", release_callback)

button=tkinter.Button(text="Save",command=save)
button.pack()
root.mainloop()

plt.figure()
plot_multi_stroke(saved_traj)  
plt.show()

