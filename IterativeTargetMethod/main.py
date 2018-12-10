# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 11:58:55 2018

@author: marvi
"""
import generateadversarialsfinal
import gui
import generateimage
import tkinter as tk

def main():
    root = tk.Tk()
    ui = gui.Gui(root)
    ui.mainloop()
    advgen = generateadversarialsfinal.AdvGenerator()
    print(ui.color_entry.get())

if __name__ == "__main__":
    main()
    