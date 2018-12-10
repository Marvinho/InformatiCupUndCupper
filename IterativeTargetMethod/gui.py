import tkinter as tk
from tkinter import N, E, W, S, HORIZONTAL, VERTICAL
import generateimage
from tkinter.filedialog import askopenfilename
import generateadv
import modelcnn
from tkinter import ttk

class Gui():
    def __init__(self, root):
        self.root = root
        root.title("GUI")
#        root = tk.Tk()
        self.mainframe = tk.Frame(root)
        self.top_frame = tk.LabelFrame(self.mainframe, text="1. Select Image", padx=10, pady=10)
        self.middle_frame = tk.LabelFrame(self.mainframe, text="2. Configure Generation", padx=10, pady=10)
#        self.bottom_frame = tk.LabelFrame(self.mainframe, text="3. Show results", padx=10, pady=10)

        self.color_lbl = tk.Label(self.top_frame, text="Color:")
        self.color_var = tk.StringVar()
        self.color_entry = tk.Entry(self.top_frame, text="color", textvariable=self.color_var, width=11)
        self.color_entry.insert(0, "Random")
        self.color_entry.get()
        
        self.create_image_button = tk.Button(self.top_frame, text="create base image", command= lambda: onClickCreateImage(), bg="midnight blue", fg="white", width=16)
        self.or_lbl = tk.Label(self.top_frame, text="OR", padx=30, pady=(5))
        self.choose_image_button = tk.Button(self.top_frame, text="choose own image", command= lambda: onClickChooseImage(), bg="midnight blue", fg="white", width=16)
        self.seperator0 = ttk.Separator(self.top_frame, orient=VERTICAL)
        self.base_folder_button = tk.Button(self.top_frame, text="show base images", command= lambda: onClickShowBaseFolder(), bg="midnight blue", fg="white", width=16)

        
        self.create_adv_button = tk.Button(self.middle_frame, text="create adversarial", command=lambda: onClickAdvButton(), bg="midnight blue", fg="white", width=16)
        
        self.iter_lbl = tk.Label(self.middle_frame, text="iterations:")
        self.iter_var = tk.IntVar()
        self.iter_entry = tk.Entry(self.middle_frame, textvariable=self.iter_var, width=7)
        self.iter_entry.insert(0, 3)
        print(self.iter_var.get())
        
        self.epsilon_lbl = tk.Label(self.middle_frame, text="epsilon:")
        self.epsilon_var = tk.DoubleVar()
        self.epsilon_entry = tk.Entry(self.middle_frame, textvariable=self.epsilon_var, width=7)
        self.epsilon_entry.delete(0, "end")
        self.epsilon_entry.insert(0, 0.25)
        print(self.epsilon_entry.get())
        
        self.alpha_lbl = tk.Label(self.middle_frame, text="alpha:")
        self.alpha_var = tk.DoubleVar()
        self.alpha_entry = tk.Entry(self.middle_frame, text="0.025", textvariable=self.alpha_var, width=7)
        self.alpha_entry.insert(3, 25)
        print(self.alpha_entry.get())
        
        self.seperator1 = ttk.Separator(self.middle_frame, orient=VERTICAL)
        self.adv_folder_button = tk.Button(self.middle_frame, text="show adversarials", command=lambda: onClickShowResults(), bg="midnight blue", fg="white", width=16)
        
        self.mainframe.grid(column=0, row=0, sticky=(N, S, E, W), padx=5, pady=5)
        self.top_frame.grid(column=0, row=0, sticky=(N, S, W, E), padx=5, pady=5)
        self.middle_frame.grid(column=0, row=1,sticky=(W, E), padx=5, pady=5)
#        self.bottom_frame.grid(column=0, row=2, sticky=(W, E), padx=5, pady=5)
        
        self.create_image_button.grid(column=0, row=1, columnspan=2, padx=(25,0))
        self.color_lbl.grid(column=0, row=0, sticky=(E), padx=(25,0), pady=(15,0))
        self.color_entry.grid(column=1, row=0, sticky=(W), pady=(15,0))
        self.or_lbl.grid(column=0, row=3, columnspan=2, padx=(25,0))
        self.choose_image_button.grid(column=0, row=4, columnspan=2, padx=(25,0), pady=(0,15))
        self.seperator0.grid(column=2, row=0, rowspan=6, sticky=(N, S), padx=50)
        self.base_folder_button.grid(column=3, row=0, rowspan=5, padx=(0,15))
             
        self.iter_lbl.grid(column=0, row=1, sticky=(E), padx=(25,0), pady=(15,0))
        self.iter_entry.grid(column=1, row=1, sticky=W, pady=(15,0))
        self.epsilon_lbl.grid(column=0, row=2, sticky=(E), padx=(25,0))
        self.epsilon_entry.grid(column=1, row=2, sticky=W)
        self.alpha_lbl.grid(column=0, row=3, sticky=(E), padx=(25,0))
        self.alpha_entry.grid(column=1, row=3, sticky=W)
        self.create_adv_button.grid(column=0, row=4, columnspan=2, padx=(25,0), pady=(0,15))
        
        self.adv_folder_button.grid(column=3, row=0, rowspan=5, sticky=(E, W), padx=(0,15))
        self.seperator1.grid(column=2, row=0, rowspan=5, sticky=(N, S), padx=(50,50))
        
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)
        self.mainframe.columnconfigure(0, weight=1)
        self.mainframe.rowconfigure(0, weight=1)
        self.mainframe.rowconfigure(1, weight=1)
        self.mainframe.rowconfigure(2, weight=1)
        
        self.top_frame.columnconfigure(0, weight=1)
        self.top_frame.columnconfigure(1, weight=1)
        self.top_frame.rowconfigure(0, weight=1)
        self.top_frame.rowconfigure(1, weight=1)
        self.top_frame.rowconfigure(2, weight=1)
        self.top_frame.rowconfigure(3, weight=1)
        
        self.middle_frame.columnconfigure(0, weight=1)
        self.middle_frame.columnconfigure(1, weight=1)
        self.middle_frame.rowconfigure(0, weight=1)
        self.middle_frame.rowconfigure(1, weight=1)
        self.middle_frame.rowconfigure(2, weight=1)
        self.middle_frame.rowconfigure(3, weight=1)
        
#        self.bottom_frame.rowconfigure(0, weight=1)
#        self.bottom_frame.columnconfigure(0, weight=1)
        
        def onClickShowBaseFolder():
            pass
        
        def onClickChooseImage():
            filename = askopenfilename()
            generateimage.chooseImage(filename)
        
        def onClickCreateImage():
            generateimage.createImage(self.color_var.get())
            
        def onClickShowBaseFolder():
            generateimage.openDir(foldername="./Images/originals/")
        
        def onClickShowResults():
            generateimage.openDir(foldername="./adversarials/")
            
        def onClickAdvButton():
            adv = generateadv.AdvGenerator()
            adv.generateAdv()

if __name__ == "__main__":  
    root = tk.Tk()
    gui = Gui(root)   
    tk.mainloop()
