from tkinter import *
from tkinter import DoubleVar

root = Tk()
mainframe = Frame(root)
top_frame = LabelFrame(mainframe, text="1. Select Image", padx=10, pady=10)
middle_frame = LabelFrame(mainframe, text="2. Configure Generation", padx=10, pady=10)
bottom_frame = LabelFrame(mainframe, text="3. Show results", padx=10, pady=10)

create_image_button = Button(top_frame, text="create base Image", bg="midnight blue", fg="white")
or_lbl = Label(top_frame, text="OR", padx=65, pady=(5))
orig_dir_button = Button(top_frame, text="choose own image", bg="midnight blue", fg="white")
color_lbl = Label(top_frame, text="Color:")
color_var = StringVar()
color_entry = Entry(top_frame, text="color", textvariable=color_var)
color_entry.insert(0, "Random")

create_adv_button = Button(middle_frame, text="create Adversarial", bg="midnight blue", fg="white")
iter_lbl = Label(middle_frame, text="number of iterations:")
iter_var = IntVar()
iter_entry = Entry(middle_frame, textvariable=iter_var)
iter_entry.insert(0, 3)
epsilon_lbl = Label(middle_frame, text="epsilon:")
epsilon_var = DoubleVar()
epsilon_entry = Entry(middle_frame, textvariable=epsilon_var)
epsilon_entry.delete(0, "end")
epsilon_entry.insert(0, 0.25)
alpha_lbl = Label(middle_frame, text="alpha:")
alpha_var = DoubleVar()
alpha_entry = Entry(middle_frame, text="0.025", textvariable=alpha_var)
alpha_entry.insert(3, 25)

adv_button = Button(bottom_frame, text="folder of adversarials", bg="midnight blue", fg="white")

mainframe.grid(column=0, row=0, sticky=(N, S, E, W), padx=5, pady=5)
top_frame.grid(column=0, row=0, sticky=(N, S, W, E), padx=5, pady=5)
middle_frame.grid(column=0, row=1,sticky=(W, E), padx=5, pady=5)
bottom_frame.grid(column=0, row=2, sticky=(W, E), padx=5, pady=5)

create_image_button.grid(column=0, row=1, columnspan=2, sticky=(N))
color_lbl.grid(column=0, row=0, sticky=(S, E))
color_entry.grid(column=1, row=0, sticky=(W))
or_lbl.grid(column=0, row=2, columnspan=2)
orig_dir_button.grid(column=0, row=3, columnspan=2)

create_adv_button.grid(column=0, row=4, columnspan=2)
iter_lbl.grid(column=0, row=1, sticky=(E))
iter_entry.grid(column=1, row=1, sticky=W)
epsilon_lbl.grid(column=0, row=2, sticky=(E))
epsilon_entry.grid(column=1, row=2, sticky=W)
alpha_lbl.grid(column=0, row=3, sticky=(E))
alpha_entry.grid(column=1, row=3, sticky=W)

adv_button.grid(column=0, row=0)

root.rowconfigure(0, weight=1)
root.columnconfigure(0, weight=1)
mainframe.columnconfigure(0, weight=1)
mainframe.rowconfigure(0, weight=1)
mainframe.rowconfigure(1, weight=1)
mainframe.rowconfigure(2, weight=1)

top_frame.columnconfigure(0, weight=1)
top_frame.columnconfigure(1, weight=1)
top_frame.rowconfigure(0, weight=1)
top_frame.rowconfigure(1, weight=1)
top_frame.rowconfigure(2, weight=1)
top_frame.rowconfigure(3, weight=1)

middle_frame.columnconfigure(0, weight=1)
middle_frame.columnconfigure(1, weight=1)
middle_frame.rowconfigure(0, weight=1)
middle_frame.rowconfigure(1, weight=1)
middle_frame.rowconfigure(2, weight=1)
middle_frame.rowconfigure(3, weight=1)

bottom_frame.rowconfigure(0, weight=1)
bottom_frame.columnconfigure(0, weight=1)



mainloop()
