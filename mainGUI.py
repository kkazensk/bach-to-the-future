#!/usr/bin/env python3

#thanks to chatGPT for helping me figure out to structure the file and class so I can deal with inport from mutiple files. Also it helped me create the dial structure as the documentation on how to create it was poor

import tkinter as tk
from tkinter import filedialog
from tkinter import ttk

class MultiFileButtonApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Bach to the Future")
        self.geometry("600x400")

        # Frame to hold buttons
        self.button_frame = tk.Frame(self)
        self.button_frame.pack(pady=20)

        # Add file button
        tk.Button(self, text="Add Files", command=self.add_files).pack(pady=10)

        # Store file-button mapping
        self.file_buttons = {}

    def add_files(self):
        file_paths = filedialog.askopenfilenames(
            filetypes=[("MusicXML or PDF", "*.musicxml *.xml *.pdf")]
        )

        for file_path in file_paths:
            if file_path not in self.file_buttons:
                btn = tk.Button(
                    self.button_frame,
                    text=f"Open {file_path.split('/')[-1]}",
                    command=lambda p=file_path: self.handle_file(p)
                )
                btn.pack(pady=2)
                self.file_buttons[file_path] = btn

    def handle_file(self, path):
        print(f"Button clicked for file: {path}")
        # Replace this with your actual logic (e.g., convert/view/etc)
        

if __name__ == "__main__":
    app = MultiFileButtonApp()
    app.mainloop()