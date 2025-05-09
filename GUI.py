#!/usr/bin/env python3
# main_gui.py
import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from tkinter import messagebox
import xml.etree.ElementTree as ET
import subprocess
import os
from handlers import analyze_file  # ✅ import your logic function
import threading

class MultiFileApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Advanced File Analyzer")
        self.geometry("600x500")

        self.file_buttons = {}
        self.key_var = tk.StringVar(value="C")
        self.transpose_var = tk.BooleanVar(value=False)

        # File button frame
        self.button_frame = tk.Frame(self)
        self.button_frame.pack(pady=10)

        # (Buttons for running scripts will appear after status label)

        self.status_label = tk.Label(self, text="Status: Ready", anchor="w", fg="blue")
        self.status_label.pack(fill="x", padx=10, pady=(0, 10))

        tk.Button(self, text="Run Training Script", command=self.run_training).pack(pady=5)
        tk.Button(self, text="Run Generation Script", command=self.run_generation).pack(pady=5)
        
        # Add file button
        tk.Button(self, text="Add Files", command=self.add_files).pack(pady=10)

        # Key selector dropdown
        tk.Label(self, text="Select Key:").pack()
        keys = ['C', 'G', 'D', 'A', 'E', 'F', 'B♭', 'E♭', 'A minor', 'E minor', 'D minor']
        ttk.OptionMenu(self, self.key_var, keys[0], *keys).pack(pady=5)

        # Transpose checkbox
        ttk.Checkbutton(self, text="Apply Transposition", variable=self.transpose_var).pack(pady=5)
        
        self.progress_bar = ttk.Progressbar(self, mode="indeterminate")
        self.progress_bar.pack(fill="x", padx=10, pady=(0, 10))
        
        self.output_box = tk.Text(self, height=10, wrap="word")
        self.output_box.pack(pady=10, fill="x", padx=10)


    def add_files(self):
        file_paths = filedialog.askopenfilenames(filetypes=[("MusicXML or PDF", "*.musicxml *.xml *.pdf")])
        for file_path in file_paths:
            filename = os.path.basename(file_path)
            if file_path not in self.file_buttons:
                tk.Button(
                    self.button_frame,
                    text=f"Analyze {filename}",
                    command=lambda p=file_path: self.run_analysis(p)
                ).pack(pady=3)
                self.file_buttons[file_path] = True
        self.last_selected_file = file_path

    def run_analysis(self, file_path):
        self.status_label.config(text="Analyzing selected file...", fg="orange")
        self.update_idletasks()

        key = self.key_var.get()
        should_transpose = self.transpose_var.get()

        valid_extensions = ('.pdf', '.xml', '.musicxml')
        if not file_path.lower().endswith(valid_extensions):
            messagebox.showerror("Invalid File Type", f"The file '{file_path}' is not a supported format.")
            return

        if file_path.lower().endswith(('.xml', '.musicxml')):
            try:
                tree = ET.parse(file_path)
                root = tree.getroot()
                movement_title = root.find('movement-title')
                measures = root.findall('.//measure')
                first_measure = root.find('.//measure')
                beats = first_measure.find('./attributes/time/beats') if first_measure is not None else None
                xml_key = first_measure.find('./attributes/key/fifths') if first_measure is not None else None

                fifths_map = {
                    '-7': 'C♭ major/A♭ minor', '-6': 'G♭ major/E♭ minor', '-5': 'D♭ major/B♭ minor',
                    '-4': 'A♭ major/F minor', '-3': 'E♭ major/C minor', '-2': 'B♭ major/G minor',
                    '-1': 'F major/D minor', '0': 'C major/A minor', '1': 'G major/E minor',
                    '2': 'D major/B minor', '3': 'A major/F♯ minor', '4': 'E major/C♯ minor',
                    '5': 'B major/G♯ minor', '6': 'F♯ major/D♯ minor', '7': 'C♯ major/A♯ minor'
                }
                key_readable = fifths_map.get(xml_key.text, f"Unknown ({xml_key.text})") if xml_key is not None else "N/A"

                analysis_result = f"Parsed File: {os.path.basename(file_path)}\n\n"
                analysis_result += f"Movement Title: {movement_title.text if movement_title is not None else 'Not found in <movement-title>'}\n"
                analysis_result += f"Number of Measures: {len(measures)}\n"

                if beats is not None:
                    analysis_result += f"Beats per Measure: {beats.text}\n"
                else:
                    analysis_result += "Beats per Measure: Not found (expected under <attributes><time><beats>)\n"

                analysis_result += f"Key: {key_readable}\n"
                
                self.output_box.delete("1.0", tk.END)
                self.output_box.insert(tk.END, analysis_result)

            except Exception as e:
                messagebox.showerror(
                    "Parsing Error",
                    "The MusicXML file could not be parsed.\n\n"
                    "Please check for the following issues:\n"
                    "• The XML structure is malformed (missing or unclosed tags)\n"
                    "• Required elements like <measure> or <attributes> are missing\n"
                    "• The file is encoded incorrectly or uses unsupported symbols\n\n"
                    f"Technical details:\n{e}"
                )
                return

        analyze_file(file_path, key, should_transpose)
        
    def run_training(self):
        threading.Thread(target=self._run_training_logic, daemon=True).start()

    def _run_training_logic(self):
        key = self.key_var.get()
        self.status_label.config(text="Training script is now running...", fg="orange")
        self.update_idletasks()
        self.progress_bar.start()
        try:
            result = subprocess.run(
                ["python3", "train_1.py", "--key", key],
                check=True,
                capture_output=True,
                text=True
            )
            self.output_box.delete("1.0", tk.END)
            self.output_box.insert(tk.END, result.stdout)
            if result.stderr:
                self.output_box.insert(tk.END, "\nErrors:\n" + result.stderr)
        except subprocess.CalledProcessError as e:
            self.output_box.delete("1.0", tk.END)
            self.output_box.insert(tk.END, f"Training script failed:\n{e}\n{e.stderr}")
            self.status_label.config(text="Training failed.", fg="red")
        finally:
            self.progress_bar.stop()

    def run_generation(self):
        threading.Thread(target=self._run_generation_logic, daemon=True).start()

    def _run_generation_logic(self):
        try:
            file_arg = getattr(self, 'last_selected_file', None)
            if not file_arg:
                messagebox.showwarning("No File", "Please add and select a file first.")
                return

            key = self.key_var.get()
            self.status_label.config(text="Generation script is now running...", fg="orange")
            self.update_idletasks()
            self.progress_bar.start()
            result = subprocess.run(
                ["python3", "generate_2.py", "--input", file_arg, "--key", key],
                check=True,
                capture_output=True,
                text=True
            )
            self.output_box.delete("1.0", tk.END)
            self.output_box.insert(tk.END, result.stdout)
            if result.stderr:
                self.output_box.insert(tk.END, "\nErrors:\n" + result.stderr)
        except subprocess.CalledProcessError as e:
            self.output_box.delete("1.0", tk.END)
            self.output_box.insert(tk.END, f"Generation script failed:\n{e}\n{e.stderr}")
            self.status_label.config(text="Generation failed.", fg="red")
        finally:
            self.progress_bar.stop()
        

if __name__ == "__main__":
    app = MultiFileApp()
    app.mainloop()