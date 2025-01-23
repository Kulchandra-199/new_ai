import tkinter as tk
from tkinter import scrolledtext
from io import StringIO
import sys

class CodeEditor:
    def __init__(self, root):
        self.root = root
        self.root.title("Custom Python Notebook UI")
        
        # Code Input Area
        self.code_input = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=80, height=20)
        self.code_input.pack(padx=10, pady=10)
        
        # Run Button
        self.run_button = tk.Button(root, text="Run Code", command=self.execute_code)
        self.run_button.pack(pady=5)
        
        # Output Area
        self.output_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=80, height=10)
        self.output_area.pack(padx=10, pady=10)
        self.output_area.config(state=tk.DISABLED)  # Make output area read-only

    def execute_code(self):
        """
        Execute the code in the input area and display the output.
        """
        # Get code from the input area
        code = self.code_input.get("1.0", tk.END)
        
        # Redirect stdout to capture output
        old_stdout = sys.stdout
        sys.stdout = output_capture = StringIO()
        
        try:
            # Execute the code
            exec(code)
        except Exception as e:
            # Display errors in the output area
            print(f"Error: {e}")
        finally:
            # Restore stdout
            sys.stdout = old_stdout
        
        # Get the output and display it
        output = output_capture.getvalue()
        self.output_area.config(state=tk.NORMAL)
        self.output_area.delete("1.0", tk.END)
        self.output_area.insert(tk.END, output)
        self.output_area.config(state=tk.DISABLED)

# Create the main window
root = tk.Tk()
editor = CodeEditor(root)
root.mainloop()