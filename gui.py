"""
Modern GUI for Real-ESRGAN Image Upscaler
Redesigned with better UX, clearer layout, and flexible controls
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
import threading
import sys
import os

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from upscaler import upscale_image, upscale_batch, download_models, check_memory_usage, MAX_MEMORY_GB, format_time
import time


class ModernUpscalerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Gandalf of Technology Upscaler")
        self.root.geometry("1450x850")
        self.root.minsize(800, 700)
        
        # Color scheme - Modern dark theme
        self.colors = {
            'bg_dark': '#0d1117',
            'bg_card': '#161b22',
            'bg_input': '#21262d',
            'accent': '#238636',
            'accent_hover': '#2ea043',
            'danger': '#da3633',
            'warning': '#d29922',
            'text': '#c9d1d9',
            'text_dim': '#8b949e',
            'border': '#30363d',
            'highlight': '#58a6ff'
        }
        
        self.root.configure(bg=self.colors['bg_dark'])
        
        # Variables
        self.input_path = tk.StringVar()
        self.output_path = tk.StringVar()
        self.scale = tk.IntVar(value=4)
        self.model = tk.StringVar(value="general")
        self.face_enhance = tk.BooleanVar(value=False)
        self.output_format = tk.StringVar(value="png")
        self.dpi = tk.IntVar(value=72)
        self.suffix = tk.StringVar(value="_upscaled")
        self.preserve_metadata = tk.BooleanVar(value=True)
        self.remove_watermark = tk.BooleanVar(value=False)
        self.grain_strength = tk.DoubleVar(value=0.0)
        self.sharpen_amount = tk.DoubleVar(value=0.0)
        
        # Processing state
        self.is_processing = False
        self.should_cancel = False
        self.last_heartbeat = 0
        self.heartbeat_timeout = 60
        self.process_start_time = 0
        
        self.setup_styles()
        self.create_layout()
        
    def setup_styles(self):
        """Configure ttk styles for modern look."""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Frame styles
        style.configure('Card.TFrame', background=self.colors['bg_card'])
        style.configure('Dark.TFrame', background=self.colors['bg_dark'])
        
        # Label styles
        style.configure('Title.TLabel', 
                       background=self.colors['bg_dark'],
                       foreground=self.colors['text'],
                       font=('Segoe UI', 24, 'bold'))
        style.configure('Subtitle.TLabel',
                       background=self.colors['bg_dark'],
                       foreground=self.colors['text_dim'],
                       font=('Segoe UI', 10))
        style.configure('Card.TLabel',
                       background=self.colors['bg_card'],
                       foreground=self.colors['text'],
                       font=('Segoe UI', 10))
        style.configure('CardTitle.TLabel',
                       background=self.colors['bg_card'],
                       foreground=self.colors['text'],
                       font=('Segoe UI', 11, 'bold'))
        style.configure('Status.TLabel',
                       background=self.colors['bg_card'],
                       foreground=self.colors['highlight'],
                       font=('Segoe UI', 10))
        
        # Button styles
        style.configure('Accent.TButton',
                       background=self.colors['accent'],
                       foreground='white',
                       font=('Segoe UI', 14, 'bold'),
                       padding=(30, 18))
        style.map('Accent.TButton',
                 background=[('active', self.colors['accent_hover'])])
        
        style.configure('Secondary.TButton',
                       background=self.colors['bg_input'],
                       foreground=self.colors['text'],
                       font=('Segoe UI', 10),
                       padding=(12, 8))
        
        # Entry style
        style.configure('Modern.TEntry',
                       fieldbackground=self.colors['bg_input'],
                       foreground=self.colors['text'],
                       insertcolor=self.colors['text'])
        
        # Radiobutton and Checkbutton
        style.configure('Card.TRadiobutton',
                       background=self.colors['bg_card'],
                       foreground=self.colors['text'],
                       font=('Segoe UI', 10))
        style.configure('Card.TCheckbutton',
                       background=self.colors['bg_card'],
                       foreground=self.colors['text'],
                       font=('Segoe UI', 10))
        
        # Combobox
        style.configure('TCombobox',
                       fieldbackground=self.colors['bg_input'],
                       background=self.colors['bg_input'],
                       foreground=self.colors['text'])
        
        # Progress bar
        style.configure('green.Horizontal.TProgressbar',
                       background=self.colors['accent'],
                       troughcolor=self.colors['bg_input'])

    def create_card(self, parent, title, row, col, colspan=1, rowspan=1):
        """Create a styled card container."""
        card = ttk.Frame(parent, style='Card.TFrame', padding=15)
        card.grid(row=row, column=col, columnspan=colspan, rowspan=rowspan,
                 sticky='nsew', padx=5, pady=5)
        
        if title:
            title_label = ttk.Label(card, text=title, style='CardTitle.TLabel')
            title_label.pack(anchor='w', pady=(0, 10))
        
        return card

    def create_layout(self):
        """Create the main layout with cards."""
        # Main container with padding
        main = ttk.Frame(self.root, style='Dark.TFrame', padding=15)
        main.pack(fill=tk.BOTH, expand=True)
        
        # Header
        header_frame = ttk.Frame(main, style='Dark.TFrame')
        header_frame.pack(fill=tk.X, pady=(0, 15))
        
        ttk.Label(header_frame, text="🧙‍♂️ Gandalf of Technology Upscaler", 
                 style='Title.TLabel').pack(side=tk.LEFT)
        
        ttk.Label(header_frame, text="Powered by Black Magic ✨ • GPU Enhanced",
                 style='Subtitle.TLabel').pack(side=tk.RIGHT, pady=10)
        
        # Content area - use grid for flexible layout
        content = ttk.Frame(main, style='Dark.TFrame')
        content.pack(fill=tk.BOTH, expand=True)
        content.columnconfigure(0, weight=1)
        content.columnconfigure(1, weight=1)
        
        # === LEFT COLUMN ===
        left_col = ttk.Frame(content, style='Dark.TFrame')
        left_col.grid(row=0, column=0, sticky='nsew', padx=(0, 5))
        
        # Input/Output Card
        io_card = self.create_card(left_col, "📂 Files", 0, 0)
        
        # Input
        ttk.Label(io_card, text="Input Image or Folder:", style='Card.TLabel').pack(anchor='w')
        input_frame = ttk.Frame(io_card, style='Card.TFrame')
        input_frame.pack(fill=tk.X, pady=(5, 10))
        
        self.input_entry = ttk.Entry(input_frame, textvariable=self.input_path, width=35)
        self.input_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        ttk.Button(input_frame, text="📄 File", style='Secondary.TButton',
                  command=self.browse_file).pack(side=tk.LEFT, padx=(5, 2))
        ttk.Button(input_frame, text="📁 Folder", style='Secondary.TButton',
                  command=self.browse_folder).pack(side=tk.LEFT)
        
        # Output
        ttk.Label(io_card, text="Output Location (optional):", style='Card.TLabel').pack(anchor='w', pady=(5, 0))
        output_frame = ttk.Frame(io_card, style='Card.TFrame')
        output_frame.pack(fill=tk.X, pady=(5, 0))
        
        self.output_entry = ttk.Entry(output_frame, textvariable=self.output_path, width=35)
        self.output_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(output_frame, text="Browse", style='Secondary.TButton',
                  command=self.browse_output).pack(side=tk.LEFT, padx=(5, 0))
        
        # Settings Card
        settings_card = self.create_card(left_col, "⚙️ Settings", 1, 0)
        
        # Model selection
        model_frame = ttk.Frame(settings_card, style='Card.TFrame')
        model_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(model_frame, text="AI Model:", style='Card.TLabel').pack(side=tk.LEFT)
        
        self.model_options = {
            "General Photo (Balanced)": "general", 
            "Anime/Drawing (Clean)": "anime", 
            "Cartoon/3D CG (Smooth)": "cartoon", 
            "Portrait (Face Focus)": "portrait",
            "Realistic Photo (Best Detail)": "realistic",
            "Crisp/Sharp (No Blur)": "crisp",
            "Vibrant Color (Punchy)": "vivid"
        }
        
        self.model_combo = ttk.Combobox(model_frame, values=list(self.model_options.keys()), 
                                        state="readonly", width=30)
        self.model_combo.set("General Photo (Balanced)")
        self.model_combo.pack(side=tk.LEFT, padx=(15, 0))
        
        # Link combo to self.model variable
        def on_model_change(event):
            selected_text = self.model_combo.get()
            self.model.set(self.model_options[selected_text])
        self.model_combo.bind("<<ComboboxSelected>>", on_model_change)
        
        # Scale selection
        scale_frame = ttk.Frame(settings_card, style='Card.TFrame')
        scale_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(scale_frame, text="Scale Factor:", style='Card.TLabel').pack(side=tk.LEFT)
        ttk.Radiobutton(scale_frame, text="2x", variable=self.scale, value=2,
                       style='Card.TRadiobutton').pack(side=tk.LEFT, padx=(15, 0))
        ttk.Radiobutton(scale_frame, text="4x (Recommended)", variable=self.scale, value=4,
                       style='Card.TRadiobutton').pack(side=tk.LEFT, padx=(15, 0))
        
        # Format and DPI row
        format_dpi_frame = ttk.Frame(settings_card, style='Card.TFrame')
        format_dpi_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(format_dpi_frame, text="Format:", style='Card.TLabel').pack(side=tk.LEFT)
        format_combo = ttk.Combobox(format_dpi_frame, textvariable=self.output_format,
                                    values=["png", "jpg", "webp"], state="readonly", width=6)
        format_combo.pack(side=tk.LEFT, padx=(10, 20))
        
        ttk.Label(format_dpi_frame, text="DPI:", style='Card.TLabel').pack(side=tk.LEFT)
        dpi_spin = ttk.Spinbox(format_dpi_frame, from_=72, to=600, textvariable=self.dpi, width=5)
        dpi_spin.pack(side=tk.LEFT, padx=(10, 5))
        
        # DPI presets
        for dpi_val in [72, 150, 300]:
            ttk.Button(format_dpi_frame, text=str(dpi_val), width=4,
                      command=lambda d=dpi_val: self.dpi.set(d)).pack(side=tk.LEFT, padx=1)
        
        # Suffix
        suffix_frame = ttk.Frame(settings_card, style='Card.TFrame')
        suffix_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(suffix_frame, text="Filename Suffix:", style='Card.TLabel').pack(side=tk.LEFT)
        ttk.Entry(suffix_frame, textvariable=self.suffix, width=12).pack(side=tk.LEFT, padx=(10, 5))
        ttk.Label(suffix_frame, text="→ image_upscaled_4x.png", 
                 style='Card.TLabel', foreground=self.colors['text_dim']).pack(side=tk.LEFT)
        
        # Checkboxes
        options_frame = ttk.Frame(settings_card, style='Card.TFrame')
        options_frame.pack(fill=tk.X)
        
        ttk.Checkbutton(options_frame, text="👤 Face Enhancement (GFPGAN)",
                       variable=self.face_enhance, style='Card.TCheckbutton').pack(anchor='w')
        ttk.Checkbutton(options_frame, text="📷 Preserve EXIF Metadata",
                       variable=self.preserve_metadata, style='Card.TCheckbutton').pack(anchor='w')
        ttk.Checkbutton(options_frame, text="✨ Remove AI Watermark (Gemini)",
                       variable=self.remove_watermark, style='Card.TCheckbutton').pack(anchor='w')
        
        # Film grain slider
        grain_frame = ttk.Frame(settings_card, style='Card.TFrame')
        grain_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Label(grain_frame, text="🎞️ Film Grain:", style='Card.TLabel').pack(side=tk.LEFT)
        grain_slider = ttk.Scale(grain_frame, from_=0.0, to=1.0, variable=self.grain_strength, orient='horizontal', length=150)
        grain_slider.pack(side=tk.LEFT, padx=(10, 5))
        self.grain_label = ttk.Label(grain_frame, text="1.0", style='Card.TLabel', width=3)
        self.grain_label.pack(side=tk.LEFT)
        
        # Update label when slider changes
        def update_grain_label(*args):
            self.grain_label.config(text=f"{self.grain_strength.get():.1f}")
        self.grain_strength.trace('w', update_grain_label)
        
        # Sharpening slider
        sharpen_frame = ttk.Frame(settings_card, style='Card.TFrame')
        sharpen_frame.pack(fill=tk.X, pady=(5, 0))
        
        ttk.Label(sharpen_frame, text="✨ Sharpening:", style='Card.TLabel').pack(side=tk.LEFT)
        sharpen_slider = ttk.Scale(sharpen_frame, from_=0.0, to=1.0, variable=self.sharpen_amount, orient='horizontal', length=150)
        sharpen_slider.pack(side=tk.LEFT, padx=(10, 5))
        self.sharpen_label = ttk.Label(sharpen_frame, text="0.0", style='Card.TLabel', width=3)
        self.sharpen_label.pack(side=tk.LEFT)
        
        def update_sharpen_label(*args):
            self.sharpen_label.config(text=f"{self.sharpen_amount.get():.1f}")
        self.sharpen_amount.trace('w', update_sharpen_label)
        
        # === RIGHT COLUMN ===
        right_col = ttk.Frame(content, style='Dark.TFrame')
        right_col.grid(row=0, column=1, sticky='nsew', padx=(5, 0))
        
        # Action Card
        action_card = self.create_card(right_col, None, 0, 0)
        
        # Big action buttons
        self.upscale_btn = ttk.Button(action_card, text="🚀 UPSCALE", 
                                      style='Accent.TButton', command=self.start_upscale)
        self.upscale_btn.pack(fill=tk.X, pady=(0, 10))
        
        btn_row = ttk.Frame(action_card, style='Card.TFrame')
        btn_row.pack(fill=tk.X)
        
        self.cancel_btn = ttk.Button(btn_row, text="❌ Cancel", 
                                     command=self.cancel_upscale, state='disabled')
        self.cancel_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 5))
        
        ttk.Button(btn_row, text="📥 Models", 
                  command=self.download_models_thread).pack(side=tk.LEFT, expand=True, fill=tk.X)
        
        # Status Card
        status_card = self.create_card(right_col, "📊 Status", 1, 0)
        
        self.status_label = ttk.Label(status_card, text="Ready to upscale", style='Status.TLabel')
        self.status_label.pack(anchor='w')
        
        self.progress_bar = ttk.Progressbar(status_card, style='green.Horizontal.TProgressbar',
                                            mode='determinate', maximum=100)
        self.progress_bar.pack(fill=tk.X, pady=(10, 5))
        
        self.time_label = ttk.Label(status_card, text="", style='Card.TLabel')
        self.time_label.pack(anchor='w')
        
        self.memory_label = ttk.Label(status_card, text=f"Memory limit: {MAX_MEMORY_GB}GB",
                                      style='Card.TLabel')
        self.memory_label.pack(anchor='w')
        
        # Log Card
        log_card = self.create_card(right_col, "📝 Activity Log", 2, 0)
        
        self.log_text = tk.Text(log_card, height=10, 
                               bg=self.colors['bg_input'], 
                               fg=self.colors['text'],
                               font=('Consolas', 9),
                               insertbackground=self.colors['text'],
                               relief='flat',
                               state='disabled')
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        # Footer
        footer = ttk.Frame(main, style='Dark.TFrame')
        footer.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Label(footer, text="💡 Tip: Drag and drop images directly onto the app (coming soon)",
                 style='Subtitle.TLabel').pack(side=tk.LEFT)

    def browse_file(self):
        """Browse for input file."""
        filetypes = [
            ("Image files", "*.jpg *.jpeg *.png *.webp *.bmp *.tiff"),
            ("All files", "*.*")
        ]
        path = filedialog.askopenfilename(filetypes=filetypes)
        if path:
            self.input_path.set(path)
            p = Path(path)
            self.output_path.set(str(p.parent / "upscaled" / f"{p.stem}{self.suffix.get()}.{self.output_format.get()}"))
            
    def browse_folder(self):
        """Browse for input folder."""
        path = filedialog.askdirectory()
        if path:
            self.input_path.set(path)
            self.output_path.set(str(Path(path) / "upscaled"))
            
    def browse_output(self):
        """Browse for output location."""
        input_p = Path(self.input_path.get()) if self.input_path.get() else None
        
        if input_p and input_p.is_file():
            path = filedialog.asksaveasfilename(
                defaultextension=f".{self.output_format.get()}",
                filetypes=[("Image files", f"*.{self.output_format.get()}")]
            )
        else:
            path = filedialog.askdirectory()
            
        if path:
            self.output_path.set(path)

    def log_message(self, message):
        """Add a message to the activity log."""
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.config(state="normal")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)
        self.log_text.config(state="disabled")
        
    def update_heartbeat(self):
        """Update the heartbeat timestamp."""
        self.last_heartbeat = time.time()
        
    def cancel_upscale(self):
        """Request cancellation of the current upscale operation."""
        self.should_cancel = True
        self.log_message("⚠️ Cancel requested...")
        self.status_label.config(text="Cancelling...")

    def start_upscale(self):
        """Start the upscaling process in a thread."""
        if not self.input_path.get():
            messagebox.showerror("Error", "Please select an input file or folder")
            return
        
        # Reset state
        self.is_processing = True
        self.should_cancel = False
        self.process_start_time = time.time()
        self.update_heartbeat()
        
        self.upscale_btn.state(["disabled"])
        self.cancel_btn.state(["!disabled"])
        self.progress_bar["value"] = 0
        self.status_label.config(text="⏳ Starting...")
        self.time_label.config(text="Initializing...")
        
        self.log_message("🚀 Starting upscale...")
        self.log_message(f"   Model: {self.model.get()} | Scale: {self.scale.get()}x")
        
        thread = threading.Thread(target=self.run_upscale)
        thread.daemon = True
        thread.start()
        
        self.monitor_health()
        
    def monitor_health(self):
        """Monitor processing health."""
        if not self.is_processing:
            return
        
        current_time = time.time()
        elapsed = current_time - self.process_start_time
        self.time_label.config(text=f"⏱️ Elapsed: {format_time(elapsed)}")
        
        try:
            mem_usage = check_memory_usage()
            self.memory_label.config(text=f"🧠 Memory: {mem_usage:.1f}GB / {MAX_MEMORY_GB}GB")
        except:
            pass
        
        if self.is_processing:
            self.root.after(200, self.monitor_health)
    
    def run_upscale(self):
        """Run the upscaling (in thread)."""
        try:
            input_path = Path(self.input_path.get())
            output_path = self.output_path.get() if self.output_path.get() else None
            
            model_map = {
                "general": "RealESRGAN_x4plus",
                "anime": "RealESRGAN_x4plus_anime_6B",
                "cartoon": "4x_foolhardy_Remacri",
                "realistic": "nomos8k_atd_jpg",
                "crisp": "4x-UltraSharp",
                "vivid": "RealESRGAN_x4plus_Vivid",
                "portrait": "RealESRGAN_x4plus"
            }
            model_name = model_map[self.model.get()]
            
            # Auto-enable face enhancement for portrait model
            face_enhance = self.face_enhance.get()
            if self.model.get() == "portrait":
                face_enhance = True
                self.root.after(0, lambda: self.log_message("👤 Portrait mode: Face enhancement auto-enabled"))
            scale = self.scale.get()
            
            self.root.after(0, lambda: self.log_message("📷 Loading AI model..."))
            self.update_heartbeat()
            
            if input_path.is_dir():
                # Batch processing
                extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff', '.tif'}
                images = [f for f in input_path.iterdir() if f.suffix.lower() in extensions]
                
                if not images:
                    self.root.after(0, lambda: self.upscale_error("No images found in folder"))
                    return
                
                self.root.after(0, lambda: self.log_message(f"📁 Found {len(images)} images"))
                
                results = []
                for idx, img_path in enumerate(images):
                    if self.should_cancel:
                        self.root.after(0, lambda r=len(results), t=len(images): 
                                       self.upscale_cancelled(r, t))
                        return
                    
                    self.update_heartbeat()
                    progress = (idx + 1) / len(images) * 100
                    
                    def update_progress(p=progress, i=idx+1, t=len(images), name=img_path.name):
                        self.progress_bar["value"] = p
                        self.status_label.config(text=f"⏳ Processing {i}/{t}: {name}")
                    
                    self.root.after(0, update_progress)
                    
                    def on_progress(msg):
                        self.update_heartbeat()
                        self.root.after(0, lambda m=msg: self.log_message(f"   ↳ {m}"))
                    
                    try:
                        out_dir = Path(output_path) if output_path else input_path / "upscaled"
                        out_path = out_dir / f"{img_path.stem}{self.suffix.get()}_{scale}x.{self.output_format.get()}"
                        
                        result = upscale_image(
                            str(img_path),
                            str(out_path),
                            model_name,
                            scale,
                            face_enhance,
                            self.output_format.get(),
                            progress_callback=on_progress,
                            dpi=self.dpi.get(),
                            suffix=self.suffix.get(),
                            preserve_metadata=self.preserve_metadata.get(),
                            remove_watermark=self.remove_watermark.get(),
                            grain_strength=self.grain_strength.get(),
                            sharpen_amount=self.sharpen_amount.get()
                        )
                        results.append(result)
                        self.root.after(0, lambda n=img_path.name: self.log_message(f"✅ {n}"))
                    except Exception as e:
                        self.root.after(0, lambda n=img_path.name, err=str(e): 
                                       self.log_message(f"❌ {n}: {err}"))
                    
                    self.update_heartbeat()
                
                self.root.after(0, lambda: self.upscale_complete(f"✅ Processed {len(results)} images!"))
            else:
                # Single image
                self.root.after(0, lambda: self.status_label.config(text=f"⏳ Processing: {input_path.name}"))
                
                def on_progress(msg):
                    self.update_heartbeat()
                    self.root.after(0, lambda m=msg: self.log_message(f"   ↳ {m}"))
                    self.root.after(0, lambda m=msg: self.status_label.config(text=f"⏳ {m}"))
                
                result = upscale_image(
                    str(input_path),
                    output_path,
                    model_name,
                    scale,
                    face_enhance,
                    self.output_format.get(),
                    progress_callback=on_progress,
                    dpi=self.dpi.get(),
                    suffix=self.suffix.get(),
                    preserve_metadata=self.preserve_metadata.get(),
                    remove_watermark=self.remove_watermark.get(),
                    grain_strength=self.grain_strength.get(),
                    sharpen_amount=self.sharpen_amount.get()
                )
                self.update_heartbeat()
                
                if result is None:
                    self.root.after(0, lambda: self.upscale_error("Upscaler returned None"))
                else:
                    self.root.after(0, lambda: self.upscale_complete(f"✅ Saved: {Path(result).name}"))
                
        except Exception as e:
            import traceback
            error_msg = f"{type(e).__name__}: {str(e)}"
            print(f"❌ Error: {error_msg}\n{traceback.format_exc()}")
            self.root.after(0, lambda: self.upscale_error(error_msg))
    
    def upscale_cancelled(self, completed, total):
        """Called when upscaling is cancelled."""
        self.is_processing = False
        self.progress_bar["value"] = 0
        self.status_label.config(text=f"⏹️ Cancelled ({completed}/{total} done)")
        self.time_label.config(text="")
        self.upscale_btn.state(["!disabled"])
        self.cancel_btn.state(["disabled"])
        self.log_message(f"⏹️ Cancelled. {completed}/{total} completed.")
        messagebox.showinfo("Cancelled", f"Cancelled.\n{completed}/{total} images completed.")
            
    def upscale_complete(self, message):
        """Called when upscaling is complete."""
        self.is_processing = False
        elapsed = time.time() - self.process_start_time
        
        self.progress_bar["value"] = 100
        self.status_label.config(text=message)
        self.time_label.config(text=f"✅ Done in {format_time(elapsed)}")
        self.upscale_btn.state(["!disabled"])
        self.cancel_btn.state(["disabled"])
        self.log_message(f"🎉 {message}")
        messagebox.showinfo("Success", message)
        
    def upscale_error(self, error):
        """Called when upscaling fails."""
        self.is_processing = False
        
        self.progress_bar["value"] = 0
        self.status_label.config(text="❌ Error occurred")
        self.time_label.config(text="")
        self.upscale_btn.state(["!disabled"])
        self.cancel_btn.state(["disabled"])
        self.log_message(f"❌ Error: {error}")
        messagebox.showerror("Error", f"Failed:\n{error}")
        
    def download_models_thread(self):
        """Download models in a thread."""
        self.progress_bar.start(10)
        self.status_label.config(text="📥 Downloading models...")
        
        def download():
            try:
                download_models()
                self.root.after(0, lambda: self.status_label.config(text="✅ Models ready!"))
                self.root.after(0, lambda: self.log_message("✅ All models downloaded"))
            except Exception as e:
                self.root.after(0, lambda: self.status_label.config(text=f"❌ Download failed"))
                self.root.after(0, lambda: self.log_message(f"❌ Download error: {e}"))
            finally:
                self.root.after(0, self.progress_bar.stop)
                
        thread = threading.Thread(target=download)
        thread.daemon = True
        thread.start()


def main():
    root = tk.Tk()
    app = ModernUpscalerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
