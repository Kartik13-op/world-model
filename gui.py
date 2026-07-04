import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from world_model.config import create_project

class WorldModelGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("AI World Model")
        self.geometry("760x560")
        self.minsize(680, 500)

        self.messages: queue.Queue[str] = queue.Queue()
        self.worker: threading.Thread | None = None

        self.project_var = tk.StringVar(value=str(Path.cwd() / "my_world"))
        self.video_var = tk.StringVar()
        self.size_var = tk.IntVar(value=128)
        self.max_frames_var = tk.StringVar()
        self.epochs_var = tk.IntVar(value=50)
        self.batch_size_var = tk.IntVar(value=2)
        self.lr_var = tk.StringVar(value="0.001")
        self.latent_channels_var = tk.IntVar(value=64)
        self.fps_var = tk.IntVar(value=30)
        self.action_strength_var = tk.StringVar(value="1.0")
        self.latent_damping_var = tk.StringVar(value="1.0")
        self.synthetic_strength_var = tk.StringVar(value="0.12")
        self.device_var = tk.StringVar(value="")

        self._build_ui()
        self.after(100, self._drain_messages)

    def _build_ui(self) -> None:
        root = ttk.Frame(self, padding=14)
        root.pack(fill="both", expand=True)

        title = ttk.Label(root, text="AI World Model", font=("Segoe UI", 18, "bold"))
        title.pack(anchor="w")

        notebook = ttk.Notebook(root)
        notebook.pack(fill="both", expand=True, pady=(12, 8))

        setup = ttk.Frame(notebook, padding=12)
        train = ttk.Frame(notebook, padding=12)
        run = ttk.Frame(notebook, padding=12)
        notebook.add(setup, text="Setup")
        notebook.add(train, text="Train")
        notebook.add(run, text="Play")

        self._path_row(setup, "Project folder", self.project_var, self._choose_project).grid(row=0, column=0, sticky="ew", pady=4)
        self._path_row(setup, "Video file", self.video_var, self._choose_video).grid(row=1, column=0, sticky="ew", pady=4)
        setup.columnconfigure(0, weight=1)

        options = ttk.LabelFrame(setup, text="Preprocess", padding=10)
        options.grid(row=2, column=0, sticky="ew", pady=(12, 4))
        for i in range(4):
            options.columnconfigure(i, weight=1)

        self._number_entry(options, "Frame size", self.size_var, 0, 0)
        self._text_entry(options, "Max frames", self.max_frames_var, 0, 1)

        buttons = ttk.Frame(setup)
        buttons.grid(row=3, column=0, sticky="w", pady=12)
        ttk.Button(buttons, text="Create folders", command=self.create_folders).pack(side="left", padx=(0, 8))
        ttk.Button(buttons, text="Preprocess video", command=self.preprocess).pack(side="left")

        train_opts = ttk.LabelFrame(train, text="Training", padding=10)
        train_opts.pack(fill="x")
        for i in range(4):
            train_opts.columnconfigure(i, weight=1)
        self._number_entry(train_opts, "Epochs", self.epochs_var, 0, 0)
        self._number_entry(train_opts, "Batch size", self.batch_size_var, 0, 1)
        self._text_entry(train_opts, "Learning rate", self.lr_var, 0, 2)
        self._number_entry(train_opts, "Latent channels", self.latent_channels_var, 0, 3)
        self._text_entry(train_opts, "Device", self.device_var, 1, 0)
        self._text_entry(train_opts, "Synthetic control strength", self.synthetic_strength_var, 1, 1)

        ttk.Button(train, text="Train world model", command=self.train_model).pack(anchor="w", pady=12)

        play_opts = ttk.LabelFrame(run, text="Runtime", padding=10)
        play_opts.pack(fill="x")
        play_opts.columnconfigure(0, weight=1)
        play_opts.columnconfigure(1, weight=1)
        play_opts.columnconfigure(2, weight=1)
        play_opts.columnconfigure(3, weight=1)
        self._number_entry(play_opts, "FPS", self.fps_var, 0, 0)
        self._text_entry(play_opts, "Action strength", self.action_strength_var, 0, 1)
        self._text_entry(play_opts, "Latent damping", self.latent_damping_var, 0, 2)
        self._text_entry(play_opts, "Device", self.device_var, 0, 3)

        ttk.Button(run, text="Play", command=self.play).pack(anchor="w", pady=12)

        controls = ttk.Label(run, text="Click the pygame window first. Controls: W/S or Up/Down forward/back, A/D left/right, Left/Right rotate, Esc quits.")
        controls.pack(anchor="w")

        log_frame = ttk.LabelFrame(root, text="Log", padding=8)
        log_frame.pack(fill="both", expand=False)
        self.log = tk.Text(log_frame, height=9, wrap="word", state="disabled")
        self.log.pack(fill="both", expand=True)

    def _path_row(self, parent, label: str, variable: tk.StringVar, command):
        frame = ttk.Frame(parent)
        frame.columnconfigure(1, weight=1)
        ttk.Label(frame, text=label).grid(row=0, column=0, sticky="w", padx=(0, 8))
        ttk.Entry(frame, textvariable=variable).grid(row=0, column=1, sticky="ew", padx=(0, 8))
        ttk.Button(frame, text="Browse", command=command).grid(row=0, column=2)
        return frame

    def _number_entry(self, parent, label: str, variable, row: int, column: int) -> None:
        frame = ttk.Frame(parent)
        frame.grid(row=row, column=column, sticky="ew", padx=4, pady=4)
        ttk.Label(frame, text=label).pack(anchor="w")
        ttk.Entry(frame, textvariable=variable, width=14).pack(fill="x")

    def _text_entry(self, parent, label: str, variable, row: int, column: int) -> None:
        self._number_entry(parent, label, variable, row, column)

    def _choose_project(self) -> None:
        folder = filedialog.askdirectory(initialdir=Path.cwd())
        if folder:
            self.project_var.set(folder)

    def _choose_video(self) -> None:
        filetypes = [("Video files", "*.mp4 *.avi *.mov *.mkv *.webm"), ("All files", "*.*")]
        filename = filedialog.askopenfilename(filetypes=filetypes)
        if filename:
            self.video_var.set(filename)

    def _project(self) -> Path:
        value = self.project_var.get().strip()
        if not value:
            raise ValueError("Choose a project folder.")
        return Path(value)

    def _device(self) -> str | None:
        value = self.device_var.get().strip()
        return value or None

    def _max_frames(self) -> int | None:
        value = self.max_frames_var.get().strip()
        return int(value) if value else None

    def _run_background(self, name: str, fn) -> None:
        if self.worker and self.worker.is_alive():
            messagebox.showinfo("Busy", "A job is already running.")
            return

        def wrapped():
            try:
                self.messages.put(f"{name} started.")
                result = fn()
                if result is not None:
                    self.messages.put(str(result))
                self.messages.put(f"{name} finished.")
            except Exception as exc:
                self.messages.put(f"Error: {exc}")

        self.worker = threading.Thread(target=wrapped, daemon=True)
        self.worker.start()

    def _drain_messages(self) -> None:
        while True:
            try:
                message = self.messages.get_nowait()
            except queue.Empty:
                break
            self._log(message)
        self.after(100, self._drain_messages)

    def _log(self, message: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", message + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def create_folders(self) -> None:
        try:
            paths = create_project(self._project())
            self._log(f"Created project folders at {paths.root.resolve()}")
        except Exception as exc:
            messagebox.showerror("Could not create folders", str(exc))

    def preprocess(self) -> None:
        def job():
            from world_model.preprocess import preprocess_video

            video = self.video_var.get().strip()
            if not video:
                raise ValueError("Choose a video file.")
            paths = preprocess_video(
                project=self._project(),
                video=video,
                size=int(self.size_var.get()),
                max_frames=self._max_frames(),
            )
            return f"Saved processed data in {paths.processed_dir.resolve()}"

        self._run_background("Preprocess", job)

    def train_model(self) -> None:
        def job():
            from world_model.train import train_world_model

            checkpoint = train_world_model(
                project=self._project(),
                epochs=int(self.epochs_var.get()),
                batch_size=int(self.batch_size_var.get()),
                accum_steps=4,
                lr=float(self.lr_var.get()),
                latent_channels=int(self.latent_channels_var.get()),
                device=self._device(),
                synthetic_controls=True,
                synthetic_strength=float(self.synthetic_strength_var.get()),
            )
            return f"Saved checkpoint to {Path(checkpoint).resolve()}"

        self._run_background("Training", job)

    def play(self) -> None:
        def job():
            from world_model.play import play_world_model

            play_world_model(
                project=self._project(),
                fps=int(self.fps_var.get()),
                device=self._device(),
                action_strength=float(self.action_strength_var.get()),
                latent_damping=float(self.latent_damping_var.get()),
            )

        self._run_background("Playback", job)


def main() -> None:
    app = WorldModelGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
