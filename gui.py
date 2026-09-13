import queue
import re
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from world_model.config import create_project

class WorldModelGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("AI World Model")
        self.geometry("940x760")
        self.minsize(780, 620)

        self.messages: queue.Queue[str] = queue.Queue()
        self.worker: threading.Thread | None = None
        self.job_started_at: float | None = None
        self.job_name_var = tk.StringVar(value="Ready")
        self.job_detail_var = tk.StringVar(value="Choose a project and run verification.")
        self.elapsed_var = tk.StringVar(value="Elapsed: 00:00")
        self.progress_var = tk.DoubleVar(value=0.0)
        self.pipeline_status_var = tk.StringVar(value="Checking pipeline…")

        self.project_var = tk.StringVar(value=str(Path.cwd() / "my_world"))
        self.video_var = tk.StringVar()
        self.size_var = tk.IntVar(value=480)
        self.max_frames_var = tk.StringVar()
        self.epochs_var = tk.IntVar(value=50)
        self.batch_size_var = tk.IntVar(value=1)
        self.lr_var = tk.StringVar(value="0.001")
        self.latent_channels_var = tk.IntVar(value=32)
        self.fps_var = tk.IntVar(value=30)
        self.action_strength_var = tk.StringVar(value="1.0")
        self.latent_damping_var = tk.StringVar(value="1.0")
        self.physics_blend_var = tk.StringVar(value="0.85")
        self.start_frame_var = tk.StringVar(value="")
        self.accum_steps_var = tk.IntVar(value=8)
        self.synthetic_strength_var = tk.StringVar(value="0.12")
        self.device_var = tk.StringVar(value="")
        self.save_every_var = tk.IntVar(value=10)
        self.synthetic_controls_var = tk.BooleanVar(value=True)

        # Display-side reconstructor vars
        self.up_epochs_var = tk.IntVar(value=6)
        self.up_lr_var = tk.StringVar(value="0.0003")
        self.up_batch_var = tk.IntVar(value=8)
        self.up_base_ch_var = tk.IntVar(value=32)
        self.up_n_res_var = tk.IntVar(value=4)
        self.up_device_var = tk.StringVar(value="")
        self.up_resume_var = tk.BooleanVar(value=True)
        self.up_patch_size_var = tk.IntVar(value=0)
        self.up_train_size_var = tk.IntVar(value=256)
        self.up_max_samples_var = tk.IntVar(value=512)
        self.up_edge_every_var = tk.IntVar(value=8)
        self.up_compile_var = tk.BooleanVar(value=False)
        self.use_reconstructor_var = tk.BooleanVar(value=True)

        self._build_ui()
        self._refresh_pipeline_status()
        self.after(100, self._drain_messages)

    def _build_ui(self) -> None:
        root = ttk.Frame(self, padding=14)
        root.pack(fill="both", expand=True)

        title = ttk.Label(root, text="AI World Model", font=("Segoe UI", 18, "bold"))
        title.pack(anchor="w")

        status = ttk.Frame(root)
        status.pack(fill="x", pady=(6, 4))
        status.columnconfigure(1, weight=1)
        ttk.Label(status, textvariable=self.job_name_var).grid(row=0, column=0, sticky="w", padx=(0, 12))
        self.progress = ttk.Progressbar(status, variable=self.progress_var, maximum=100, mode="determinate")
        self.progress.grid(row=0, column=1, sticky="ew")
        ttk.Label(status, textvariable=self.elapsed_var, width=16).grid(row=0, column=2, sticky="e", padx=(10, 0))
        ttk.Label(root, textvariable=self.job_detail_var, foreground="gray").pack(anchor="w")

        notebook = ttk.Notebook(root)
        notebook.pack(fill="both", expand=True, pady=(12, 8))

        dashboard = ttk.Frame(notebook, padding=12)
        setup   = ttk.Frame(notebook, padding=12)
        train   = ttk.Frame(notebook, padding=12)
        run     = ttk.Frame(notebook, padding=12)
        upscale = ttk.Frame(notebook, padding=12)
        notebook.add(dashboard, text="Dashboard")
        notebook.add(setup,   text="Setup")
        notebook.add(train,   text="Train")
        notebook.add(run,     text="Play")
        notebook.add(upscale, text="Reconstructor")

        # ── Dashboard tab ────────────────────────────────────────────────────
        dash_title = ttk.Label(dashboard, text="Pipeline overview", font=("Segoe UI", 13, "bold"))
        dash_title.pack(anchor="w")
        ttk.Label(dashboard, textvariable=self.pipeline_status_var, wraplength=760, justify="left").pack(anchor="w", pady=(4, 12))
        self.pipeline_tree = ttk.Treeview(dashboard, columns=("state", "artifact"), show="headings", height=6)
        self.pipeline_tree.heading("state", text="Stage")
        self.pipeline_tree.heading("artifact", text="Status / artifact")
        self.pipeline_tree.column("state", width=180, anchor="w")
        self.pipeline_tree.column("artifact", width=600, anchor="w")
        self.pipeline_tree.pack(fill="x", pady=(0, 12))
        dash_buttons = ttk.Frame(dashboard)
        dash_buttons.pack(anchor="w")
        ttk.Button(dash_buttons, text="Verify pipeline", command=self.verify_pipeline).pack(side="left", padx=(0, 8))
        ttk.Button(dash_buttons, text="Refresh status", command=self._refresh_pipeline_status).pack(side="left")

        # ── Setup tab ─────────────────────────────────────────────────────────
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

        # ── Train tab ─────────────────────────────────────────────────────────
        train_opts = ttk.LabelFrame(train, text="Training", padding=10)
        train_opts.pack(fill="x")
        for i in range(4):
            train_opts.columnconfigure(i, weight=1)
        self._number_entry(train_opts, "Epochs", self.epochs_var, 0, 0)
        self._number_entry(train_opts, "Batch size", self.batch_size_var, 0, 1)
        self._text_entry(train_opts, "Learning rate", self.lr_var, 0, 2)
        self._number_entry(train_opts, "Latent channels", self.latent_channels_var, 0, 3)
        self._number_entry(train_opts, "Accum steps", self.accum_steps_var, 1, 0)
        self._text_entry(train_opts, "Device", self.device_var, 1, 1)
        self._text_entry(train_opts, "Synthetic control strength", self.synthetic_strength_var, 1, 2)
        self._number_entry(train_opts, "Save every N epochs", self.save_every_var, 1, 3)
        ttk.Checkbutton(train_opts, text="Synthetic controls", variable=self.synthetic_controls_var).grid(row=2, column=0, columnspan=2, sticky="w", padx=4, pady=4)

        ttk.Button(train, text="Train world model", command=self.train_model).pack(anchor="w", pady=12)

        # ── Play tab ──────────────────────────────────────────────────────────
        play_opts = ttk.LabelFrame(run, text="Runtime", padding=10)
        play_opts.pack(fill="x")
        for i in range(4):
            play_opts.columnconfigure(i, weight=1)
        self._number_entry(play_opts, "FPS", self.fps_var, 0, 0)
        self._text_entry(play_opts, "Action strength", self.action_strength_var, 0, 1)
        self._text_entry(play_opts, "Latent damping", self.latent_damping_var, 0, 2)
        self._text_entry(play_opts, "Device", self.device_var, 0, 3)
        self._text_entry(play_opts, "Physics blend", self.physics_blend_var, 1, 0)
        self._text_entry(play_opts, "Start frame (blank=random)", self.start_frame_var, 1, 1)

        up_chk = ttk.Checkbutton(
            run,
            text="Use visual reconstructor on idle (display-only restoration)",
            variable=self.use_reconstructor_var,
        )
        up_chk.pack(anchor="w", pady=(8, 0))

        ttk.Button(run, text="Play", command=self.play).pack(anchor="w", pady=12)

        controls = ttk.Label(run, text="Click the pygame window first. Controls: W/S or Up/Down forward/back, A/D left/right, Left/Right rotate, R reset, Esc quits.")
        controls.pack(anchor="w")

        # ── Reconstructor tab ─────────────────────────────────────────────────
        up_info = ttk.Label(
            upscale,
            text=(
                "Train a separate display-side visual reconstructor on the original training video.\n"
                "It learns a per-video visual prior and repairs degraded world-model outputs without feeding them back into the latent state."
            ),
            wraplength=640,
            justify="left",
        )
        up_info.pack(anchor="w", pady=(0, 10))

        up_opts = ttk.LabelFrame(upscale, text="Training options", padding=10)
        up_opts.pack(fill="x")
        for i in range(4):
            up_opts.columnconfigure(i, weight=1)

        self._number_entry(up_opts, "Epochs",          self.up_epochs_var,  0, 0)
        self._text_entry  (up_opts, "Learning rate",   self.up_lr_var,      0, 1)
        self._number_entry(up_opts, "Batch size",      self.up_batch_var,   0, 2)
        self._text_entry  (up_opts, "Device",          self.up_device_var,  0, 3)
        self._number_entry(up_opts, "Base channels",   self.up_base_ch_var, 1, 0)
        self._number_entry(up_opts, "Residual blocks", self.up_n_res_var,   1, 1)
        ttk.Checkbutton(up_opts, text="Resume existing checkpoint", variable=self.up_resume_var).grid(row=1, column=2, columnspan=2, sticky="w", padx=4, pady=4)
        self._number_entry(up_opts, "Patch size (0=full frame)", self.up_patch_size_var, 2, 0)
        self._number_entry(up_opts, "Edge loss interval", self.up_edge_every_var, 2, 1)
        ttk.Checkbutton(up_opts, text="Compile for CUDA", variable=self.up_compile_var).grid(row=2, column=2, columnspan=2, sticky="w", padx=4, pady=4)
        self._number_entry(up_opts, "Fast train size", self.up_train_size_var, 3, 0)
        self._number_entry(up_opts, "Frames per epoch", self.up_max_samples_var, 3, 1)

        btn_frame = ttk.Frame(upscale)
        btn_frame.pack(anchor="w", pady=12)
        ttk.Button(btn_frame, text="Train reconstructor", command=self.train_reconstructor).pack(side="left", padx=(0, 8))

        self._up_status = ttk.Label(upscale, text="", foreground="gray")
        self._up_status.pack(anchor="w")
        self._refresh_reconstructor_status()

        # ── Log ───────────────────────────────────────────────────────────────
        log_frame = ttk.LabelFrame(root, text="Log", padding=8)
        log_frame.pack(fill="both", expand=False)
        self.log = tk.Text(log_frame, height=9, wrap="word", state="disabled")
        self.log.pack(fill="both", expand=True)

    # ── helpers ───────────────────────────────────────────────────────────────

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

    def _up_device(self) -> str | None:
        value = self.up_device_var.get().strip()
        return value or None

    def _max_frames(self) -> int | None:
        value = self.max_frames_var.get().strip()
        return int(value) if value else None

    def _set_job(self, name: str, detail: str, running: bool = True) -> None:
        self.job_name_var.set(name)
        self.job_detail_var.set(detail)
        if running:
            self.progress.configure(mode="indeterminate")
            self.progress.start(12)
        else:
            self.progress.stop()
            self.progress.configure(mode="determinate")

    def _progress_update(self, completed: int, total: int) -> None:
        if total > 0:
            self.progress.stop()
            self.progress.configure(mode="determinate")
            self.progress_var.set(min(100.0, 100.0 * completed / total))
            self.job_detail_var.set(f"Epoch {completed}/{total}")

    def _format_elapsed(self) -> str:
        if self.job_started_at is None:
            return "Elapsed: 00:00"
        seconds = int(time.monotonic() - self.job_started_at)
        return f"Elapsed: {seconds // 60:02d}:{seconds % 60:02d}"

    def _refresh_pipeline_status(self) -> None:
        try:
            from world_model.config import ProjectPaths
            paths = ProjectPaths(self._project())
            rows = []
            frames_ok = paths.frames_file.exists() and paths.actions_file.exists()
            rows.append(("Preprocess", "Ready — frames.npy and actions.npy" if frames_ok else "Missing processed data"))
            rows.append(("World model", f"Ready — {paths.model_file}" if paths.model_file.exists() else "Not trained yet"))
            recon = paths.reconstructor_file
            rows.append(("Reconstructor", f"Ready — {recon}" if recon.exists() else "Optional; not trained yet"))
            rows.append(("Playback", "Ready" if paths.model_file.exists() and frames_ok else "Needs preprocess + world training"))
            for item in self.pipeline_tree.get_children():
                self.pipeline_tree.delete(item)
            for stage, detail in rows:
                self.pipeline_tree.insert("", "end", values=(stage, detail))
            self.pipeline_status_var.set("Pipeline status refreshed. The reconstructor is display-only and can be retrained independently.")
        except Exception as exc:
            self.pipeline_status_var.set(f"Cannot inspect project: {exc}")

    def verify_pipeline(self) -> None:
        def job():
            from world_model.verify import verify_pipeline
            messages = verify_pipeline(self._project(), device=self._device() or "cpu")
            for message in messages:
                self.messages.put(message)
            return "Pipeline verification passed."

        self._run_background("Pipeline verification", job, indeterminate=False)

    def _refresh_reconstructor_status(self) -> None:
        try:
            from world_model.config import ProjectPaths
            paths = ProjectPaths(self._project())
            ckpt = paths.reconstructor_file
            if ckpt.exists():
                self._up_status.config(
                    text=f"✓ Checkpoint found: {ckpt}",
                    foreground="green",
                )
            else:
                self._up_status.config(
                    text="No reconstructor checkpoint yet — train one above.",
                    foreground="gray",
                )
        except Exception:
            self._up_status.config(text="", foreground="gray")
        self.after(3000, self._refresh_reconstructor_status)

    def _run_background(self, name: str, fn, indeterminate: bool = True) -> None:
        if self.worker and self.worker.is_alive():
            messagebox.showinfo("Busy", "A job is already running.")
            return

        self.job_started_at = time.monotonic()
        self.progress_var.set(0.0)
        self._set_job(name, "Working…", running=indeterminate)

        def wrapped():
            try:
                self.messages.put(f"{name} started.")
                result = fn()
                if result is not None:
                    self.messages.put(str(result))
                self.messages.put(f"{name} finished.")
            except ModuleNotFoundError as exc:
                if exc.name == "torch":
                    self.messages.put(
                        "PyTorch is not installed in the Python interpreter "
                        f"running this GUI: {sys.executable}. "
                        "Install the project dependencies with "
                        f'"{sys.executable}" -m pip install -r requirements.txt'
                    )
                else:
                    self.messages.put(f"Missing Python module: {exc.name}")
            except Exception as exc:
                self.messages.put(f"Error: {exc}")
            finally:
                self.messages.put("__JOB_FINISHED__")

        self.worker = threading.Thread(target=wrapped, daemon=True)
        self.worker.start()

    def _drain_messages(self) -> None:
        while True:
            try:
                message = self.messages.get_nowait()
            except queue.Empty:
                break
            if message.startswith("__PROGRESS__:"):
                _, completed, total = message.split(":", 2)
                self._progress_update(int(completed), int(total))
                continue
            if message == "__JOB_FINISHED__":
                self._set_job("Ready", "Job finished. Review the log for details.", running=False)
                self.progress_var.set(100.0)
                self._refresh_pipeline_status()
                continue
            self._log(message)
            self.job_detail_var.set(message)
        self.elapsed_var.set(self._format_elapsed())
        self.after(100, self._drain_messages)

    def _log(self, message: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", message + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    # ── actions ───────────────────────────────────────────────────────────────

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
                accum_steps=int(self.accum_steps_var.get()),
                lr=float(self.lr_var.get()),
                latent_channels=int(self.latent_channels_var.get()),
                device=self._device(),
                save_every=int(self.save_every_var.get()),
                synthetic_controls=bool(self.synthetic_controls_var.get()),
                synthetic_strength=float(self.synthetic_strength_var.get()),
                grad_checkpoint=True,
                progress_fn=lambda epoch, total: self.messages.put(f"__PROGRESS__:{epoch}:{total}"),
            )
            return f"Saved checkpoint to {Path(checkpoint).resolve()}"

        self._run_background("Training", job)

    def train_reconstructor(self) -> None:
        def job():
            from world_model.reconstructor import train_reconstructor

            ckpt = train_reconstructor(
                project=self._project(),
                epochs=int(self.up_epochs_var.get()),
                batch_size=int(self.up_batch_var.get()),
                lr=float(self.up_lr_var.get()),
                base_ch=int(self.up_base_ch_var.get()),
                n_res=int(self.up_n_res_var.get()),
                patch_size=(int(self.up_patch_size_var.get()) or None),
                device=self._up_device(),
                log_fn=self.messages.put,
                resume=bool(self.up_resume_var.get()),
                compile_model=bool(self.up_compile_var.get()),
                edge_loss_every=int(self.up_edge_every_var.get()),
                progress_fn=lambda epoch, total: self.messages.put(f"__PROGRESS__:{epoch}:{total}"),
                train_size=(int(self.up_train_size_var.get()) or None),
                max_samples=(int(self.up_max_samples_var.get()) or None),
            )
            return f"Reconstructor saved → {Path(ckpt).resolve()}"

        self._run_background("Reconstructor training", job)

    def play(self) -> None:
        def job():
            from world_model.play import play_world_model

            start_raw = self.start_frame_var.get().strip()
            start_frame = int(start_raw) if start_raw else -1

            play_world_model(
                project=self._project(),
                fps=int(self.fps_var.get()),
                device=self._device(),
                action_strength=float(self.action_strength_var.get()),
                latent_damping=float(self.latent_damping_var.get()),
                start_frame=start_frame,
                physics_blend=float(self.physics_blend_var.get()),
                use_reconstructor=bool(self.use_reconstructor_var.get()),
            )

        self._run_background("Playback", job)


def main() -> None:
    app = WorldModelGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
