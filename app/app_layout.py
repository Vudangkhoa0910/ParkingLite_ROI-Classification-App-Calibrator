import tkinter as tk
from tkinter import ttk
import classification

class LayoutMixin:
    def _build_ui(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TFrame', background='#1e1e2e')
        style.configure('TLabel', background='#1e1e2e', foreground='#cdd6f4',
                        font=('Segoe UI', 10))
        style.configure('TButton', font=('Segoe UI', 9), padding=4)
        style.configure('Header.TLabel', font=('Segoe UI', 11, 'bold'),
                        foreground='#89b4fa')
        style.configure('Mode.TButton', font=('Segoe UI', 10, 'bold'), padding=6,
                        background='#2e2e3f', foreground='#cdd6f4')
        style.configure('ActiveMode.TButton', font=('Segoe UI', 10, 'bold'), padding=6,
                        background='#89b4fa', foreground='#1e1e2e')
        style.map('ActiveMode.TButton', background=[('active', '#7fa2ee')])
        style.configure('Action.TButton', font=('Segoe UI', 10), padding=5)
        style.configure('Accent.TButton', font=('Segoe UI', 10, 'bold'), padding=6)

        top = ttk.Frame(self, style='TFrame')
        top.pack(fill=tk.X, padx=5, pady=(5, 0))

        ttk.Button(top, text="Open Image", command=self._open_image,
                   style='Action.TButton').pack(side=tk.LEFT, padx=2)
        ttk.Button(top, text="Load Config", command=self._open_config,
                   style='Action.TButton').pack(side=tk.LEFT, padx=2)

        ttk.Separator(top, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y,
                                                      padx=8, pady=2)

        self.btn_draw = ttk.Button(top, text="Draw Slot",
                                    command=lambda: self._set_mode('draw'),
                                    style='Mode.TButton')
        self.btn_draw.pack(side=tk.LEFT, padx=2)
        self.btn_edit = ttk.Button(top, text="Edit / Move",
                                    command=lambda: self._set_mode('edit'),
                                    style='Mode.TButton')
        self.btn_edit.pack(side=tk.LEFT, padx=2)
        self.btn_select = ttk.Button(top, text="Select Grid",
                                      command=lambda: self._set_mode('select'),
                                      style='Mode.TButton')
        self.btn_select.pack(side=tk.LEFT, padx=2)
        self.btn_grid = ttk.Button(top, text="Auto Grid",
                                    command=lambda: self._set_mode('grid'),
                                    style='Mode.TButton')
        self.btn_grid.pack(side=tk.LEFT, padx=2)
        self.btn_tile = ttk.Button(top, text="Tile Box",
                                   command=lambda: self._set_mode('tile'),
                                   style='Mode.TButton')
        self.btn_tile.pack(side=tk.LEFT, padx=2)
        ttk.Separator(top, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y,
                                                      padx=8, pady=2)

        ttk.Button(top, text="Zoom +", command=self._zoom_in,
                   style='TButton').pack(side=tk.LEFT, padx=1)
        ttk.Button(top, text="Zoom -", command=self._zoom_out,
                   style='TButton').pack(side=tk.LEFT, padx=1)
        ttk.Button(top, text="Fit", command=self._zoom_fit,
                   style='TButton').pack(side=tk.LEFT, padx=1)

        ttk.Separator(top, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y,
                                                      padx=8, pady=2)

        ttk.Button(top, text="Undo", command=self._undo,
                   style='TButton').pack(side=tk.LEFT, padx=2)
        ttk.Button(top, text="Redo", command=self._redo,
               style='TButton').pack(side=tk.LEFT, padx=2)

        main = ttk.Frame(self, style='TFrame')
        main.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        canvas_frame = ttk.Frame(main, style='TFrame')
        canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(canvas_frame, bg='#11111b', highlightthickness=0,
                                cursor='crosshair')
        self.canvas.pack(fill=tk.BOTH, expand=True)

        right = ttk.Frame(main, style='TFrame', width=300)
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))
        right.pack_propagate(False)

        self._build_right_panel(right)

        self.status_var = tk.StringVar(value="Open an image to start.")
        status = ttk.Label(self, textvariable=self.status_var,
                           font=('Consolas', 9), foreground='#a6adc8')
        status.pack(fill=tk.X, padx=5, pady=(0, 3))

    def _build_right_panel(self, parent):
        ttk.Label(parent, text="ACTIONS", style='Header.TLabel').pack(
            anchor=tk.W, pady=(5, 3))

        btn_frame = ttk.Frame(parent, style='TFrame')
        btn_frame.pack(fill=tk.X, pady=2)

        ttk.Button(btn_frame, text="Auto Number (N)",
                   command=self._auto_number, style='Action.TButton').pack(
                       fill=tk.X, pady=1)
        ttk.Button(btn_frame, text="Delete Selected (Del)",
                   command=self._delete_selected, style='Action.TButton').pack(
                       fill=tk.X, pady=1)
        ttk.Button(btn_frame, text="Duplicate (D)",
                   command=self._duplicate, style='Action.TButton').pack(
                       fill=tk.X, pady=1)
        ttk.Button(btn_frame, text="Clear All",
                   command=self._clear_all, style='Action.TButton').pack(
                       fill=tk.X, pady=1)

        ttk.Separator(parent, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8)

        ttk.Label(parent, text="MODE SETTINGS", style='Header.TLabel').pack(
            anchor=tk.W, pady=(0, 3))

        self.grid_settings_frame = ttk.Frame(parent, style='TFrame')
        ttk.Label(self.grid_settings_frame, text="Auto Grid Rows:").pack(side=tk.LEFT)
        ttk.Spinbox(self.grid_settings_frame, from_=1, to=8, width=4,
                    textvariable=self.grid_rows).pack(side=tk.LEFT, padx=4)
        ttk.Label(self.grid_settings_frame, text="Cols:").pack(side=tk.LEFT, padx=(8, 0))
        ttk.Spinbox(self.grid_settings_frame, from_=1, to=16, width=4,
                    textvariable=self.grid_cols).pack(side=tk.LEFT, padx=4)
        self.grid_settings_frame.pack(fill=tk.X, pady=2)

        self.tile_settings_frame = ttk.Frame(parent, style='TFrame')
        ttk.Label(self.tile_settings_frame, text="Tile Rows:").pack(side=tk.LEFT)
        ttk.Spinbox(self.tile_settings_frame, from_=1, to=8, width=4,
                    textvariable=self.tile_rows_var).pack(side=tk.LEFT, padx=4)
        ttk.Label(self.tile_settings_frame, text="Cols:").pack(side=tk.LEFT, padx=(8, 0))
        ttk.Spinbox(self.tile_settings_frame, from_=1, to=16, width=4,
                    textvariable=self.tile_cols_var).pack(side=tk.LEFT, padx=4)
        self.tile_settings_frame.pack(fill=tk.X, pady=2)

        ttk.Separator(parent, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8)

        ttk.Label(parent, text="CLASSIFY", style='Header.TLabel').pack(
            anchor=tk.W, pady=(0, 3))

        ref_f = ttk.Frame(parent, style='TFrame')
        ref_f.pack(fill=tk.X, pady=1)
        ttk.Button(ref_f, text="Set Reference Image",
                   command=self._set_ref).pack(fill=tk.X)

        test_f = ttk.Frame(parent, style='TFrame')
        test_f.pack(fill=tk.X, pady=1)
        ttk.Button(test_f, text="Set Test Image",
                   command=self._set_test).pack(fill=tk.X)

        self.ref_label = ttk.Label(parent, text="Ref: (none)",
                                    font=('Consolas', 8))
        self.ref_label.pack(anchor=tk.W)
        self.test_label = ttk.Label(parent, text="Test: (none)",
                                     font=('Consolas', 8))
        self.test_label.pack(anchor=tk.W)

        self.classifier_key_to_label = {key: label for key, label in classification.CLASSIFIER_OPTIONS}
        self.classifier_label_to_key = {label: key for key, label in classification.CLASSIFIER_OPTIONS}
        default_label = self.classifier_key_to_label.get(
            classification.DEFAULT_CLASSIFIER,
            classification.CLASSIFIER_OPTIONS[0][1],
        )
        if not hasattr(self, 'classify_method_var'):
            self.classify_method_var = tk.StringVar(value=default_label)
        elif not self.classify_method_var.get():
            self.classify_method_var.set(default_label)

        ttk.Label(parent, text="ROI Method:", font=('Segoe UI', 9, 'bold')).pack(anchor=tk.W, pady=(6, 1))
        self.classifier_combo = ttk.Combobox(
            parent,
            state='readonly',
            textvariable=self.classify_method_var,
            values=[label for _, label in classification.CLASSIFIER_OPTIONS],
        )
        self.classifier_combo.pack(fill=tk.X, pady=(0, 3))
        self.classifier_combo.bind('<<ComboboxSelected>>', self._on_classifier_changed)

        ttk.Button(parent, text="Run ROI Classification",
                   command=self._classify, style='Accent.TButton').pack(
                       fill=tk.X, pady=2)

        ttk.Button(parent, text="Quick: Load Test & Classify",
                   command=self._quick_classify, style='Accent.TButton').pack(
                       fill=tk.X, pady=2)

        view_f = ttk.Frame(parent, style='TFrame')
        view_f.pack(fill=tk.X, pady=2)
        ttk.Button(view_f, text="View: Reference",
                   command=self._show_ref_image).pack(side=tk.LEFT, expand=True,
                                                       fill=tk.X, padx=(0, 2))
        ttk.Button(view_f, text="View: Test",
                   command=self._show_test_image).pack(side=tk.LEFT, expand=True,
                                                        fill=tk.X, padx=(2, 0))

        ttk.Separator(parent, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8)

        ttk.Label(parent, text="EXPORT", style='Header.TLabel').pack(
            anchor=tk.W, pady=(0, 3))
        ttk.Button(parent, text="Save Config (JSON + C + Python)",
                   command=self._save, style='Accent.TButton').pack(
                       fill=tk.X, pady=2)

        ttk.Separator(parent, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8)

        ttk.Label(parent, text="SLOT LIST", style='Header.TLabel').pack(
            anchor=tk.W, pady=(0, 3))

        list_frame = ttk.Frame(parent, style='TFrame')
        list_frame.pack(fill=tk.BOTH, expand=True)

        self.slot_list = tk.Listbox(
            list_frame, bg='#181825', fg='#cdd6f4', selectbackground='#45475a',
            font=('Consolas', 9), borderwidth=0, highlightthickness=0)
        scrollbar = ttk.Scrollbar(list_frame, command=self.slot_list.yview)
        self.slot_list.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.slot_list.pack(fill=tk.BOTH, expand=True)
        self.slot_list.bind('<<ListboxSelect>>', self._on_list_select)

    def _update_mode_settings_panel(self):
        if not hasattr(self, 'grid_settings_frame') or not hasattr(self, 'tile_settings_frame'):
            return
        if not self.grid_settings_frame.winfo_ismapped():
            self.grid_settings_frame.pack(fill=tk.X, pady=2)
        if not self.tile_settings_frame.winfo_ismapped():
            self.tile_settings_frame.pack(fill=tk.X, pady=2)
