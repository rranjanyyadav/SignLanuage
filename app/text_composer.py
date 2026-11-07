"""ISL Alphabet Live Text Composer.

A desktop application that converts recognized Indian Sign Language (ISL) alphabet
signs into composed text with stable-letter confirmation, hold-to-confirm feedback,
and interactive text editing controls.

Reuses the production ResNet model (models/isl_resnet.pth) and MediaPipe landmarker
pipeline with guarded N/M landmark disambiguation.
"""
import argparse
from dataclasses import dataclass
from collections import Counter, deque
from pathlib import Path
import queue
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox

import cv2
import mediapipe as mp
import numpy as np
import torch
from PIL import Image, ImageTk

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR))

from data_preparation.prepare_cropped_dataset import compute_hand_box_and_mask
try:
    from app.model_config import resolve_checkpoint, get_active_model_info
except ImportError:
    from model_config import resolve_checkpoint, get_active_model_info

from app.live_image_inference import (
    build_model,
    create_transform,
    smooth_box,
    stabilize_two_hand_box,
    verify_hand_count_consistency,
    correct_n_m_prediction,
    ISL_SINGLE_HAND_CLASSES,
    ISL_TWO_HAND_CLASSES,
    MODEL_DIR,
    LANDMARKER_PATH,
)


@dataclass
class FramePrediction:
    """Diagnostic and prediction output from a single camera frame."""
    predicted_char: str | None
    raw_char: str | None
    confidence: float
    is_valid: bool
    num_hands: int
    reason: str
    display_text: str
    box: tuple | None
    preview_crop: np.ndarray | None
    nm_info: dict


class TextComposerState:
    """Deterministic state machine managing stable-letter hold confirmation and text editing.

    Rules:
    1. A sign must be valid (meets confidence and consistency) and held steadily.
    2. The candidate letter must be held continuously for `hold_duration` seconds (default 0.8s).
    3. Once confirmed, the letter is appended exactly once, and entry is DISARMED to prevent
       accidental repetition while the user continues holding the sign.
    4. To enter the same letter again, the user must RELEASE (no hand / invalid sign) or switch
       to a different letter before re-signing.
    5. Unstable predictions, confidence drops, or invalid hand counts reset the hold timer.
    6. SPACE, UNDO, and CLEAR controls operate directly on the composed string.
    7. Supports deterministic time injection via `clock_func` or explicit `timestamp` arguments.
    """

    def __init__(
        self,
        hold_duration: float = 0.8,
        min_confidence: float = 0.50,
        clock_func=time.time,
    ):
        self.hold_duration = max(0.1, float(hold_duration))
        self.min_confidence = float(min_confidence)
        self._clock = clock_func

        self.composed_text = ""
        self.current_candidate = None
        self.hold_start_time = None
        self.last_confirmed_letter = None
        self.is_armed = True
        self.last_event_note = ""

    def update(
        self,
        predicted_char: str | None,
        is_valid: bool = True,
        confidence: float = 1.0,
        timestamp: float | None = None,
    ) -> bool:
        """Process a frame's prediction.

        Returns True if a letter was confirmed and appended during this update, False otherwise.
        """
        now = timestamp if timestamp is not None else self._clock()
        char = str(predicted_char).upper() if predicted_char else None

        # 1. Invalid, uncertain, or empty prediction -> Reset hold & re-arm
        if not char or not is_valid or confidence < self.min_confidence:
            self.current_candidate = None
            self.hold_start_time = None
            # Releasing re-arms the composer, allowing the same letter to be typed again
            self.is_armed = True
            return False

        # 2. Candidate changed -> restart timer
        if char != self.current_candidate:
            self.current_candidate = char
            self.hold_start_time = now
            # If switching to a completely different letter, re-arm immediately
            if char != self.last_confirmed_letter:
                self.is_armed = True
            return False

        # 3. Holding current candidate: check armed state
        if not self.is_armed:
            # Already confirmed during this hold session; waiting for release or change
            return False

        # 4. Check if hold duration threshold is reached
        elapsed = now - (self.hold_start_time if self.hold_start_time is not None else now)
        if (elapsed + 1e-5) >= self.hold_duration:
            self.composed_text += char
            self.last_confirmed_letter = char
            self.is_armed = False  # Disarm to prevent repeated entry while holding
            self.hold_start_time = None
            self.last_event_note = f"confirmed_{char}"
            return True

        return False

    def get_hold_progress(self, timestamp: float | None = None) -> float:
        """Return confirmation hold progress fraction [0.0, 1.0]."""
        if not self.is_armed or not self.current_candidate or self.hold_start_time is None:
            return 0.0
        now = timestamp if timestamp is not None else self._clock()
        elapsed = max(0.0, now - self.hold_start_time)
        return min(1.0, elapsed / self.hold_duration)

    def add_space(self) -> str:
        """Append space to composed text."""
        self.composed_text += " "
        self.is_armed = True
        self.last_event_note = "space"
        return self.composed_text

    def undo(self) -> str:
        """Delete the last character."""
        if self.composed_text:
            self.composed_text = self.composed_text[:-1]
        self.is_armed = True
        self.last_event_note = "undo"
        return self.composed_text

    def clear(self) -> str:
        """Clear all composed text."""
        self.composed_text = ""
        self.last_confirmed_letter = None
        self.is_armed = True
        self.last_event_note = "clear"
        return self.composed_text

    def get_text(self) -> str:
        return self.composed_text


class SignPredictor:
    """Encapsulates the complete live inference pipeline for one camera session.

    Reuses:
    - Production ResNet checkpoint (models/isl_resnet.pth).
    - MediaPipe HandLandmarker detector.
    - Two-hand spatial bounding box & legacy silhouette mask.
    - Temporal EMA probability smoothing & majority-voting history.
    - Guarded N/M landmark disambiguation.
    - Hand count consistency verification.
    """

    def __init__(
        self,
        checkpoint_path: Path = None,
        landmarker_path: Path = LANDMARKER_PATH,
        min_confidence: float = 0.50,
        min_consensus: float = 0.40,
        history_size: int = 5,
        prob_ema_alpha: float = 0.60,
        padding: float = 0.25,
    ):
        checkpoint_path, self.model_display_name = resolve_checkpoint(checkpoint_arg=checkpoint_path)
        self.checkpoint_path = checkpoint_path
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
        if not landmarker_path.is_file():
            raise FileNotFoundError(f"MediaPipe task file not found: {landmarker_path}")

        self.checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        self.classes = self.checkpoint["classes"]
        self.model_type = self.checkpoint.get("model_type", "resnet")
        self.is_gray = self.checkpoint.get("grayscale", False)
        self.mask_mode = self.checkpoint.get("mask_mode", "legacy")
        if self.mask_mode not in ("legacy", "fine"):
            self.mask_mode = "legacy"
        self.should_mask = True  # Production ResNet was trained on dataset_ISL_masked

        self.model = build_model(self.model_type, self.checkpoint)
        self.transform = create_transform(self.checkpoint, force_grayscale=self.is_gray)

        self.min_confidence = min_confidence
        self.min_consensus = min_consensus
        self.prob_ema_alpha = prob_ema_alpha
        self.padding = padding

        # State tracking
        self.history = deque(maxlen=history_size)
        self.smoothed_probs = None
        self.smooth_box_coords = None
        self.last_two_hand_box = None
        self.two_hand_hold_counter = 0
        self.last_timestamp_ms = 0

        # MediaPipe Landmarker
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(landmarker_path)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.4,
            min_hand_presence_confidence=0.4,
            min_tracking_confidence=0.4,
        )
        self.landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)

    def process_frame(self, frame: np.ndarray, mirror: bool = False) -> FramePrediction:
        """Run complete inference pipeline on a single frame."""
        if mirror:
            frame = cv2.flip(frame, 1)

        height, width = frame.shape[:2]
        timestamp_ms = int(time.time() * 1000)
        if timestamp_ms <= self.last_timestamp_ms:
            timestamp_ms = self.last_timestamp_ms + 1
        self.last_timestamp_ms = timestamp_ms

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect_for_video(mp_img, timestamp_ms)

        num_hands = len(result.hand_landmarks) if result.hand_landmarks else 0
        raw_box, crop_mask = None, None

        if num_hands > 0:
            raw_box, crop_mask = compute_hand_box_and_mask(
                result.hand_landmarks, width, height, padding=self.padding, square=True,
                mask_mode=self.mask_mode,
            )
            effective_box, self.last_two_hand_box, self.two_hand_hold_counter = stabilize_two_hand_box(
                raw_box, num_hands, self.last_two_hand_box, self.two_hand_hold_counter, max_hold=4
            )
        else:
            effective_box = None
            self.last_two_hand_box = None
            self.two_hand_hold_counter = 0

        if effective_box is None:
            self.history.clear()
            self.smoothed_probs = None
            self.smooth_box_coords = None
            return FramePrediction(
                predicted_char=None,
                raw_char=None,
                confidence=0.0,
                is_valid=False,
                num_hands=0,
                reason="No hand detected",
                display_text="No hand detected",
                box=None,
                preview_crop=None,
                nm_info={},
            )

        self.smooth_box_coords = smooth_box(effective_box, self.smooth_box_coords, alpha=0.45)
        left, top, right, bottom = self.smooth_box_coords
        crop = frame[top:bottom, left:right]

        if crop.size == 0:
            return FramePrediction(
                predicted_char=None,
                raw_char=None,
                confidence=0.0,
                is_valid=False,
                num_hands=num_hands,
                reason="Crop error",
                display_text="Crop error",
                box=None,
                preview_crop=None,
                nm_info={},
            )

        if self.should_mask and crop_mask is not None:
            if crop_mask.shape != crop.shape[:2]:
                crop_mask = cv2.resize(crop_mask, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_NEAREST)
            crop = cv2.bitwise_and(crop, crop, mask=crop_mask)

        # Model forward pass
        inputs = self.transform(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).unsqueeze(0)
        with torch.inference_mode():
            probs = torch.softmax(self.model(inputs), dim=1)[0]

        if self.smoothed_probs is None:
            self.smoothed_probs = probs
        else:
            self.smoothed_probs = self.prob_ema_alpha * probs + (1 - self.prob_ema_alpha) * self.smoothed_probs

        confidence, index = self.smoothed_probs.max(dim=0)
        self.history.append(int(index))
        stable_index, votes = Counter(self.history).most_common(1)[0]
        vote_frac = votes / len(self.history)
        conf_val = float(confidence)

        raw_char = self.classes[stable_index].lower()

        # Guarded N vs M landmark correction
        effective_char, nm_info = correct_n_m_prediction(
            raw_char,
            result.hand_landmarks if result else None,
        )

        is_consistent, reason = verify_hand_count_consistency(effective_char, num_hands)
        hand_desc = "2-Hand" if num_hands > 1 else "1-Hand"
        display_char = effective_char.upper()

        is_valid = (
            is_consistent
            and conf_val >= self.min_confidence
            and vote_frac >= self.min_consensus
        )

        if not is_consistent:
            status_text = f"Uncertain: {display_char} ({reason})"
        elif is_valid:
            status_text = f"{display_char} ({conf_val * 100:.1f}%) [{hand_desc}]"
        else:
            status_text = f"Uncertain: {display_char} ({conf_val * 100:.1f}%)"

        preview_crop = cv2.resize(crop, (120, 120))

        return FramePrediction(
            predicted_char=display_char,
            raw_char=raw_char.upper(),
            confidence=conf_val,
            is_valid=is_valid,
            num_hands=num_hands,
            reason=reason if not is_consistent else ("valid" if is_valid else "low_consensus_or_conf"),
            display_text=status_text,
            box=self.smooth_box_coords,
            preview_crop=preview_crop,
            nm_info=nm_info,
        )

    def close(self):
        """Release MediaPipe landmarker resources."""
        if hasattr(self, "landmarker") and self.landmarker is not None:
            self.landmarker.close()


class TextComposerApp:
    """Tkinter desktop application for live ISL text composition."""

    def __init__(
        self,
        root: tk.Tk,
        checkpoint_path: Path = None,
        camera_id: int = 0,
        hold_duration: float = 0.8,
        mirror: bool = False,
    ):
        self.root = root
        self.root.title("ISL Alphabet Live Text Composer")
        self.root.geometry("1080x720")
        self.root.minsize(960, 640)
        self.root.configure(bg="#0f172a")

        self.camera_id = camera_id
        self.mirror = mirror
        self.running = True

        # Pipeline & state engines
        self.predictor = SignPredictor(checkpoint_path=checkpoint_path)
        self.composer = TextComposerState(hold_duration=hold_duration)

        # Background thread queue for smooth, non-blocking UI updates
        self.frame_queue = queue.Queue(maxsize=1)

        # Setup GUI Components
        self._build_gui()

        # Camera Capture Thread
        self.cap = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap.release()
            self.cap = cv2.VideoCapture(self.camera_id)

        if not self.cap.isOpened():
            messagebox.showerror(
                "Camera Error",
                f"Could not open camera {self.camera_id}.\nPlease ensure it is connected and not in use by another application.",
            )
            self.on_close()
            return

        self.capture_thread = threading.Thread(target=self._camera_worker, daemon=True)
        self.capture_thread.start()

        # Periodic UI update tick
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.after(20, self._ui_tick)

    def _build_gui(self):
        # 1. Top Header Banner
        header = tk.Frame(self.root, bg="#1e293b", height=50)
        header.pack(fill=tk.X, side=tk.TOP)

        title = tk.Label(
            header,
            text="ISL ALPHABET LIVE TEXT COMPOSER",
            font=("Segoe UI", 14, "bold"),
            fg="#38bdf8",
            bg="#1e293b",
            padx=16,
            pady=8,
        )
        title.pack(side=tk.LEFT)

        badge_text = f"Model: {getattr(self.predictor, 'model_display_name', self.predictor.checkpoint_path.name)} | Hold: {self.composer.hold_duration:.1f}s"
        model_badge = tk.Label(
            header,
            text=badge_text,
            font=("Segoe UI", 9),
            fg="#94a3b8",
            bg="#1e293b",
            padx=16,
        )
        model_badge.pack(side=tk.RIGHT)

        # 2. Main Body (Horizontal Split: Left=Camera, Right=Composer)
        main_body = tk.Frame(self.root, bg="#0f172a")
        main_body.pack(fill=tk.BOTH, expand=True, padx=16, pady=12)

        # Left Column: Video Feed & Prediction Feedback
        left_col = tk.Frame(main_body, bg="#1e293b", width=540)
        left_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(0, 10))

        # Camera Display Canvas
        self.video_canvas = tk.Label(left_col, bg="#000000", width=520, height=390)
        self.video_canvas.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # Recognition Status Strip
        stat_frame = tk.Frame(left_col, bg="#1e293b")
        stat_frame.pack(fill=tk.X, padx=10, pady=(0, 6))

        self.candidate_label = tk.Label(
            stat_frame,
            text="—",
            font=("Segoe UI", 32, "bold"),
            fg="#38bdf8",
            bg="#1e293b",
            width=3,
        )
        self.candidate_label.pack(side=tk.LEFT, padx=4)

        info_box = tk.Frame(stat_frame, bg="#1e293b")
        info_box.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)

        self.status_detail_label = tk.Label(
            info_box,
            text="Position hands in camera view",
            font=("Segoe UI", 11, "bold"),
            fg="#f8fafc",
            bg="#1e293b",
            anchor="w",
        )
        self.status_detail_label.pack(fill=tk.X)

        self.progress_text = tk.Label(
            info_box,
            text="Stability: Hold sign steady",
            font=("Segoe UI", 9),
            fg="#94a3b8",
            bg="#1e293b",
            anchor="w",
        )
        self.progress_text.pack(fill=tk.X)

        # Visual Confirmation Hold Progress Bar
        self.progress_bar = ttk.Progressbar(left_col, orient="horizontal", mode="determinate", length=400)
        self.progress_bar.pack(fill=tk.X, padx=12, pady=(0, 10))

        # Right Column: Text Composition Area, Instructions, and Controls
        right_col = tk.Frame(main_body, bg="#1e293b")
        right_col.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Instruction Card
        inst_frame = tk.Frame(right_col, bg="#0f172a", relief=tk.RIDGE, bd=1)
        inst_frame.pack(fill=tk.X, padx=12, pady=10)

        inst_title = tk.Label(
            inst_frame,
            text="HOW TO COMPOSE:",
            font=("Segoe UI", 9, "bold"),
            fg="#38bdf8",
            bg="#0f172a",
            anchor="w",
            padx=8,
            pady=4,
        )
        inst_title.pack(fill=tk.X)

        inst_body = (
            "1. Form an ISL sign and hold steadily for ~0.8 seconds.\n"
            "2. When the green progress bar fills, the letter is added once.\n"
            "3. Release hands briefly before repeating the same letter.\n"
            "4. Use SPACE, UNDO or CLEAR buttons below as needed."
        )
        inst_text = tk.Label(
            inst_frame,
            text=inst_body,
            font=("Segoe UI", 8),
            fg="#cbd5e1",
            bg="#0f172a",
            justify=tk.LEFT,
            anchor="w",
            padx=8,
            pady=4,
        )
        inst_text.pack(fill=tk.X)

        # Text Composition Title
        comp_title = tk.Label(
            right_col,
            text="COMPOSED TEXT",
            font=("Segoe UI", 10, "bold"),
            fg="#94a3b8",
            bg="#1e293b",
            anchor="w",
            padx=12,
        )
        comp_title.pack(fill=tk.X, pady=(4, 0))

        # Multiline Text Display Box
        self.text_display = tk.Text(
            right_col,
            font=("Consolas", 22, "bold"),
            bg="#0f172a",
            fg="#38bdf8",
            insertbackground="#38bdf8",
            wrap=tk.WORD,
            height=6,
            relief=tk.FLAT,
            padx=14,
            pady=12,
        )
        self.text_display.pack(fill=tk.BOTH, expand=True, padx=12, pady=6)

        # Control Buttons Strip
        btn_frame = tk.Frame(right_col, bg="#1e293b")
        btn_frame.pack(fill=tk.X, padx=12, pady=10)

        space_btn = tk.Button(
            btn_frame,
            text="SPACE  [ ␣ ]",
            font=("Segoe UI", 10, "bold"),
            bg="#2563eb",
            fg="#ffffff",
            activebackground="#1d4ed8",
            relief=tk.FLAT,
            padx=14,
            pady=8,
            command=self._on_space,
        )
        space_btn.pack(side=tk.LEFT, padx=(0, 6))

        undo_btn = tk.Button(
            btn_frame,
            text="UNDO  [ ⌫ ]",
            font=("Segoe UI", 10, "bold"),
            bg="#475569",
            fg="#ffffff",
            activebackground="#334155",
            relief=tk.FLAT,
            padx=14,
            pady=8,
            command=self._on_undo,
        )
        undo_btn.pack(side=tk.LEFT, padx=6)

        clear_btn = tk.Button(
            btn_frame,
            text="CLEAR  [ 🗑 ]",
            font=("Segoe UI", 10, "bold"),
            bg="#7f1d1d",
            fg="#fca5a5",
            activebackground="#991b1b",
            relief=tk.FLAT,
            padx=14,
            pady=8,
            command=self._on_clear,
        )
        clear_btn.pack(side=tk.LEFT, padx=6)

        copy_btn = tk.Button(
            btn_frame,
            text="COPY",
            font=("Segoe UI", 10),
            bg="#334155",
            fg="#f8fafc",
            activebackground="#1e293b",
            relief=tk.FLAT,
            padx=12,
            pady=8,
            command=self._on_copy,
        )
        copy_btn.pack(side=tk.LEFT, padx=6)

        exit_btn = tk.Button(
            btn_frame,
            text="EXIT",
            font=("Segoe UI", 10, "bold"),
            bg="#b91c1c",
            fg="#ffffff",
            activebackground="#991b1b",
            relief=tk.FLAT,
            padx=16,
            pady=8,
            command=self.on_close,
        )
        exit_btn.pack(side=tk.RIGHT, padx=(6, 0))

    def _camera_worker(self):
        """Worker thread executing camera capture and ResNet inference asynchronously."""
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.03)
                continue

            prediction = self.predictor.process_frame(frame, mirror=self.mirror)

            # Draw visual annotation on camera frame
            display_frame = frame.copy()
            if self.mirror:
                display_frame = cv2.flip(display_frame, 1)

            if prediction.box is not None:
                l, t, r, b = prediction.box
                box_color = (0, 220, 0) if prediction.is_valid else (0, 165, 255)
                cv2.rectangle(display_frame, (l, t), (r, b), box_color, 2)

            # Send latest frame and prediction to GUI thread
            try:
                # Discard stale frame if GUI hasn't consumed yet
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                self.frame_queue.put_nowait((display_frame, prediction))
            except queue.Full:
                pass

    def _ui_tick(self):
        """Consume latest frame from worker thread and update UI widgets."""
        if not self.running:
            return

        try:
            display_frame, pred = self.frame_queue.get_nowait()

            # 1. Update text composer state
            new_letter_appended = self.composer.update(
                predicted_char=pred.predicted_char,
                is_valid=pred.is_valid,
                confidence=pred.confidence,
            )

            # 2. Update status and progress bar
            progress = self.composer.get_hold_progress()
            self.progress_bar["value"] = int(progress * 100)

            if pred.predicted_char and pred.is_valid:
                self.candidate_label.config(text=pred.predicted_char, fg="#10b981")
                self.status_detail_label.config(
                    text=f"{pred.display_text}",
                    fg="#10b981",
                )
                if not self.composer.is_armed:
                    self.progress_text.config(
                        text=f"Confirmed [{pred.predicted_char}]! Release to enter next.",
                        fg="#38bdf8",
                    )
                else:
                    self.progress_text.config(
                        text=f"Holding steady: {int(progress * 100)}%",
                        fg="#fcd34d",
                    )
            elif pred.predicted_char:
                self.candidate_label.config(text=pred.predicted_char, fg="#f59e0b")
                self.status_detail_label.config(text=f"{pred.display_text}", fg="#f59e0b")
                self.progress_text.config(text="Unstable or incomplete hand count", fg="#94a3b8")
            else:
                self.candidate_label.config(text="—", fg="#94a3b8")
                self.status_detail_label.config(text="No hand detected", fg="#94a3b8")
                self.progress_text.config(text="Position hand in camera frame", fg="#64748b")

            # 3. Update text area if content changed
            current_composed = self.composer.get_text()
            if self.text_display.get("1.0", tk.END).strip() != current_composed:
                self.text_display.delete("1.0", tk.END)
                self.text_display.insert(tk.END, current_composed)
                self.text_display.see(tk.END)

            # 4. Render video frame in Tkinter label
            rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            # Resize proportionally to fit canvas
            target_w = 520
            target_h = int(h * (target_w / w))
            resized = cv2.resize(rgb, (target_w, target_h))
            img = Image.fromarray(resized)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_canvas.imgtk = imgtk
            self.video_canvas.configure(image=imgtk)

        except queue.Empty:
            pass

        if self.running:
            self.root.after(20, self._ui_tick)

    def _on_space(self):
        self.composer.add_space()
        self._refresh_text()

    def _on_undo(self):
        self.composer.undo()
        self._refresh_text()

    def _on_clear(self):
        self.composer.clear()
        self._refresh_text()

    def _on_copy(self):
        text = self.composer.get_text()
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        messagebox.showinfo("Copied", "Composed text copied to clipboard!")

    def _refresh_text(self):
        self.text_display.delete("1.0", tk.END)
        self.text_display.insert(tk.END, self.composer.get_text())
        self.text_display.see(tk.END)

    def on_close(self):
        """Clean shutdown releasing camera and closing windows."""
        self.running = False
        try:
            if hasattr(self, "cap") and self.cap.isOpened():
                self.cap.release()
            if hasattr(self, "predictor"):
                self.predictor.close()
        except Exception:
            pass
        self.root.destroy()


def parse_args():
    parser = argparse.ArgumentParser(description="Run ISL Alphabet Live Text Composer.")
    parser.add_argument("--model", choices=("large", "resnet_large", "balanced", "resnet_balanced", "resnet", "cnn"), default=None,
                        help="Model alias to load (default: active model from config, e.g. isl_resnet_large.pth).")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Explicit path to trained model checkpoint.")
    parser.add_argument("--camera", type=int, default=0, help="Webcam device ID.")
    parser.add_argument("--hold-duration", type=float, default=0.8,
                        help="Duration in seconds sign must be held steady to confirm letter.")
    parser.add_argument("--mirror", action="store_true", default=False,
                        help="Mirror preview horizontally (DEFAULT: False).")
    return parser.parse_args()


def main():
    args = parse_args()
    ckpt_path, display_name = resolve_checkpoint(model_arg=args.model, checkpoint_arg=args.checkpoint)
    print(f"Starting Text Composer with model: {display_name} ({ckpt_path.name})")
    root = tk.Tk()
    app = TextComposerApp(
        root=root,
        checkpoint_path=ckpt_path,
        camera_id=args.camera,
        hold_duration=args.hold_duration,
        mirror=args.mirror,
    )
    root.mainloop()


if __name__ == "__main__":
    main()
