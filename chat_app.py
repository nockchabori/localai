import customtkinter as ctk
import os
import customtkinter as ctk
import os
import json
import threading
from huggingface_hub import hf_hub_download

try:
    from llama_cpp import Llama
    LLAMA_AVAILABLE = True
except ImportError:
    LLAMA_AVAILABLE = False

ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

# 다국어 번역 딕셔너리
LANGUAGES = {
    "한국어": {
        "app_title": "Local LLM Chat App",
        "chat_title": "💬 로컬 AI 메신저",
        "btn_settings": "⚙️ 모델 선택 및 설정",
        "input_placeholder": "메시지를 입력하세요...",
        "btn_send": "전송",
        "btn_clear": "대화 비우기",
        "welcome_msg": "안녕하세요! 상단의 [⚙️ 모델 선택 및 설정] 버튼을 눌러 원하는 LLM을 로드해 보세요.",
        "clear_msg": "대화 내역이 초기화되었습니다.",
        "settings_title": "⚙️ 설정 및 모델 관리",
        "tab_model": "모델 다운로드 / 로드",
        "tab_settings": "파라미터 / 환경 설정",
        "lbl_preset": "추천 모델 프리셋 선택:",
        "lbl_repo": "Hugging Face Repo ID:",
        "lbl_file": "Model File Name (.gguf):",
        "btn_download": "📥 모델 다운로드",
        "lbl_local": "보유 중인 로컬 모델 선택:",
        "btn_load": "⚡ 선택한 모델 로드",
        "lbl_sys_prompt": "시스템 프롬프트 (System Prompt):",
        "lbl_temp": "Temperature (창의성):",
        "lbl_tokens": "최대 생성 토큰 (Max Tokens):",
        "lbl_lang": "언어 설정 (Language):",
        "lbl_theme": "화면 테마:",
        "btn_save": "💾 설정 저장",
        "btn_reset": "🔄 초기화",
        "status_idle": "상태: 대기 중",
        "status_downloading": "다운로드 진행 중...",
        "status_download_done": "다운로드 완료!",
        "status_loading": "메모리 로딩 중...",
        "status_ready": "모델 준비 완료 (Ready)",
        "default_sys_prompt": "You are a helpful AI assistant. Answer kindly in Korean."
    },
    "English": {
        "app_title": "Local LLM Chat App",
        "chat_title": "💬 Local AI Messenger",
        "btn_settings": "⚙️ Settings & Models",
        "input_placeholder": "Type a message...",
        "btn_send": "Send",
        "btn_clear": "Clear Chat",
        "welcome_msg": "Hello! Click [⚙️ Settings & Models] above to download and load an LLM.",
        "clear_msg": "Chat history has been cleared.",
        "settings_title": "⚙️ Settings & Model Management",
        "tab_model": "Download / Load Model",
        "tab_settings": "Parameters & Environment",
        "lbl_preset": "Preset Models:",
        "lbl_repo": "Hugging Face Repo ID:",
        "lbl_file": "Model File Name (.gguf):",
        "btn_download": "📥 Download Model",
        "lbl_local": "Select Local Model:",
        "btn_load": "⚡ Load Selected Model",
        "lbl_sys_prompt": "System Prompt:",
        "lbl_temp": "Temperature:",
        "lbl_tokens": "Max Tokens:",
        "lbl_lang": "Language:",
        "lbl_theme": "Appearance Theme:",
        "btn_save": "💾 Save Settings",
        "btn_reset": "🔄 Reset",
        "status_idle": "Status: Idle",
        "status_downloading": "Downloading...",
        "status_download_done": "Download completed!",
        "status_loading": "Loading into memory...",
        "status_ready": "Model Ready",
        "default_sys_prompt": "You are a helpful AI assistant. Answer kindly in English."
    },
    "日本語": {
        "app_title": "Local LLM Chat App",
        "chat_title": "💬 ローカル AI メッセンジャー",
        "btn_settings": "⚙️ モデル選択・設定",
        "input_placeholder": "メッセージを入力してください...",
        "btn_send": "送信",
        "btn_clear": "履歴削除",
        "welcome_msg": "こんにちは！上部の [⚙️ モデル選択・設定] からモデルをロードしてください。",
        "clear_msg": "チャット履歴が消去されました。",
        "settings_title": "⚙️ 設定とモデル管理",
        "tab_model": "モデル取得 / ロード",
        "tab_settings": "パラメータ / 環境設定",
        "lbl_preset": "おすすめモデルプリセット:",
        "lbl_repo": "Hugging Face Repo ID:",
        "lbl_file": "モデルファイル名 (.gguf):",
        "btn_download": "📥 モデルをダウンロード",
        "lbl_local": "保存済みローカルモデル:",
        "btn_load": "⚡ 選択したモデルをロード",
        "lbl_sys_prompt": "システムプロンプト (System Prompt):",
        "lbl_temp": "Temperature (創造性):",
        "lbl_tokens": "最大生成トークン数 (Max Tokens):",
        "lbl_lang": "言語設定 (Language):",
        "lbl_theme": "テーマ設定:",
        "btn_save": "💾 設定保存",
        "btn_reset": "🔄 初期化",
        "status_idle": "状態: 待機中",
        "status_downloading": "ダウンロード中...",
        "status_download_done": "ダウンロード完了！",
        "status_loading": "メモリに読み込み中...",
        "status_ready": "モデル準備完了 (Ready)",
        "default_sys_prompt": "You are a helpful AI assistant. Answer kindly in Japanese."
    }
}

MODEL_PRESETS = {
    "Qwen 2.5 0.5B (Lightweight)": {
        "repo_id": "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        "filename": "qwen2.5-0.5b-instruct-q4_k_m.gguf"
    },
    "Llama 3.2 1B (Smart & Fast)": {
        "repo_id": "bartowski/Llama-3.2-1B-Instruct-GGUF",
        "filename": "Llama-3.2-1B-Instruct-Q4_K_M.gguf"
    },
    "Gemma 2 2B (Google)": {
        "repo_id": "bartowski/gemma-2-2b-it-GGUF",
        "filename": "gemma-2-2b-it-Q4_K_M.gguf"
    },
    "Phi-3.5 Mini 3.8B (High Quality)": {
        "repo_id": "bartowski/Phi-3.5-mini-instruct-GGUF",
        "filename": "Phi-3.5-mini-instruct-Q4_K_M.gguf"
    },
    "Custom / 직접 입력": {
        "repo_id": "",
        "filename": ""
    }
}

DEFAULT_CONFIG = {
    "repo_id": "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
    "filename": "qwen2.5-0.5b-instruct-q4_k_m.gguf",
    "system_prompt": "You are a helpful AI assistant. Answer kindly in Korean.",
    "temperature": 0.7,
    "max_tokens": 256,
    "theme": "System",
    "language": "한국어"
}

# ==========================================
# 별도 팝업 설정 윈도우 (CTkToplevel)
# ==========================================
class SettingsWindow(ctk.CTkToplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent

        self.geometry("490x650")
        self.minsize(450, 580)
        self.attributes("-topmost", True)

        self._build_ui()
        self.update_texts()
        self.load_current_values()
        self.refresh_local_models()

    def _build_ui(self):
        self.tabview = ctk.CTkTabview(self)
        self.tabview.pack(padx=15, pady=15, fill="both", expand=True)

        self.tab_model = self.tabview.add("tab_model")
        self.tab_settings = self.tabview.add("tab_settings")

        # --- [탭 1: 모델 관리] ---
        self.lbl_preset = ctk.CTkLabel(self.tab_model, text="", font=ctk.CTkFont(size=12, weight="bold"))
        self.lbl_preset.pack(padx=5, anchor="w", pady=(5, 0))
        self.preset_option = ctk.CTkOptionMenu(self.tab_model, values=list(MODEL_PRESETS.keys()), command=self._on_preset_selected)
        self.preset_option.pack(padx=5, pady=(2, 10), fill="x")

        self.lbl_repo = ctk.CTkLabel(self.tab_model, text="", font=ctk.CTkFont(size=12))
        self.lbl_repo.pack(padx=5, anchor="w")
        self.repo_entry = ctk.CTkEntry(self.tab_model)
        self.repo_entry.pack(padx=5, pady=(2, 6), fill="x")

        self.lbl_file = ctk.CTkLabel(self.tab_model, text="", font=ctk.CTkFont(size=12))
        self.lbl_file.pack(padx=5, anchor="w")
        self.file_entry = ctk.CTkEntry(self.tab_model)
        self.file_entry.pack(padx=5, pady=(2, 6), fill="x")

        self.download_btn = ctk.CTkButton(self.tab_model, text="", command=self.start_download)
        self.download_btn.pack(padx=5, pady=4, fill="x")

        self.progress_bar = ctk.CTkProgressBar(self.tab_model)
        self.progress_bar.set(0)
        self.progress_bar.pack(padx=5, pady=4, fill="x")

        self.lbl_local = ctk.CTkLabel(self.tab_model, text="", font=ctk.CTkFont(size=12, weight="bold"))
        self.lbl_local.pack(padx=5, anchor="w", pady=(10, 0))
        self.local_models_option = ctk.CTkOptionMenu(self.tab_model, values=["(Empty)"], command=self._on_local_model_selected)
        self.local_models_option.pack(padx=5, pady=(2, 6), fill="x")

        self.load_btn = ctk.CTkButton(self.tab_model, text="", fg_color="#2b8a3e", hover_color="#237032", command=self.start_load_model)
        self.load_btn.pack(padx=5, pady=4, fill="x")

        self.status_label = ctk.CTkLabel(self.tab_model, text="", text_color="gray", wraplength=380)
        self.status_label.pack(padx=5, pady=5)

        # --- [탭 2: 파라미터 / 테마 / 언어] ---
        self.lbl_sys_prompt = ctk.CTkLabel(self.tab_settings, text="", font=ctk.CTkFont(size=12, weight="bold"))
        self.lbl_sys_prompt.pack(padx=5, anchor="w", pady=(5, 0))
        self.sys_prompt_entry = ctk.CTkTextbox(self.tab_settings, height=75)
        self.sys_prompt_entry.pack(padx=5, pady=(2, 6), fill="x")

        self.temp_label = ctk.CTkLabel(self.tab_settings, text="", font=ctk.CTkFont(size=12))
        self.temp_label.pack(padx=5, anchor="w")
        self.temp_slider = ctk.CTkSlider(self.tab_settings, from_=0.0, to=1.5, number_of_steps=15, command=self._update_temp_label)
        self.temp_slider.pack(padx=5, pady=(2, 6), fill="x")

        self.lbl_tokens = ctk.CTkLabel(self.tab_settings, text="", font=ctk.CTkFont(size=12))
        self.lbl_tokens.pack(padx=5, anchor="w")
        self.tokens_entry = ctk.CTkEntry(self.tab_settings)
        self.tokens_entry.pack(padx=5, pady=(2, 6), fill="x")

        # 언어 설정 드롭다운
        self.lbl_lang = ctk.CTkLabel(self.tab_settings, text="", font=ctk.CTkFont(size=12, weight="bold"))
        self.lbl_lang.pack(padx=5, anchor="w", pady=(4, 0))
        self.lang_option = ctk.CTkOptionMenu(self.tab_settings, values=list(LANGUAGES.keys()), command=self._on_language_change)
        self.lang_option.pack(padx=5, pady=(2, 6), fill="x")

        self.lbl_theme = ctk.CTkLabel(self.tab_settings, text="", font=ctk.CTkFont(size=12))
        self.lbl_theme.pack(padx=5, anchor="w")
        self.theme_option = ctk.CTkOptionMenu(self.tab_settings, values=["System", "Dark", "Light"], command=self.change_theme)
        self.theme_option.pack(padx=5, pady=(2, 10), fill="x")

        btn_frame = ctk.CTkFrame(self.tab_settings, fg_color="transparent")
        btn_frame.pack(fill="x", pady=5)
        btn_frame.grid_columnconfigure((0, 1), weight=1)

        self.save_btn = ctk.CTkButton(btn_frame, text="", command=self.save_settings)
        self.save_btn.grid(row=0, column=0, padx=3, sticky="ew")

        self.reset_btn = ctk.CTkButton(btn_frame, text="", fg_color="#6c757d", hover_color="#5a6268", command=self.reset_settings)
        self.reset_btn.grid(row=0, column=1, padx=3, sticky="ew")

    def update_texts(self):
        """선택된 언어에 맞춰 텍스트 일괄 갱신"""
        lang = self.parent.config.get("language", "한국어")
        t = LANGUAGES.get(lang, LANGUAGES["한국어"])

        self.title(t["settings_title"])
        self.tabview._segmented_button._buttons_dict["tab_model"].configure(text=t["tab_model"])
        self.tabview._segmented_button._buttons_dict["tab_settings"].configure(text=t["tab_settings"])

        self.lbl_preset.configure(text=t["lbl_preset"])
        self.lbl_repo.configure(text=t["lbl_repo"])
        self.lbl_file.configure(text=t["lbl_file"])
        self.download_btn.configure(text=t["btn_download"])
        self.lbl_local.configure(text=t["lbl_local"])
        self.load_btn.configure(text=t["btn_load"])
        self.status_label.configure(text=t["status_idle"])

        self.lbl_sys_prompt.configure(text=t["lbl_sys_prompt"])
        self.temp_label.configure(text=f"{t['lbl_temp']} {float(self.temp_slider.get()):.1f}")
        self.lbl_tokens.configure(text=t["lbl_tokens"])
        self.lbl_lang.configure(text=t["lbl_lang"])
        self.lbl_theme.configure(text=t["lbl_theme"])
        self.save_btn.configure(text=t["btn_save"])
        self.reset_btn.configure(text=t["btn_reset"])

    def _on_language_change(self, new_lang):
        self.parent.config["language"] = new_lang
        # 기본 시스템 프롬프트도 해당 언어에 맞게 자동 전환
        self.sys_prompt_entry.delete("1.0", "end")
        self.sys_prompt_entry.insert("1.0", LANGUAGES[new_lang]["default_sys_prompt"])

        self.update_texts()
        self.parent.update_ui_texts()

    def _on_preset_selected(self, choice):
        preset = MODEL_PRESETS.get(choice, {})
        if preset.get("repo_id"):
            self.repo_entry.delete(0, 'end')
            self.repo_entry.insert(0, preset["repo_id"])
            self.file_entry.delete(0, 'end')
            self.file_entry.insert(0, preset["filename"])

    def _on_local_model_selected(self, choice):
        if choice and not choice.startswith("("):
            self.file_entry.delete(0, 'end')
            self.file_entry.insert(0, choice)

    def refresh_local_models(self):
        if os.path.exists(self.parent.models_dir):
            files = [f for f in os.listdir(self.parent.models_dir) if f.endswith(".gguf")]
            if files:
                self.local_models_option.configure(values=files)
                self.local_models_option.set(files[0])
                return
        self.local_models_option.configure(values=["(No GGUF models)"])
        self.local_models_option.set("(No GGUF models)")

    def _update_temp_label(self, value):
        lang = self.parent.config.get("language", "한국어")
        prefix = LANGUAGES.get(lang, LANGUAGES["한국어"])["lbl_temp"]
        self.temp_label.configure(text=f"{prefix} {float(value):.1f}")

    def change_theme(self, mode):
        ctk.set_appearance_mode(mode)
        self.parent.config["theme"] = mode

    def load_current_values(self):
        cfg = self.parent.config
        self.repo_entry.delete(0, 'end')
        self.repo_entry.insert(0, cfg["repo_id"])

        self.file_entry.delete(0, 'end')
        self.file_entry.insert(0, cfg["filename"])

        self.sys_prompt_entry.delete("1.0", "end")
        self.sys_prompt_entry.insert("1.0", cfg["system_prompt"])

        self.temp_slider.set(cfg["temperature"])
        self._update_temp_label(cfg["temperature"])

        self.tokens_entry.delete(0, 'end')
        self.tokens_entry.insert(0, str(cfg["max_tokens"]))

        self.theme_option.set(cfg["theme"])
        self.lang_option.set(cfg.get("language", "한국어"))

    def save_settings(self):
        try:
            self.parent.config = {
                "repo_id": self.repo_entry.get().strip(),
                "filename": self.file_entry.get().strip(),
                "system_prompt": self.sys_prompt_entry.get("1.0", "end-1c").strip(),
                "temperature": float(self.temp_slider.get()),
                "max_tokens": int(self.tokens_entry.get().strip() or 256),
                "theme": self.theme_option.get(),
                "language": self.lang_option.get()
            }
            with open(self.parent.config_path, "w", encoding="utf-8") as f:
                json.dump(self.parent.config, f, indent=4, ensure_ascii=False)
            self.status_label.configure(text="Saved successfully!", text_color="#2b8a3e")
        except Exception as e:
            self.status_label.configure(text=f"Error: {e}", text_color="red")

    def reset_settings(self):
        self.parent.config = DEFAULT_CONFIG.copy()
        self.load_current_values()
        self.update_texts()
        self.parent.update_ui_texts()
        self.save_settings()

    # --- 다운로드 & 로드 스레드 ---
    def start_download(self):
        threading.Thread(target=self._download_model_task, daemon=True).start()

    def _download_model_task(self):
        repo_id = self.repo_entry.get().strip()
        filename = self.file_entry.get().strip()
        lang = self.parent.config.get("language", "한국어")
        t = LANGUAGES.get(lang, LANGUAGES["한국어"])

        if not repo_id or not filename:
            self.status_label.configure(text="Error: Enter Repo ID and filename.", text_color="red")
            return

        self.download_btn.configure(state="disabled")
        self.status_label.configure(text=t["status_downloading"], text_color="#3B8ED0")
        self.progress_bar.start()

        try:
            downloaded_path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=self.parent.models_dir)
            self.parent.model_path = downloaded_path
            self.status_label.configure(text=f"{t['status_download_done']}\n{filename}", text_color="#2b8a3e")
            self.refresh_local_models()
        except Exception as e:
            self.status_label.configure(text=f"Download Error: {str(e)[:35]}...", text_color="red")
        finally:
            self.progress_bar.stop()
            self.progress_bar.set(1.0)
            self.download_btn.configure(state="normal")

    def start_load_model(self):
        threading.Thread(target=self._load_model_task, daemon=True).start()

    def _load_model_task(self):
        filename = self.file_entry.get().strip()
        target_path = os.path.join(self.parent.models_dir, filename)
        lang = self.parent.config.get("language", "한국어")
        t = LANGUAGES.get(lang, LANGUAGES["한국어"])

        if not os.path.exists(target_path):
            self.status_label.configure(text="Error: Model file not found.", text_color="red")
            return

        if not LLAMA_AVAILABLE:
            self.status_label.configure(text="Error: llama-cpp-python not installed.", text_color="red")
            return

        self.load_btn.configure(state="disabled")
        self.status_label.configure(text=t["status_loading"], text_color="#3B8ED0")
        try:
            self.parent.llm = None
            self.parent.llm = Llama(model_path=target_path, n_ctx=2048, verbose=False)
            self.status_label.configure(text=f"{t['status_ready']}: {filename}", text_color="#2b8a3e")
            self.parent.add_message(f"[{filename}] Loaded successfully.", sender="bot")
        except Exception as e:
            self.status_label.configure(text=f"Load Error: {str(e)[:35]}...", text_color="red")
        finally:
            self.load_btn.configure(state="normal")


# ==========================================
# 메인 채팅 윈도우
# ==========================================
class LLMChatApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.geometry("650x700")
        self.minsize(500, 500)

        base_dir = os.path.dirname(__file__) if "__file__" in globals() else os.getcwd()
        self.models_dir = os.path.join(base_dir, "models")
        self.config_path = os.path.join(base_dir, "config.json")
        os.makedirs(self.models_dir, exist_ok=True)

        self.llm = None
        self.model_path = None
        self.config = self.load_config()
        self.settings_window = None

        ctk.set_appearance_mode(self.config.get("theme", "System"))

        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self._build_top_bar()
        self._build_chat_area()
        self.update_ui_texts()

        # 시작 환영 메시지
        lang = self.config.get("language", "한국어")
        self.add_message(LANGUAGES.get(lang, LANGUAGES["한국어"])["welcome_msg"], sender="bot")

    def load_config(self):
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    return {**DEFAULT_CONFIG, **json.load(f)}
            except Exception:
                return DEFAULT_CONFIG.copy()
        return DEFAULT_CONFIG.copy()

    def update_ui_texts(self):
        """메인 윈도우 UI 텍스트 언어 변경 반영"""
        lang = self.config.get("language", "한국어")
        t = LANGUAGES.get(lang, LANGUAGES["한국어"])

        self.title(t["app_title"])
        self.title_lbl.configure(text=t["chat_title"])
        self.settings_btn.configure(text=t["btn_settings"])
        self.msg_entry.configure(placeholder_text=t["input_placeholder"])
        self.send_button.configure(text=t["btn_send"])
        self.clear_btn.configure(text=t["btn_clear"])

    def open_settings_window(self):
        if self.settings_window is None or not self.settings_window.winfo_exists():
            self.settings_window = SettingsWindow(self)
        else:
            self.settings_window.focus()

    def _build_top_bar(self):
        top_bar = ctk.CTkFrame(self, fg_color="transparent")
        top_bar.grid(row=0, column=0, sticky="ew", padx=15, pady=(15, 5))

        self.title_lbl = ctk.CTkLabel(top_bar, text="", font=ctk.CTkFont(size=18, weight="bold"))
        self.title_lbl.pack(side="left")

        self.settings_btn = ctk.CTkButton(
            top_bar, 
            text="", 
            width=140, 
            height=32, 
            fg_color="#343a40", 
            hover_color="#212529", 
            command=self.open_settings_window
        )
        self.settings_btn.pack(side="right")

    def _build_chat_area(self):
        self.chat_history_frame = ctk.CTkScrollableFrame(self)
        self.chat_history_frame.grid(row=1, column=0, sticky="nsew", padx=15, pady=(5, 10))
        self.chat_history_frame.grid_columnconfigure(0, weight=1)

        self.input_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.input_frame.grid(row=2, column=0, sticky="ew", padx=15, pady=(0, 15))
        self.input_frame.grid_columnconfigure(0, weight=1)

        self.msg_entry = ctk.CTkEntry(self.input_frame, height=40)
        self.msg_entry.grid(row=0, column=0, padx=(0, 10), sticky="ew")
        self.msg_entry.bind("<Return>", lambda event: self.send_message())

        self.send_button = ctk.CTkButton(self.input_frame, width=70, height=40, command=self.send_message)
        self.send_button.grid(row=0, column=1, padx=(0, 5))

        self.clear_btn = ctk.CTkButton(self.input_frame, width=80, height=40, fg_color="#c92a2a", hover_color="#a61e1e", command=self.clear_chat)
        self.clear_btn.grid(row=0, column=2)

    def clear_chat(self):
        for widget in self.chat_history_frame.winfo_children():
            widget.destroy()
        lang = self.config.get("language", "한국어")
        self.add_message(LANGUAGES.get(lang, LANGUAGES["한국어"])["clear_msg"], sender="bot")

    def add_message(self, text, sender="user"):
        if sender == "user":
            msg_frame = ctk.CTkFrame(self.chat_history_frame, fg_color="#1f6aa5", corner_radius=12)
            align = "e"
        else:
            msg_frame = ctk.CTkFrame(self.chat_history_frame, fg_color="#3a3a3a", corner_radius=12)
            align = "w"

        msg_label = ctk.CTkLabel(
            msg_frame,
            text=text,
            wraplength=420,
            justify="left" if sender == "bot" else "right",
            text_color="white"
        )
        msg_label.pack(padx=12, pady=8)
        msg_frame.pack(anchor=align, pady=5, padx=5)
        self.chat_history_frame._parent_canvas.yview_moveto(1.0)

    def send_message(self):
        user_text = self.msg_entry.get().strip()
        if not user_text:
            return

        self.add_message(user_text, sender="user")
        self.msg_entry.delete(0, 'end')

        threading.Thread(target=self._generate_bot_reply, args=(user_text,), daemon=True).start()

    def _generate_bot_reply(self, user_text):
        if self.llm is not None:
            try:
                sys_prompt = self.config.get("system_prompt", "You are a helpful AI assistant.")
                temperature = float(self.config.get("temperature", 0.7))
                max_tokens = int(self.config.get("max_tokens", 256))

                response = self.llm.create_chat_completion(
                    messages=[
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": user_text}
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                reply = response['choices'][0]['message']['content'].strip()
            except Exception as e:
                reply = f"[Error]: {e}"
        else:
            lang = self.config.get("language", "한국어")
            if lang == "한국어":
                reply = f"(모델 미로드 상태) '{user_text}'. 상단 설정 창에서 모델을 로드해 주세요."
            elif lang == "English":
                reply = f"(Model not loaded) '{user_text}'. Please load a model from Settings."
            else:
                reply = f"(モデル未ロード) '{user_text}'. 設定画面からモデルをロードしてください。"

        self.after(0, lambda: self.add_message(reply, sender="bot"))

if __name__ == "__main__":
    app = LLMChatApp()
    app.mainloop()
