from pathlib import Path
import os
import torch
import logging
from typing import Dict
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

class Config:
    def __init__(self):
        # .env ファイルがあれば読み込む
        load_dotenv()
        
        # 基本設定
        self.base_dir = Path(__file__).parent.parent
        self.temp_dir = self.base_dir / "temp"
        self.temp_dir.mkdir(exist_ok=True)
        
        # デバイス設定
        self.force_cpu = os.getenv('FORCE_CPU', 'false').lower() == 'true'
        self.device = self._get_device()
        
        # GPU設定
        if self.force_cpu:
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
        
        # モデル設定
        self.whisper_model = os.getenv('WHISPER_MODEL', 'base')
        self.summarization_model = os.getenv('SUMMARIZATION_MODEL', 'sonoisa/t5-base-japanese-summarization')
        
        # 処理設定
        self.batch_size = int(os.getenv('BATCH_SIZE', '1'))
        self.max_workers = int(os.getenv('MAX_WORKERS', '2'))
        
        logger.info(f"Initialized config: device={self.device}, "f"whisper_model={self.whisper_model}, "f"force_cpu={self.force_cpu}")
    def _get_device(self) -> str:
        """
        実行環境に応じて適切なデバイスを選択
        """
        if self.force_cpu:
            logger.info("Forcing CPU usage as per configuration")
            return "cpu"
            
        if torch.cuda.is_available():
            logger.info(f"CUDA is available: {torch.cuda.get_device_name(0)}")
            return "cuda"
            
        logger.info("CUDA is not available, using CPU")
        return "cpu"

    def get_model_config(self) -> Dict:
        """
        モデル関連の設定をまとめて取得
        """
        return {
            "device": self.device,
            "whisper_model": self.whisper_model,
            "summarization_model": self.summarization_model,
            "batch_size": self.batch_size
        }

    def to_dict(self) -> Dict:
        """
        全ての設定値を辞書形式で取得
        """
        return {
            "base_dir": str(self.base_dir),
            "temp_dir": str(self.temp_dir),
            "device": self.device,
            "force_cpu": self.force_cpu,
            "whisper_model": self.whisper_model,
            "summarization_model": self.summarization_model,
            "batch_size": self.batch_size,
            "max_workers": self.max_workers
        }