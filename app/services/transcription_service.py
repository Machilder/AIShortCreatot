import whisper
import torch
import os
import logging
from pathlib import Path
from typing import Optional, Dict
from app.config import Config

logger = logging.getLogger(__name__)

class TranscriptionService:
    def __init__(self, config: Config):
        self.config = config
        self.model = whisper.load_model(config.whisper_model)
        logger.info(f"Initialized Whisper model '{config.whisper_model}' on device '{config.device}'")

    async def transcribe_video(self, video_path: str) -> Optional[Dict]:
        """
        動画ファイルから音声を抽出し、文字起こしを行う
        """
        try:
            video_path = str(Path(fr"{video_path}").resolve())
            logger.info(f"Starting transcription for video: {video_path}")
            
            if not os.path.exists(video_path):
                logger.error(f"Video file not found: {video_path}")
                return None
                
            # 設定情報のログ出力
            logger.info(f"Using model: {self.config.whisper_model} on device: {self.config.device}")
            
            import asyncio
            loop = asyncio.get_event_loop()
            
            # デバッグ用のファイルサイズチェック
            file_size = os.path.getsize(video_path) / (1024 * 1024)  # MBに変換
            logger.info(f"Processing file of size: {file_size:.2f}MB")
            
            result = await loop.run_in_executor(
                None,
                lambda: self.model.transcribe(
                    str(video_path),
                    fp16=False,
                    language="ja",
                    task="transcribe"  # タスクを明示的に指定
                )
            )

            # セグメント情報を含む結果を返す
            transcription = {
                'text': result['text'],
                'segments': [
                    {
                        'start': segment['start'],
                        'end': segment['end'],
                        'text': segment['text']
                    }
                    for segment in result['segments']
                ]
            }

            logger.info(f"Transcription completed successfully")
            return transcription

        except Exception as e:
            logger.error(f"Error during transcription: {str(e)}", exc_info=True)
            return None

    def cleanup(self):
        """
        モデルのクリーンアップ処理
        """
        try:
            # GPUメモリの解放
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # その他必要なクリーンアップ処理
            self.model = None
            
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")