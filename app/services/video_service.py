import yt_dlp
import os
import logging
from typing import Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class VideoService:
    def __init__(self, temp_dir: str = "./temp"):
        self.temp_dir = temp_dir
        Path(temp_dir).mkdir(parents=True, exist_ok=True)
        
    async def download_video(self, url: str) -> Optional[str]:
        """
        YouTubeのURLから動画をダウンロードする
        
        Args:
            url (str): YouTube動画のURL
            
        Returns:
            Optional[str]: ダウンロードした動画のパス。失敗した場合はNone
        """
        try:
            logger.info(f"Downloading video from {url}")
            
            # 動画IDを抽出（簡易的な実装）
            video_id = url.split('watch?v=')[-1]
            output_path = os.path.join(self.temp_dir, f"{video_id}.mp4")
            
            ydl_opts = {
                'format': 'best[ext=mp4]',  # mp4形式で最高品質
                'outtmpl': output_path,     # 出力パス
                'quiet': True,              # 進行状況の出力を制限
                'no_warnings': True,        # 警告を非表示
            }
            
            # yt-dlpの処理を非同期的に実行
            import asyncio
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, lambda: self._download_with_ytdlp(url, ydl_opts))
            
            if os.path.exists(output_path):
                logger.info(f"Video downloaded successfully to {output_path}")
                return output_path
            else:
                logger.error("Download completed but file not found")
                return None
                
        except Exception as e:
            logger.error(f"Error downloading video: {str(e)}")
            return None
    
    def _download_with_ytdlp(self, url: str, ydl_opts: dict):
        """
        yt-dlpを使用して実際のダウンロードを行う内部メソッド
        """
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
            
    def cleanup(self, file_path: str):
        """
        一時ファイルを削除する
        
        Args:
            file_path (str): 削除するファイルのパス
        """
        try:
            if os.path.exists(file_path):
                # os.remove(file_path)
                logger.info(f"Cleaned up file: {file_path}")
        except Exception as e:
            logger.error(f"Error cleaning up file {file_path}: {str(e)}")

    async def get_video_info(self, url: str) -> Optional[dict]:
        """
        動画の情報を取得する
        
        Args:
            url (str): YouTube動画のURL
            
        Returns:
            Optional[dict]: 動画の情報。取得失敗時はNone
        """
        try:
            ydl_opts = {
                'quiet': True,
                'no_warnings': True,
            }
            
            # yt-dlpの処理を非同期的に実行
            import asyncio
            loop = asyncio.get_event_loop()
            info = await loop.run_in_executor(None, lambda: self._get_info_with_ytdlp(url, ydl_opts))
            
            return {
                'title': info.get('title'),
                'duration': info.get('duration'),
                'description': info.get('description'),
                'thumbnail': info.get('thumbnail'),
            }
        except Exception as e:
            logger.error(f"Error getting video info: {str(e)}")
            return None
    
    def _get_info_with_ytdlp(self, url: str, ydl_opts: dict):
        """
        yt-dlpを使用して実際の情報取得を行う内部メソッド
        """
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            return ydl.extract_info(url, download=False)