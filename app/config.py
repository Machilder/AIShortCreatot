from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    # アプリケーション設定
    APP_NAME: str = "Video Summary API"
    DEBUG: bool = False
    
    # AWS設定
    AWS_ACCESS_KEY_ID: str
    AWS_SECRET_ACCESS_KEY: str
    AWS_REGION: str = "ap-northeast-1"
    S3_BUCKET: str
    
    # YouTube API設定
    YOUTUBE_API_KEY: str
    
    # 一時ファイル保存パス
    TEMP_FILE_PATH: str = "./temp"
    
    class Config:
        env_file = ".env"

@lru_cache()
def get_settings():
    return Settings()