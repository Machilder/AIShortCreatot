from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, HttpUrl
import torch
import uuid
from typing import Optional
import logging
from .services.video_service import VideoService
from .services.transcription_service import TranscriptionService
from .services.summarization_service import SummarizationService
from .config import Config

# 設定の初期化
config = Config()

transcription_service = TranscriptionService(config)
summarization_service = SummarizationService(config)

class VideoRequest(BaseModel):
    url: HttpUrl
    max_duration: Optional[int] = 180  # 要約動画の最大時間（秒）
    quality: Optional[str] = "best"    # 動画品質（'best', 'medium', 'worst'）
    include_audio: bool = True         # 音声を含むかどうか

class VideoInfo(BaseModel):
    title: str
    duration: int
    description: Optional[str]
    thumbnail: Optional[str]

class ProcessingProgress:
    def __init__(self):
        self.total_steps = 5  # ダウンロード、文字起こし、要約、編集、アップロード
        self.current_step = 0
        self.status = "processing"
        self.details = ""

def cleanup_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ロギングの設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# VideoServiceのインスタンスを作成
video_service = VideoService()

class ProcessingStatus:
    def __init__(self):
        self.status = "processing"
        self.progress = 0
        self.message = "Initializing"
        self.error = None
        self.result = None  # 結果を保存するフィールドを追加

# 処理状態を保持する辞書
processing_statuses = {}

app = FastAPI(
    title="Video Summary API",
    description="YouTube動画の要約動画を作成するAPI",
    version="1.0.0"
)

class VideoRequest(BaseModel):
    url: HttpUrl
    max_duration: Optional[int] = 180  # 要約動画の最大時間（秒）

class VideoResponse(BaseModel):
    task_id: str
    status: str
    message: str

@app.get("/")
async def read_root():
    return {"message": "Welcome to Video Summary API"}

@app.post("/api/v1/videos/summarize", response_model=VideoResponse)
async def create_video_summary(video_request: VideoRequest, background_tasks: BackgroundTasks):
    try:
        logger.info(f"Received request to summarize video: {video_request.url}")
        
        # UUIDを使用してタスクIDを生成
        task_id = str(uuid.uuid4())
        
        # バックグラウンドタスクの登録（task_idを渡す）
        background_tasks.add_task(process_video, video_request, task_id)
        
        return VideoResponse(
            task_id=task_id,
            status="processing",
            message="Video processing started"
        )
    
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/videos/{task_id}", response_model=VideoResponse)
async def get_video_status(task_id: str):
    try:
        # タスクのステータスを取得する処理を実装
        # この例では仮の実装
        return VideoResponse(
            task_id=task_id,
            status="processing",
            message="Video is being processed"
        )
    
    except Exception as e:
        logger.error(f"Error getting task status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


async def process_video(video_request: VideoRequest, task_id: str):
    """
    動画処理のメインロジック
    """
    video_path = None
    status = ProcessingStatus()
    processing_statuses[task_id] = status
    
    try:
        # 処理状態の更新
        status = processing_statuses[task_id]
        status.message = "Downloading video"
        
        # 動画情報の取得
        video_info = await video_service.get_video_info(str(video_request.url))
        if not video_info:
            raise HTTPException(status_code=404, detail="Could not fetch video info")
            
        # 動画の長さチェック
        if video_info['duration'] < 10:  # 最小時間チェック
            raise ValueError("Video is too short (minimum 10 seconds required)")
            
        # 動画のダウンロード
        status.message = "Downloading video"
        video_path = await video_service.download_video(str(video_request.url))
        
        if not video_path:
            raise HTTPException(status_code=500, detail="Failed to download video")
            
        status.progress = 25
        status.message = "Download completed"
        
        # Whisperによる文字起こし
        status.message = "Transcribing video"
        transcription_result = await transcription_service.transcribe_video(video_path)
        
        if not transcription_result:
            raise HTTPException(status_code=500, detail="Failed to transcribe video")
        
        status.progress = 50
        status.message = "Transcription completed"

        # BARTによる要約
        status.message = "Generating summary"
        summary_result = await summarization_service.summarize_text(
            transcription_result['text'],
            transcription_result['segments'],
            target_duration=90  # 90秒を目標に
        )
        
        if not summary_result:
            status.status = "failed"
            status.error = "Failed to generate summary"
            return
            
        status.progress = 75
        status.message = "Summary generation completed"
        
        # 処理成功
        status.progress = 100
        status.status = "completed"
        status.message = "Processing completed"

        result = {
            "task_id": task_id,
            "status": "completed",
            "video_info": video_info,
            "transcription": transcription_result,
            "summary": summary_result
        }
        
        status.progress = 100
        status.status = "completed"
        status.message = "Processing completed"
        status.result = result  # 結果を保存
        
        return result
        
    except Exception as e:
        logger.error(f"Error in video processing: {str(e)}")
        if task_id in processing_statuses:
            status = processing_statuses[task_id]
            status.status = "failed"
            status.error = str(e)
            status.message = f"Processing failed: {str(e)}"
        raise
    finally:
        # 処理完了後に一時ファイルを削除
        if video_path:
            video_service.cleanup(video_path)
            # GPUメモリの解放
            cleanup_gpu_memory()

@app.get("/api/v1/videos/info")
async def get_video_info(url: HttpUrl):
    """
    動画の情報を取得するエンドポイント
    """
    try:
        video_info = await video_service.get_video_info(str(url))
        if not video_info:
            raise HTTPException(status_code=404, detail="Could not fetch video info")
        return VideoInfo(**video_info)
    except Exception as e:
        logger.error(f"Error fetching video info: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/api/v1/videos/{task_id}/status")
async def get_processing_status(task_id: str):
    if task_id not in processing_statuses:
        raise HTTPException(status_code=404, detail="Task not found")
        
    status = processing_statuses[task_id]
    response = {
        "task_id": task_id,
        "status": status.status,
        "progress": status.progress,
        "message": status.message,
        "error": status.error,
    }
    
    # 完了している場合は結果も含める
    if hasattr(status, 'result'):
        response["result"] = status.result
    
    return response