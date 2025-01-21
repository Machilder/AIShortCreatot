import logging
from typing import Optional, List, Dict
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
from app.config import Config
import os

logger = logging.getLogger(__name__)

class SummarizationService:
    def __init__(self, config: Config):
        self.config = config
        logger.info(f"Initializing Summarization Service with device: {config.device}")
        
        # よりアクセスしやすい公開モデルを使用
        # self.model_name = "google/mt5-small"  # 変更：より安定したモデルを使用
        self.model_name = "rinna/japanese-gpt-1b"
        
        # モデルの保存先ディレクトリを設定
        self.model_dir = os.path.join(config.base_dir, "models")
        os.makedirs(self.model_dir, exist_ok=True)
        
        try:
            # ローカルからモデルをロード
            logger.info("Attempting to load model from local directory")
            self.tokenizer = T5Tokenizer.from_pretrained(self.model_dir)
            self.model = T5ForConditionalGeneration.from_pretrained(self.model_dir)
        except:
            # ローカルにない場合はダウンロード
            logger.info("Downloading model from Hugging Face")
            self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
            self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)
            # ローカルに保存
            self.tokenizer.save_pretrained(self.model_dir)
            self.model.save_pretrained(self.model_dir)
        
        # CPU環境での最適化設定
        if config.device == "cpu":
            self.batch_size = 1
            self.num_beams = 2
            torch.set_num_threads(config.max_workers)
        else:
            self.batch_size = config.batch_size
            self.num_beams = 4
        
        # デバイスの設定
        self.device = torch.device(config.device)
        self.model.to(self.device)
        
        logger.info(f"Initialized Summarization model on device: {self.device} "f"with batch_size: {self.batch_size}")

    async def summarize_text(self, text: str, segments: list, target_duration: int = 90) -> Optional[Dict]:
        """
        テキストを要約し、指定された時間内に収まるセグメントを選択する

        Args:
            text: 要約する元のテキスト
            segments: 元の字幕セグメントリスト
            target_duration: 目標とする動画の長さ（秒）
        """
        try:
            logger.info(f"Starting text summarization for {target_duration}s video")
            
            # まず全体を要約
            summary = await self._generate_summary(text)
            
            # 重要なセグメントを選択
            selected_segments = self._select_important_segments(
                summary, 
                segments, 
                target_duration
            )
            
            result = {
                'summary': summary,
                'original_length': len(text),
                'summary_length': len(summary),
                'selected_segments': selected_segments,
                'total_duration': sum(s['end'] - s['start'] for s in selected_segments),
                'original_duration': segments[-1]['end'] - segments[0]['start']
            }
            
            logger.info(f"Summarization completed: {result['total_duration']:.1f}s selected from {result['original_duration']:.1f}s")
            return result

        except Exception as e:
            logger.error(f"Error during summarization: {str(e)}", exc_info=True)
            return None

    async def _generate_summary(self, text: str) -> str:
        """テキストの要約を生成"""
        inputs = self.tokenizer(
            f"次の文章を要約してください：{text}",
            max_length=1024,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            output_ids = self.model.generate(
                inputs["input_ids"],
                max_length=150,
                min_length=50,
                num_beams=4,
                early_stopping=True
            )
        
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

    def _select_important_segments(
        self, 
        summary: str, 
        segments: list, 
        target_duration: int,
        margin: float = 0.1  # 10%のマージン
    ) -> List[Dict]:
        """
        要約に基づいて重要なセグメントを選択

        Args:
            summary: 生成された要約文
            segments: 元の字幕セグメント
            target_duration: 目標とする動画の長さ（秒）
            margin: 許容される時間のマージン
        """
        max_duration = target_duration * (1 + margin)  # マージンを含めた最大時間
        min_duration = target_duration * (1 - margin)  # 最小時間
        
        # 各セグメントにスコアを付与
        scored_segments = []
        for segment in segments:
            score = self._calculate_segment_importance(
                segment['text'], 
                summary,
                segment['end'] - segment['start']
            )
            scored_segments.append({
                **segment,
                'score': score
            })
        
        # スコアで降順ソート
        scored_segments.sort(key=lambda x: x['score'], reverse=True)
        
        # 時間内に収まるように選択
        selected = []
        total_duration = 0
        
        for segment in scored_segments:
            duration = segment['end'] - segment['start']
            if total_duration + duration <= max_duration:
                selected.append(segment)
                total_duration += duration
            if total_duration >= min_duration:
                break
        
        # 時系列順にソート
        selected.sort(key=lambda x: x['start'])
        
        return selected

    def _calculate_segment_importance(self, segment_text: str, summary: str, duration: float) -> float:
        """
        セグメントの重要度を計算
        - 要約文との類似度
        - セグメントの長さ
        - キーワードの存在
        などを考慮
        """
        # 簡単な実装例：要約文に含まれる単語の割合をスコアとする
        summary_words = set(summary.split())
        segment_words = set(segment_text.split())
        common_words = summary_words & segment_words
        
        base_score = len(common_words) / len(segment_words) if segment_words else 0
        
        # 極端に短いセグメントは優先度を下げる
        if duration < 1.0:
            base_score *= 0.5
        
        return base_score

    def _split_text(self, text: str, max_chunk_size: int = 1000) -> List[str]:
        """
        長いテキストを処理可能なサイズのチャンクに分割
        """
        # 文単位で分割
        sentences = text.split('。')
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_size = len(sentence)
            if current_size + sentence_size > max_chunk_size:
                if current_chunk:
                    chunks.append('。'.join(current_chunk) + '。')
                current_chunk = [sentence]
                current_size = sentence_size
            else:
                current_chunk.append(sentence)
                current_size += sentence_size
        
        if current_chunk:
            chunks.append('。'.join(current_chunk) + '。')
        
        return chunks

    def cleanup(self):
        """
        モデルのクリーンアップ処理
        """
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.model = None
            self.tokenizer = None
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")