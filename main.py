#!/usr/bin/env python3
"""
Production AI Wedding Video Processing Service
Real computer vision analysis for wedding moments
"""

import os
import json
import asyncio
import logging
import tempfile
import shutil
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import subprocess

# Core AI dependencies
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import pipeline, AutoImageProcessor, AutoModelForImageClassification
import face_recognition
from sklearn.cluster import DBSCAN
import librosa
from PIL import Image, ImageEnhance

# Video processing
from moviepy.editor import VideoFileClip, concatenate_videoclips, CompositeVideoClip
import requests
from urllib.parse import urlparse

# Web framework
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class WeddingMoment:
    timestamp: float
    duration: float
    moment_type: str
    confidence: float
    description: str
    emotions: List[str]
    people_count: int
    audio_features: Dict
    visual_score: float = 0.0
    audio_score: float = 0.0

@dataclass
class ProcessingJob:
    job_id: str
    status: str
    progress: float
    detected_moments: List[WeddingMoment]
    total_videos: int = 0
    processed_videos: int = 0
    error_message: Optional[str] = None
    created_at: datetime = None
    completed_at: Optional[datetime] = None

class ProductionVideoProcessor:
    def __init__(self):
        """Initialize production AI models"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Initializing AI models on device: {self.device}")
        
        self.setup_models()
        self.wedding_classifier = self.create_wedding_classifier()
        
        # Processing settings
        self.min_face_size = (50, 50)
        self.emotion_threshold = 0.6
        self.audio_chunk_duration = 5.0  # seconds
        
    def setup_models(self):
        """Load production AI models"""
        try:
            # Emotion detection model
            self.emotion_processor = AutoImageProcessor.from_pretrained("j-hartmann/emotion-english-distilroberta-base")
            self.emotion_model = AutoModelForImageClassification.from_pretrained("j-hartmann/emotion-english-distilroberta-base")
            
            # Scene classification (general)
            self.scene_classifier = pipeline(
                "image-classification",
                model="google/vit-base-patch16-224",
                device=0 if self.device == "cuda" else -1
            )
            
            # Audio classification
            self.audio_classifier = pipeline(
                "audio-classification",
                model="MIT/ast-finetuned-audioset-10-10-0.4593",
                device=0 if self.device == "cuda" else -1
            )
            
            logger.info("✅ All AI models loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Error loading AI models: {e}")
            # Fallback to basic processing
            self.emotion_model = None
            self.scene_classifier = None
            self.audio_classifier = None
            
    def create_wedding_classifier(self) -> Dict:
        """Advanced wedding moment classification patterns"""
        return {
            'ceremony': {
                'visual_keywords': ['altar', 'aisle', 'white_dress', 'suit', 'bouquet', 'church', 'outdoor_wedding'],
                'audio_keywords': ['wedding_march', 'organ', 'classical', 'speech', 'vows'],
                'face_patterns': ['formal_pose', 'facing_each_other', 'standing'],
                'emotion_patterns': ['joy', 'love', 'surprise', 'tears'],
                'time_patterns': ['beginning', 'formal_moment'],
                'people_count_range': (2, 50)
            },
            'reception': {
                'visual_keywords': ['dance_floor', 'tables', 'cake', 'party', 'celebration'],
                'audio_keywords': ['music', 'dancing', 'party', 'celebration', 'DJ'],
                'face_patterns': ['dancing', 'laughing', 'group_interaction'],
                'emotion_patterns': ['joy', 'excitement', 'surprise'],
                'time_patterns': ['evening', 'party_atmosphere'],
                'people_count_range': (10, 200)
            },
            'emotional': {
                'visual_keywords': ['tears', 'hugging', 'kissing', 'close_up'],
                'audio_keywords': ['speech', 'crying', 'laughter', 'applause'],
                'face_patterns': ['crying', 'intense_emotion', 'close_contact'],
                'emotion_patterns': ['sadness', 'joy', 'love', 'surprise'],
                'time_patterns': ['any'],
                'people_count_range': (1, 20)
            },
            'group': {
                'visual_keywords': ['group_photo', 'family', 'friends', 'posed'],
                'audio_keywords': ['photographer', 'say_cheese', 'group_laughter'],
                'face_patterns': ['looking_at_camera', 'posed', 'grouped'],
                'emotion_patterns': ['joy', 'neutral', 'posed_smile'],
                'time_patterns': ['photo_session'],
                'people_count_range': (5, 100)
            }
        }

    async def download_video(self, video_url: str, job_id: str) -> str:
        """Download video from URL"""
        try:
            # Create unique filename
            url_hash = hashlib.md5(video_url.encode()).hexdigest()[:8]
            filename = f"video_{job_id}_{url_hash}.mp4"
            filepath = f"/tmp/{filename}"
            
            logger.info(f"Downloading video: {video_url[:100]}...")
            
            # Download with streaming
            response = requests.get(video_url, stream=True, timeout=300)
            response.raise_for_status()
            
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            file_size = os.path.getsize(filepath) / (1024 * 1024)  # MB
            logger.info(f"✅ Downloaded: {filename} ({file_size:.1f}MB)")
            
            return filepath
            
        except Exception as e:
            logger.error(f"❌ Error downloading video: {e}")
            raise

    async def process_video(self, video_url: str, job_id: str) -> List[WeddingMoment]:
        """Main production video processing pipeline"""
        moments = []
        temp_video_path = None
        
        try:
            # Download video
            temp_video_path = await self.download_video(video_url, job_id)
            
            # Load video with OpenCV
            cap = cv2.VideoCapture(temp_video_path)
            if not cap.isOpened():
                raise Exception(f"Could not open video: {temp_video_path}")
            
            # Get video properties
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0
            
            logger.info(f"📹 Processing video: {duration:.1f}s, {fps}fps, {total_frames} frames")
            
            # Extract audio for analysis
            audio_features = await self.extract_audio_features(temp_video_path)
            
            # Process video in intelligent chunks (every 2 seconds)
            frame_interval = max(1, fps * 2)  # Every 2 seconds
            current_frame = 0
            
            while current_frame < total_frames:
                # Read frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                timestamp = current_frame / fps
                
                # Analyze this frame
                frame_moments = await self.analyze_frame_comprehensive(
                    frame, timestamp, audio_features, job_id
                )
                
                moments.extend(frame_moments)
                
                # Update progress
                progress = (current_frame / total_frames) * 80  # 80% for frame analysis
                await self.update_job_progress(job_id, progress)
                
                current_frame += frame_interval
            
            cap.release()
            
            # Post-process moments
            moments = self.post_process_moments(moments)
            
            # Update final progress
            await self.update_job_progress(job_id, 90)
            
            logger.info(f"✅ Detected {len(moments)} wedding moments")
            return moments
            
        except Exception as e:
            logger.error(f"❌ Error processing video: {e}")
            raise
        finally:
            # Cleanup
            if temp_video_path and os.path.exists(temp_video_path):
                os.remove(temp_video_path)

    async def extract_audio_features(self, video_path: str) -> Dict:
        """Extract comprehensive audio features"""
        try:
            # Load audio with librosa
            y, sr = librosa.load(video_path, duration=300)  # Max 5 minutes for analysis
            
            # Extract features
            features = {
                'tempo': float(librosa.beat.tempo(y=y, sr=sr)[0]),
                'spectral_centroid': float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))),
                'zero_crossing_rate': float(np.mean(librosa.feature.zero_crossing_rate(y))),
                'mfcc': np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1).tolist(),
                'energy': float(np.mean(librosa.feature.rms(y=y))),
                'duration': float(len(y) / sr)
            }
            
            # Classify audio content
            if self.audio_classifier:
                # Sample audio for classification (first 30 seconds)
                sample_audio = y[:min(len(y), sr * 30)]
                try:
                    audio_classification = self.audio_classifier(sample_audio, sampling_rate=sr)
                    features['audio_type'] = audio_classification[0]['label'] if audio_classification else 'unknown'
                    features['audio_confidence'] = audio_classification[0]['score'] if audio_classification else 0.0
                except:
                    features['audio_type'] = 'unknown'
                    features['audio_confidence'] = 0.0
            
            return features
            
        except Exception as e:
            logger.warning(f"Audio extraction failed: {e}")
            return {'tempo': 120, 'energy': 0.5, 'audio_type': 'unknown'}

    async def analyze_frame_comprehensive(self, frame: np.ndarray, timestamp: float, 
                                        audio_features: Dict, job_id: str) -> List[WeddingMoment]:
        """Comprehensive frame analysis with real AI"""
        moments = []
        
        try:
            # Face analysis
            face_analysis = await self.analyze_faces_advanced(frame)
            
            # Scene analysis  
            scene_analysis = await self.analyze_scene_advanced(frame)
            
            # Composition analysis
            composition = self.analyze_composition_advanced(frame)
            
            # Wedding moment classification
            moment_type, confidence, description = self.classify_wedding_moment_advanced(
                face_analysis, scene_analysis, composition, audio_features, timestamp
            )
            
            # Create moment if confidence is high enough
            if confidence > 0.4:  # Stricter threshold for production
                moment = WeddingMoment(
                    timestamp=timestamp,
                    duration=2.0,  # Default duration
                    moment_type=moment_type,
                    confidence=confidence,
                    description=description,
                    emotions=face_analysis.get('emotions', []),
                    people_count=face_analysis.get('people_count', 0),
                    audio_features=audio_features,
                    visual_score=scene_analysis.get('visual_score', 0.0),
                    audio_score=audio_features.get('audio_confidence', 0.0)
                )
                moments.append(moment)
                
                logger.debug(f"🎬 Detected {moment_type} at {timestamp:.1f}s (confidence: {confidence:.2f})")
            
            return moments
            
        except Exception as e:
            logger.warning(f"Frame analysis error at {timestamp}s: {e}")
            return []

    async def analyze_faces_advanced(self, frame: np.ndarray) -> Dict:
        """Advanced face and emotion analysis"""
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Face detection with face_recognition (more accurate)
            face_locations = face_recognition.face_locations(rgb_frame, model="hog")
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            
            emotions = []
            face_qualities = []
            
            for (top, right, bottom, left) in face_locations:
                # Extract face region
                face_img = rgb_frame[top:bottom, left:right]
                
                if face_img.size == 0:
                    continue
                
                # Resize face for emotion analysis
                face_pil = Image.fromarray(face_img)
                face_pil = face_pil.resize((224, 224))
                
                # Emotion detection
                if self.emotion_model:
                    try:
                        emotion = await self.detect_emotion_advanced(face_pil)
                        emotions.append(emotion)
                    except:
                        emotions.append('neutral')
                else:
                    # Fallback: basic smile detection
                    emotion = self.detect_emotion_basic(face_img)
                    emotions.append(emotion)
                
                # Face quality assessment
                quality = self.assess_face_quality(face_img)
                face_qualities.append(quality)
            
            return {
                'people_count': len(face_locations),
                'face_locations': face_locations,
                'emotions': emotions,
                'face_encodings': face_encodings,
                'avg_face_quality': np.mean(face_qualities) if face_qualities else 0.0,
                'emotion_diversity': len(set(emotions)) if emotions else 0
            }
            
        except Exception as e:
            logger.warning(f"Face analysis error: {e}")
            return {'people_count': 0, 'emotions': [], 'face_encodings': []}

    async def detect_emotion_advanced(self, face_pil: Image) -> str:
        """Advanced emotion detection using transformer model"""
        try:
            # Prepare image
            inputs = self.emotion_processor(face_pil, return_tensors="pt")
            
            # Get prediction
            with torch.no_grad():
                outputs = self.emotion_model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                
            # Get top emotion
            emotion_idx = torch.argmax(predictions, dim=-1).item()
            confidence = torch.max(predictions).item()
            
            # Map to emotion labels (adjust based on your model)
            emotion_labels = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
            emotion = emotion_labels[emotion_idx] if emotion_idx < len(emotion_labels) else 'neutral'
            
            return emotion if confidence > self.emotion_threshold else 'neutral'
            
        except Exception as e:
            logger.warning(f"Advanced emotion detection failed: {e}")
            return 'neutral'

    def detect_emotion_basic(self, face_img: np.ndarray) -> str:
        """Fallback basic emotion detection"""
        try:
            gray = cv2.cvtColor(face_img, cv2.COLOR_RGB2GRAY)
            
            # Basic smile detection
            smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
            smiles = smile_cascade.detectMultiScale(gray, 1.8, 20)
            
            if len(smiles) > 0:
                return 'joy'
            
            # Basic eye detection for other emotions
            eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
            eyes = eye_cascade.detectMultiScale(gray)
            
            if len(eyes) >= 2:
                return 'neutral'
            else:
                return 'unknown'
                
        except:
            return 'neutral'

    def assess_face_quality(self, face_img: np.ndarray) -> float:
        """Assess face image quality for better moment scoring"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(face_img, cv2.COLOR_RGB2GRAY)
            
            # Calculate sharpness (Laplacian variance)
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Calculate brightness
            brightness = np.mean(gray) / 255.0
            
            # Size score (larger faces are generally better)
            size_score = min(face_img.shape[0] * face_img.shape[1] / 10000, 1.0)
            
            # Combine scores
            quality = (sharpness / 1000 * 0.4 + 
                      (1 - abs(brightness - 0.5) * 2) * 0.3 + 
                      size_score * 0.3)
            
            return min(quality, 1.0)
            
        except:
            return 0.5

    async def analyze_scene_advanced(self, frame: np.ndarray) -> Dict:
        """Advanced scene analysis with AI"""
        try:
            # Convert for scene classification
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_frame = Image.fromarray(rgb_frame)
            
            scene_info = {}
            
            # Scene classification
            if self.scene_classifier:
                try:
                    scene_results = self.scene_classifier(pil_frame)
                    scene_info['scene_type'] = scene_results[0]['label']
                    scene_info['scene_confidence'] = scene_results[0]['score']
                except:
                    scene_info['scene_type'] = 'unknown'
                    scene_info['scene_confidence'] = 0.0
            
            # Color analysis
            color_analysis = self.analyze_colors_advanced(frame)
            scene_info.update(color_analysis)
            
            # Lighting analysis
            lighting = self.analyze_lighting(frame)
            scene_info.update(lighting)
            
            # Wedding-specific object detection
            wedding_objects = self.detect_wedding_objects_advanced(frame)
            scene_info['wedding_objects'] = wedding_objects
            
            # Overall visual score
            scene_info['visual_score'] = self.calculate_visual_score(scene_info)
            
            return scene_info
            
        except Exception as e:
            logger.warning(f"Scene analysis error: {e}")
            return {'visual_score': 0.5}

    def analyze_colors_advanced(self, frame: np.ndarray) -> Dict:
        """Advanced color analysis for wedding detection"""
        try:
            # Convert to different color spaces
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            
            # Analyze dominant colors
            pixels = frame.reshape(-1, 3)
            
            # White/cream detection (important for wedding dresses)
            white_mask = np.all(pixels > 200, axis=1)
            white_percentage = np.sum(white_mask) / len(pixels)
            
            # Formal color detection (blacks, dark blues)
            formal_mask = np.all(pixels < 50, axis=1)
            formal_percentage = np.sum(formal_mask) / len(pixels)
            
            # Colorfulness measure
            colorfulness = self.calculate_colorfulness(frame)
            
            # Saturation analysis
            saturation = np.mean(hsv[:, :, 1]) / 255.0
            
            return {
                'white_presence': float(white_percentage),
                'formal_colors': float(formal_percentage),
                'colorfulness': colorfulness,
                'saturation': float(saturation),
                'brightness': float(np.mean(frame) / 255.0)
            }
            
        except:
            return {'white_presence': 0.0, 'formal_colors': 0.0, 'colorfulness': 0.5}

    def calculate_colorfulness(self, frame: np.ndarray) -> float:
        """Calculate colorfulness metric"""
        try:
            # Split into RGB channels
            (B, G, R) = cv2.split(frame.astype("float"))
            
            # Compute rg = R - G
            rg = np.absolute(R - G)
            
            # Compute yb = 0.5 * (R + G) - B
            yb = np.absolute(0.5 * (R + G) - B)
            
            # Compute the mean and standard deviation of both rg and yb
            (rbMean, rbStd) = (np.mean(rg), np.std(rg))
            (ybMean, ybStd) = (np.mean(yb), np.std(yb))
            
            # Combine the mean and standard deviation
            stdRoot = np.sqrt((rbStd ** 2) + (ybStd ** 2))
            meanRoot = np.sqrt((rbMean ** 2) + (ybMean ** 2))
            
            # Return the "colorfulness" metric
            return stdRoot + (0.3 * meanRoot)
            
        except:
            return 50.0  # Default value

    def analyze_lighting(self, frame: np.ndarray) -> Dict:
        """Analyze lighting conditions"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Overall brightness
            brightness = np.mean(gray) / 255.0
            
            # Contrast (standard deviation)
            contrast = np.std(gray) / 255.0
            
            # Histogram analysis
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            
            # Check for proper exposure (avoid over/under exposure)
            overexposed = np.sum(hist[240:]) / (frame.shape[0] * frame.shape[1])
            underexposed = np.sum(hist[:15]) / (frame.shape[0] * frame.shape[1])
            
            return {
                'brightness': float(brightness),
                'contrast': float(contrast),
                'overexposed': float(overexposed),
                'underexposed': float(underexposed),
                'lighting_quality': float(1.0 - overexposed - underexposed)
            }
            
        except:
            return {'brightness': 0.5, 'contrast': 0.5, 'lighting_quality': 0.5}

    def detect_wedding_objects_advanced(self, frame: np.ndarray) -> List[str]:
        """Detect wedding-specific objects using advanced techniques"""
        objects = []
        
        try:
            # Convert to grayscale for template matching
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Dress detection (look for large white regions)
            white_threshold = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)[1]
            contours, _ = cv2.findContours(white_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            large_white_areas = [c for c in contours if cv2.contourArea(c) > 5000]
            if large_white_areas:
                objects.append('wedding_dress')
            
            # Formal wear detection (dark suit regions)
            dark_threshold = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)[1]
            dark_contours, _ = cv2.findContours(dark_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            large_dark_areas = [c for c in dark_contours if cv2.contourArea(c) > 3000]
            if large_dark_areas:
                objects.append('formal_wear')
            
            # Flower detection (using color analysis)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Look for flower-like colors (various hues with good saturation)
            flower_mask = cv2.inRange(hsv, (0, 100, 100), (180, 255, 255))
            flower_pixels = cv2.countNonZero(flower_mask)
            
            if flower_pixels > 1000:
                objects.append('flowers')
            
            # Architecture detection (straight lines, geometric patterns)
            edges = cv2.Canny(gray, 50, 150)
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
            
            if lines is not None and len(lines) > 10:
                objects.append('architecture')
            
            return objects
            
        except:
            return []

    def analyze_composition_advanced(self, frame: np.ndarray) -> Dict:
        """Advanced composition analysis"""
        try:
            h, w = frame.shape[:2]
            
            # Rule of thirds analysis
            third_h, third_w = h // 3, w // 3
            
            # Analyze center vs edges
            center_region = frame[third_h:2*third_h, third_w:2*third_w]
            center_brightness = np.mean(center_region)
            overall_brightness = np.mean(frame)
            
            # Edge detection for composition analysis
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (h * w)
            
            # Symmetry analysis
            left_half = frame[:, :w//2]
            right_half = cv2.flip(frame[:, w//2:], 1)
            
            # Resize right half to match left half if needed
            if right_half.shape[1] != left_half.shape[1]:
                right_half = cv2.resize(right_half, (left_half.shape[1], left_half.shape[0]))
            
            symmetry = cv2.matchTemplate(left_half, right_half, cv2.TM_CCOEFF_NORMED)[0][0]
            
            return {
                'center_focus': float(center_brightness / (overall_brightness + 1e-6)),
                'edge_density': float(edge_density),
                'symmetry': float(abs(symmetry)),
                'aspect_ratio': float(w / h),
                'composition_score': float((edge_density + abs(symmetry)) / 2)
            }
            
        except:
            return {'center_focus': 1.0, 'composition_score': 0.5}

    def calculate_visual_score(self, scene_info: Dict) -> float:
        """Calculate overall visual quality score"""
        try:
            # Weight different factors
            lighting_score = scene_info.get('lighting_quality', 0.5) * 0.3
            composition_score = scene_info.get('composition_score', 0.5) * 0.2
            color_score = min(scene_info.get('colorfulness', 50) / 100, 1.0) * 0.2
            wedding_object_score = len(scene_info.get('wedding_objects', [])) / 4 * 0.3  # Max 4 objects
            
            total_score = lighting_score + composition_score + color_score + wedding_object_score
            return min(total_score, 1.0)
            
        except:
            return 0.5

    def classify_wedding_moment_advanced(self, face_analysis: Dict, scene_analysis: Dict, 
                                       composition: Dict, audio_features: Dict, 
                                       timestamp: float) -> Tuple[str, float, str]:
        """Advanced wedding moment classification using all AI features"""
        
        scores = {}
        
        # Get analysis data
        people_count = face_analysis.get('people_count', 0)
        emotions = face_analysis.get('emotions', [])
        white_presence = scene_analysis.get('white_presence', 0)
        wedding_objects = scene_analysis.get('wedding_objects', [])
        visual_score = scene_analysis.get('visual_score', 0)
        audio_type = audio_features.get('audio_type', 'unknown')
        tempo = audio_features.get('tempo', 120)
        
        # Ceremony detection (advanced)
        ceremony_score = 0.0
        if white_presence > 0.15:  # Wedding dress present
            ceremony_score += 0.3
        if 'wedding_dress' in wedding_objects or 'formal_wear' in wedding_objects:
            ceremony_score += 0.25
        if people_count >= 2 and people_count <= 20:  # Intimate ceremony size
            ceremony_score += 0.2
        if 'joy' in emotions or 'surprise' in emotions:
            ceremony_score += 0.15
        if audio_type in ['speech', 'music'] or tempo < 100:  # Slower, formal music
            ceremony_score += 0.1
        
        scores['ceremony'] = ceremony_score
        
        # Reception detection (advanced)
        reception_score = 0.0
        if people_count > 10:  # Party atmosphere
            reception_score += 0.3
        if tempo > 100:  # Dance music
            reception_score += 0.2
        if emotions.count('joy') > 2:  # Multiple happy people
            reception_score += 0.2
        if 'flowers' in wedding_objects:
            reception_score += 0.15
        if scene_analysis.get('colorfulness', 0) > 60:  # Colorful party scene
            reception_score += 0.15
        
        scores['reception'] = reception_score
        
        # Emotional moment detection (advanced)
        emotional_score = 0.0
        emotion_diversity = face_analysis.get('emotion_diversity', 0)
        
        if 'sadness' in emotions or 'surprise' in emotions:  # Tears or surprise
            emotional_score += 0.4
        if emotion_diversity >= 2:  # Mix of emotions
            emotional_score += 0.2
        if people_count <= 5:  # Intimate moment
            emotional_score += 0.2
        if composition.get('center_focus', 0) > 1.2:  # Close-up shot
            emotional_score += 0.15
        if audio_type == 'speech':  # Likely vows or speech
            emotional_score += 0.05
        
        scores['emotional'] = emotional_score
        
        # Group moment detection (advanced)
        group_score = 0.0
        if people_count > 5:
            group_score += 0.4
        if composition.get('symmetry', 0) > 0.7:  # Posed group photo
            group_score += 0.3
        if scene_analysis.get('lighting_quality', 0) > 0.7:  # Good lighting for photos
            group_score += 0.2
        if face_analysis.get('avg_face_quality', 0) > 0.6:  # Clear faces
            group_score += 0.1
        
        scores['group'] = group_score
        
        # Find best classification
        if not scores or max(scores.values()) < 0.4:
            return 'other', 0.0, "General wedding moment"
        
        best_moment = max(scores.items(), key=lambda x: x[1])
        moment_type, confidence = best_moment
        
        # Generate detailed description
        description = self.generate_detailed_description(
            moment_type, people_count, emotions, wedding_objects, confidence
        )
        
        return moment_type, confidence, description

    def generate_detailed_description(self, moment_type: str, people_count: int, 
                                    emotions: List[str], wedding_objects: List[str], 
                                    confidence: float) -> str:
        """Generate detailed, human-readable descriptions"""
        
        emotion_text = ", ".join(set(emotions[:3])) if emotions else "neutral"
        object_text = ", ".join(wedding_objects[:2]) if wedding_objects else ""
        
        descriptions = {
            'ceremony': f"Ceremony moment with {people_count} people showing {emotion_text} emotions" + 
                       (f" (detected: {object_text})" if object_text else ""),
            'reception': f"Reception celebration with {people_count} guests, emotions: {emotion_text}" + 
                        (f" (scene includes: {object_text})" if object_text else ""),
            'emotional': f"Emotional moment featuring {emotion_text} expressions from {people_count} people" + 
                        (f" (context: {object_text})" if object_text else ""),
            'group': f"Group interaction with {people_count} people" + 
                    (f", emotions: {emotion_text}" if emotions else "") + 
                    (f" (setting: {object_text})" if object_text else "")
        }
        
        base_description = descriptions.get(moment_type, f"{moment_type.title()} wedding moment")
        return f"{base_description} (confidence: {confidence:.0%})"

    def post_process_moments(self, moments: List[WeddingMoment]) -> List[WeddingMoment]:
        """Advanced post-processing of detected moments"""
        if not moments:
            return moments
        
        # Sort by timestamp
        moments.sort(key=lambda m: m.timestamp)
        
        # Merge similar adjacent moments
        merged_moments = []
        current_moment = moments[0]
        
        for next_moment in moments[1:]:
            # Check if moments should be merged
            time_gap = next_moment.timestamp - (current_moment.timestamp + current_moment.duration)
            
            should_merge = (
                time_gap < 10.0 and  # Within 10 seconds
                next_moment.moment_type == current_moment.moment_type and
                abs(next_moment.confidence - current_moment.confidence) < 0.3
            )
            
            if should_merge:
                # Merge moments
                current_moment.duration = (
                    next_moment.timestamp + next_moment.duration - current_moment.timestamp
                )
                current_moment.confidence = max(current_moment.confidence, next_moment.confidence)
                current_moment.people_count = max(current_moment.people_count, next_moment.people_count)
                
                # Merge emotions
                current_moment.emotions = list(set(current_moment.emotions + next_moment.emotions))
                
                # Update scores
                current_moment.visual_score = max(current_moment.visual_score, next_moment.visual_score)
                current_moment.audio_score = max(current_moment.audio_score, next_moment.audio_score)
                
            else:
                merged_moments.append(current_moment)
                current_moment = next_moment
        
        merged_moments.append(current_moment)
        
        # Filter by quality (keep only high-confidence moments)
        quality_moments = [
            m for m in merged_moments 
            if m.confidence > 0.5 or m.visual_score > 0.7 or m.audio_score > 0.7
        ]
        
        # Ensure we have a good distribution of moment types
        final_moments = self.balance_moment_types(quality_moments)
        
        logger.info(f"📊 Post-processing: {len(moments)} → {len(merged_moments)} → {len(final_moments)}")
        
        return final_moments

    def balance_moment_types(self, moments: List[WeddingMoment]) -> List[WeddingMoment]:
        """Ensure balanced representation of different moment types"""
        if not moments:
            return moments
        
        # Group by moment type
        moment_groups = {}
        for moment in moments:
            if moment.moment_type not in moment_groups:
                moment_groups[moment.moment_type] = []
            moment_groups[moment.moment_type].append(moment)
        
        # Sort each group by confidence and select top moments
        balanced_moments = []
        max_per_type = max(3, len(moments) // 4)  # At least 3, or 1/4 of total
        
        for moment_type, group in moment_groups.items():
            # Sort by combined score
            group.sort(key=lambda m: (m.confidence + m.visual_score + m.audio_score) / 3, reverse=True)
            balanced_moments.extend(group[:max_per_type])
        
        # Sort final list by timestamp
        balanced_moments.sort(key=lambda m: m.timestamp)
        
        return balanced_moments

    async def create_highlight_reel_advanced(self, video_paths: List[str], moments: List[WeddingMoment], 
                                           output_path: str, preferences: Dict) -> str:
        """Create professional highlight reel with real video editing"""
        try:
            if not moments:
                raise Exception("No moments detected for highlight reel")
            
            max_duration = preferences.get('duration', 180)  # Default 3 minutes
            style = preferences.get('style', 'cinematic')
            
            logger.info(f"🎬 Creating {style} highlight reel ({max_duration}s) from {len(moments)} moments")
            
            # Select best moments for highlight reel
            selected_moments = self.select_moments_for_reel(moments, max_duration, style)
            
            # Create video clips from selected moments
            clips = []
            total_duration = 0
            
            for moment in selected_moments:
                try:
                    # Find the source video (simplified - use first video for now)
                    source_video = video_paths[0] if video_paths else None
                    if not source_video or not os.path.exists(source_video):
                        continue
                    
                    # Create clip from moment
                    clip_duration = min(moment.duration, max_duration - total_duration)
                    if clip_duration <= 0:
                        break
                    
                    # Load video clip
                    video_clip = VideoFileClip(source_video)
                    moment_clip = video_clip.subclip(moment.timestamp, moment.timestamp + clip_duration)
                    
                    # Apply style-specific effects
                    styled_clip = self.apply_video_style(moment_clip, style, moment.moment_type)
                    
                    clips.append(styled_clip)
                    total_duration += clip_duration
                    
                    video_clip.close()
                    
                    if total_duration >= max_duration:
                        break
                        
                except Exception as e:
                    logger.warning(f"Failed to create clip for moment at {moment.timestamp}s: {e}")
                    continue
            
            if not clips:
                raise Exception("No video clips could be created")
            
            # Concatenate clips with transitions
            final_video = self.add_transitions(clips, style)
            
            # Add background music if specified
            if preferences.get('add_music', False):
                final_video = self.add_background_music(final_video, style)
            
            # Export final video
            final_video.write_videofile(
                output_path,
                codec='libx264',
                audio_codec='aac',
                temp_audiofile='temp-audio.m4a',
                remove_temp=True,
                fps=24,
                preset='medium'
            )
            
            # Cleanup
            for clip in clips:
                clip.close()
            final_video.close()
            
            logger.info(f"✅ Highlight reel created: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"❌ Error creating highlight reel: {e}")
            raise

    def select_moments_for_reel(self, moments: List[WeddingMoment], max_duration: int, style: str) -> List[WeddingMoment]:
        """Intelligently select moments for highlight reel based on style"""
        
        # Style-specific preferences
        style_weights = {
            'cinematic': {'ceremony': 0.4, 'emotional': 0.3, 'reception': 0.2, 'group': 0.1},
            'documentary': {'ceremony': 0.3, 'reception': 0.3, 'emotional': 0.2, 'group': 0.2},
            'romantic': {'ceremony': 0.5, 'emotional': 0.4, 'reception': 0.1, 'group': 0.0},
            'energetic': {'reception': 0.5, 'group': 0.3, 'ceremony': 0.1, 'emotional': 0.1}
        }
        
        weights = style_weights.get(style, style_weights['cinematic'])
        
        # Score moments based on style preferences
        for moment in moments:
            style_bonus = weights.get(moment.moment_type, 0.1)
            moment.style_score = (moment.confidence + moment.visual_score + moment.audio_score) / 3 + style_bonus
        
        # Sort by style score
        moments.sort(key=lambda m: m.style_score, reverse=True)
        
        # Select moments within duration limit
        selected = []
        total_duration = 0
        
        for moment in moments:
            clip_duration = min(moment.duration, 15)  # Max 15 seconds per clip
            if total_duration + clip_duration <= max_duration:
                selected.append(moment)
                total_duration += clip_duration
            
            if total_duration >= max_duration * 0.9:  # 90% of target duration
                break
        
        # Sort selected moments by timestamp for chronological order
        selected.sort(key=lambda m: m.timestamp)
        
        return selected

    def apply_video_style(self, clip, style: str, moment_type: str):
        """Apply style-specific video effects"""
        try:
            if style == 'cinematic':
                # Cinematic: slight slow motion, color grading
                if moment_type == 'emotional':
                    clip = clip.fx(lambda c: c.speedx(0.8))  # Slight slow motion
                
            elif style == 'energetic':
                # Energetic: faster cuts, higher saturation
                if moment_type == 'reception':
                    clip = clip.fx(lambda c: c.speedx(1.2))  # Slight speed up
            
            # Add fade in/out
            clip = clip.fadein(0.5).fadeout(0.5)
            
            return clip
            
        except Exception as e:
            logger.warning(f"Failed to apply style effects: {e}")
            return clip

    def add_transitions(self, clips, style: str):
        """Add transitions between clips"""
        try:
            if len(clips) <= 1:
                return clips[0] if clips else None
            
            # Simple crossfade transitions
            transition_duration = 0.5
            
            final_clips = [clips[0]]
            
            for i in range(1, len(clips)):
                # Create crossfade transition
                prev_clip = final_clips[-1]
                current_clip = clips[i]
                
                # Crossfade
                transition = CompositeVideoClip([
                    prev_clip.fadeout(transition_duration),
                    current_clip.set_start(prev_clip.duration - transition_duration).fadein(transition_duration)
                ])
                
                final_clips[-1] = transition
            
            return concatenate_videoclips(final_clips, method="compose")
            
        except Exception as e:
            logger.warning(f"Failed to add transitions: {e}")
            return concatenate_videoclips(clips, method="compose")

    def add_background_music(self, video_clip, style: str):
        """Add background music based on style"""
        try:
            # This would integrate with a music library
            # For now, just return the original clip
            logger.info("Background music feature not implemented yet")
            return video_clip
            
        except Exception as e:
            logger.warning(f"Failed to add background music: {e}")
            return video_clip

    async def update_job_progress(self, job_id: str, progress: float):
        """Update job progress"""
        if job_id in processing_jobs:
            processing_jobs[job_id].progress = progress
        logger.info(f"📊 Job {job_id} progress: {progress:.1f}%")

# FastAPI Application
app = FastAPI(
    title="Production Wedding Video AI Service", 
    version="2.0.0",
    description="Advanced AI-powered wedding video analysis and highlight reel creation"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global processor instance
processor = ProductionVideoProcessor()

# Processing jobs storage
processing_jobs: Dict[str, ProcessingJob] = {}

class ProcessingRequest(BaseModel):
    video_urls: List[str]
    preferences: Dict = {}

class ProcessingResponse(BaseModel):
    job_id: str
    status: str
    message: str

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Wedding Video AI",
        "version": "2.0.0",
        "status": "running",
        "tier": "production",
        "features": ["face_detection", "emotion_analysis", "scene_classification", "highlight_creation"],
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "tier": "production",
        "gpu_available": torch.cuda.is_available(),
        "models_loaded": processor.emotion_model is not None
    }

@app.post("/process-wedding-videos", response_model=ProcessingResponse)
async def process_wedding_videos(request: ProcessingRequest, background_tasks: BackgroundTasks):
    """Start production wedding video processing"""
    job_id = f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(processing_jobs)}"
    
    # Create processing job
    job = ProcessingJob(
        job_id=job_id,
        status="queued",
        progress=0.0,
        detected_moments=[],
        total_videos=len(request.video_urls),
        created_at=datetime.now()
    )
    
    processing_jobs[job_id] = job
    
    # Start background processing
    background_tasks.add_task(process_videos_background_advanced, job_id, request.video_urls, request.preferences)
    
    return ProcessingResponse(
        job_id=job_id,
        status="queued",
        message=f"Advanced AI processing started for {len(request.video_urls)} videos"
    )

@app.get("/job-status/{job_id}")
async def get_job_status(job_id: str):
    """Get processing job status"""
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = processing_jobs[job_id]
    return {
        "job_id": job_id,
        "status": job.status,
        "progress": job.progress,
        "total_videos": job.total_videos,
        "processed_videos": job.processed_videos,
        "detected_moments": [asdict(moment) for moment in job.detected_moments],
        "error_message": job.error_message,
        "created_at": job.created_at.isoformat() if job.created_at else None,
        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
        "tier": "production"
    }

@app.get("/download-highlight/{job_id}")
async def download_highlight_reel(job_id: str):
    """Download the generated highlight reel"""
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = processing_jobs[job_id]
    if job.status != "completed":
        raise HTTPException(status_code=400, detail="Job not completed")
    
    output_path = f"/tmp/highlight_{job_id}.mp4"
    if os.path.exists(output_path):
        return FileResponse(
            output_path, 
            media_type="video/mp4", 
            filename=f"wedding_highlight_{job_id}.mp4"
        )
    else:
        raise HTTPException(status_code=404, detail="Highlight reel not found")

async def process_videos_background_advanced(job_id: str, video_urls: List[str], preferences: Dict):
    """Advanced background processing with real AI"""
    temp_video_paths = []
    
    try:
        job = processing_jobs[job_id]
        job.status = "processing"
        
        logger.info(f"🚀 Starting advanced AI processing for job {job_id}")
        
        all_moments = []
        
        # Process each video
        for i, video_url in enumerate(video_urls):
            logger.info(f"📹 Processing video {i+1}/{len(video_urls)}")
            
            # Process video with advanced AI
            moments = await processor.process_video(video_url, job_id)
            all_moments.extend(moments)
            
            job.processed_videos = i + 1
            job.progress = (job.processed_videos / job.total_videos) * 85  # 85% for processing
            
            logger.info(f"✅ Video {i+1} complete: {len(moments)} moments detected")
        
        # Create highlight reel if we have moments
        if all_moments:
            logger.info(f"🎬 Creating highlight reel from {len(all_moments)} moments")
            
            # Download videos again for highlight creation (needed for editing)
            for i, video_url in enumerate(video_urls):
                temp_path = await processor.download_video(video_url, f"{job_id}_reel_{i}")
                temp_video_paths.append(temp_path)
            
            output_path = f"/tmp/highlight_{job_id}.mp4"
            await processor.create_highlight_reel_advanced(
                temp_video_paths, all_moments, output_path, preferences
            )
            
            job.progress = 100.0
        
        # Complete job
        job.status = "completed"
        job.detected_moments = all_moments
        job.completed_at = datetime.now()
        
        logger.info(f"🎉 Job {job_id} completed: {len(all_moments)} moments, highlight reel created")
        
    except Exception as e:
        logger.error(f"❌ Error in advanced processing job {job_id}: {e}")
        job.status = "failed"
        job.error_message = str(e)
    
    finally:
        # Cleanup temporary video files
        for temp_path in temp_video_paths:
            if os.path.exists(temp_path):
                os.remove(temp_path)

# Get port from environment
port = int(os.environ.get("PORT", 8000))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=port)
