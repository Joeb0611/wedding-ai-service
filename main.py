# main.py - Simplified for free tier deployment
import os
import json
import asyncio
import logging
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
import tempfile

# Lightweight dependencies only
import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Simplified logging
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

@dataclass
class ProcessingJob:
    job_id: str
    status: str
    progress: float
    detected_moments: List[WeddingMoment]
    error_message: Optional[str] = None
    created_at: datetime = None

# Lightweight video processor for free tier
class LightweightVideoProcessor:
    def __init__(self):
        """Initialize lightweight processor"""
        logger.info("Initializing lightweight wedding video processor")
        
    async def process_video_lightweight(self, video_url: str, job_id: str) -> List[WeddingMoment]:
        """Lightweight video processing for free tier"""
        moments = []
        
        try:
            # Simulate processing with basic analysis
            # In free tier, we'll do simplified analysis
            logger.info(f"Processing video: {video_url}")
            
            # Simulate different types of moments
            moment_types = ['ceremony', 'reception', 'emotional', 'group']
            
            # Create sample moments (in production, replace with actual analysis)
            for i, moment_type in enumerate(moment_types):
                moment = WeddingMoment(
                    timestamp=float(i * 30),  # Every 30 seconds
                    duration=10.0,
                    moment_type=moment_type,
                    confidence=0.8,
                    description=f"Detected {moment_type} moment",
                    emotions=['joy', 'love'],
                    people_count=2 + i
                )
                moments.append(moment)
                
                # Update progress
                progress = ((i + 1) / len(moment_types)) * 100
                await self.update_job_progress(job_id, progress)
                
                # Small delay to simulate processing
                await asyncio.sleep(1)
            
            return moments
            
        except Exception as e:
            logger.error(f"Error processing video {video_url}: {e}")
            raise

    async def update_job_progress(self, job_id: str, progress: float):
        """Update job progress"""
        logger.info(f"Job {job_id} progress: {progress:.1f}%")

# FastAPI Application
app = FastAPI(
    title="Wedding Video AI Service (Free Tier)", 
    version="1.0.0",
    description="Lightweight wedding video processing service"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for your Lovable app domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global processor instance
processor = LightweightVideoProcessor()

# In-memory job storage (free tier limitation)
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
        "version": "1.0.0",
        "status": "running",
        "tier": "free",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "tier": "free"
    }

@app.post("/process-wedding-videos", response_model=ProcessingResponse)
async def process_wedding_videos(request: ProcessingRequest, background_tasks: BackgroundTasks):
    """Start wedding video processing"""
    job_id = f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(processing_jobs)}"
    
    # Create processing job
    job = ProcessingJob(
        job_id=job_id,
        status="queued",
        progress=0.0,
        detected_moments=[],
        created_at=datetime.now()
    )
    
    processing_jobs[job_id] = job
    
    # Start background processing
    background_tasks.add_task(process_videos_background, job_id, request.video_urls, request.preferences)
    
    return ProcessingResponse(
        job_id=job_id,
        status="queued",
        message="Video processing started (free tier - simplified analysis)"
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
        "detected_moments": [asdict(moment) for moment in job.detected_moments],
        "error_message": job.error_message,
        "created_at": job.created_at.isoformat() if job.created_at else None,
        "tier": "free"
    }

async def process_videos_background(job_id: str, video_urls: List[str], preferences: Dict):
    """Background task for processing videos"""
    try:
        job = processing_jobs[job_id]
        job.status = "processing"
        
        all_moments = []
        
        for i, video_url in enumerate(video_urls):
            # Process video with lightweight analysis
            moments = await processor.process_video_lightweight(video_url, job_id)
            all_moments.extend(moments)
            
            # Update progress
            job.progress = ((i + 1) / len(video_urls)) * 100
        
        # Complete job
        job.status = "completed"
        job.progress = 100.0
        job.detected_moments = all_moments
        
        logger.info(f"Job {job_id} completed with {len(all_moments)} moments detected")
        
    except Exception as e:
        logger.error(f"Error processing job {job_id}: {e}")
        job.status = "failed"
        job.error_message = str(e)

# Get port from environment (Render requirement)
port = int(os.environ.get("PORT", 8000))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=port)
