"""
Avatar Streaming Service - Main Application
FastAPI application with cold model loading and avatar registration
"""

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn

from app.config.settings import Settings
from app.config.logging_config import setup_logging
from app.core.model_loader import ColdModelLoader
from app.core.avatar_registrar import ColdAvatarRegistrar
from app.core.face_cache_manager import FaceCacheManager
from app.core.resource_manager import GPUResourceManager
from app.services.wav2lip_service import PreloadedWav2LipService
from app.services.avatar_service import AvatarManagementService
from app.services.tts_service import OptimizedTTSService
from app.services.rag_service import CachedRAGService
from app.streaming.websocket_handler import BinaryWebSocketHandler
from app.utils.error_handler import GlobalErrorHandler
from app.models.avatar_models import AvatarRegistrationRequest, AvatarRegistrationResponse
from app.models.response_models import ServiceHealthResponse, AvatarProcessingResponse
from app.middleware.rate_limiter import RateLimitingMiddleware
from app.middleware.security_middleware import SecurityMiddleware
from app.middleware.monitoring_middleware import MonitoringMiddleware

# Global instances
model_loader: ColdModelLoader = None
avatar_registrar: ColdAvatarRegistrar = None
face_cache_manager: FaceCacheManager = None
resource_manager: GPUResourceManager = None
services: Dict[str, Any] = {}
websocket_handler: BinaryWebSocketHandler = None
error_handler: GlobalErrorHandler = None

# Settings
settings = Settings()

class AvatarApplication:
    """Main application class for Avatar Streaming Service"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.initialization_start_time = None
        self.services_ready = False
        
    async def initialize_services(self):
        """Cold load all models and services"""
        global model_loader, avatar_registrar, face_cache_manager, resource_manager, services, websocket_handler, error_handler
        
        self.initialization_start_time = time.time()
        self.logger.info("🚀 Starting Avatar Service initialization...")
        
        try:
            # Step 1: Validate system requirements
            await self.validate_system_requirements()
            
            # Step 2: Initialize error handler
            error_handler = GlobalErrorHandler()
            self.logger.info("✅ Error handler initialized")
            
            # Step 3: Initialize model loader and cold load models
            if settings.preload_models:
                self.logger.info("❄️ Starting cold model loading...")
                model_loader = ColdModelLoader()
                await model_loader.load_all_models()
                self.logger.info("✅ Models loaded into GPU memory")
            
            # Step 4: Initialize face cache manager
            face_cache_manager = FaceCacheManager()
            await face_cache_manager.initialize()
            self.logger.info("✅ Face cache manager initialized")
            
            # Step 5: Initialize avatar registrar
            avatar_registrar = ColdAvatarRegistrar(
                face_detector=model_loader.get_model_instance("face_detector"),
                face_cache_manager=face_cache_manager
            )
            self.logger.info("✅ Avatar registrar initialized")
            
            # Step 6: Initialize resource manager
            resource_manager = GPUResourceManager(
                model_instances=model_loader.loaded_models,
                avatar_cache=face_cache_manager
            )
            self.logger.info("✅ GPU resource manager initialized")
            
            # Step 7: Initialize services with pre-loaded models
            await self.initialize_processing_services()
            
            # Step 8: Initialize avatar cache if enabled
            if settings.preload_avatars:
                await self.initialize_avatar_cache()
            
            # Step 9: Initialize WebSocket handler
            websocket_handler = BinaryWebSocketHandler(
                services=services,
                resource_manager=resource_manager
            )
            self.logger.info("✅ WebSocket handler initialized")
            
            self.services_ready = True
            initialization_time = time.time() - self.initialization_start_time
            self.logger.info(f"🎉 Avatar Service initialization completed in {initialization_time:.2f} seconds")
            
        except Exception as e:
            self.logger.error(f"❌ Service initialization failed: {str(e)}")
            raise
    
    async def initialize_processing_services(self):
        """Initialize all processing services with pre-loaded models"""
        global services
        
        # Wav2Lip service with pre-loaded models
        services["wav2lip"] = PreloadedWav2LipService()
        await services["wav2lip"].initialize_with_preloaded_models(
            model_loader=model_loader,
            face_cache=face_cache_manager
        )
        
        # Avatar management service
        services["avatar"] = AvatarManagementService(
            avatar_registrar=avatar_registrar,
            face_cache_manager=face_cache_manager
        )
        
        # TTS service with Persian optimization
        services["tts"] = OptimizedTTSService()
        
        # RAG service with MCP integration
        services["rag"] = CachedRAGService()
        
        self.logger.info("✅ All processing services initialized with pre-loaded models")
    
    async def initialize_avatar_cache(self):
        """Pre-load avatar cache for immediate access"""
        try:
            # Get list of frequently used avatars
            avatar_list = await services["avatar"].get_avatar_list()
            avatar_ids = [avatar.avatar_id for avatar in avatar_list[:10]]  # Top 10 avatars
            
            if avatar_ids:
                warmup_report = await services["avatar"].warm_up_avatar_cache(avatar_ids)
                self.logger.info(f"✅ Avatar cache warmed up: {warmup_report}")
            else:
                self.logger.info("ℹ️ No avatars found for cache warmup")
                
        except Exception as e:
            self.logger.warning(f"⚠️ Avatar cache warmup failed: {str(e)}")
    
    async def validate_system_requirements(self):
        """Validate GPU and system requirements"""
        import onnxruntime as ort
        import psutil
        
        # Check GPU availability
        if ort.get_device() != 'GPU':
            self.logger.warning("⚠️ GPU not available, performance will be degraded")
        
        # Check memory availability
        memory = psutil.virtual_memory()
        if memory.available < 8 * 1024 * 1024 * 1024:  # 8GB
            raise RuntimeError("Insufficient memory available")
        
        self.logger.info("✅ System requirements validated")

# Create application instance
app_instance = AvatarApplication()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    await app_instance.initialize_services()
    yield
    # Shutdown
    logging.info("🛑 Shutting down Avatar Service...")
    # Add cleanup logic here

# Create FastAPI application
app = FastAPI(
    title="Avatar Streaming Service",
    description="Dockerized AI avatar system with cold model loading and avatar registration",
    version="1.0.0",
    lifespan=lifespan
)

# Add middleware in correct order (outer to inner)
# 1. Monitoring middleware (outermost - logs everything)
app.add_middleware(MonitoringMiddleware)

# 2. Security middleware (blocks malicious requests early)
app.add_middleware(SecurityMiddleware)

# 3. Rate limiting middleware (controls request frequency)
app.add_middleware(RateLimitingMiddleware)

# 4. CORS middleware (innermost - handles cross-origin requests)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# API Routes

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve main interface"""
    with open("static/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.get("/avatar-registration", response_class=HTMLResponse)
async def avatar_registration_page():
    """Serve avatar registration interface"""
    with open("static/avatar-registration.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.post("/avatar/register", response_model=AvatarRegistrationResponse)
async def register_avatar(
    file: UploadFile = File(...),
    avatar_name: str = Form(...),
    user_id: str = Form(...),
    description: str = Form(None)
):
    """Register new avatar with face pre-processing"""
    if not app_instance.services_ready:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    try:
        # Read file data
        file_data = await file.read()
        
        # Register avatar
        result = await services["avatar"].register_new_avatar(
            file_data=file_data,
            filename=file.filename,
            user_id=user_id,
            avatar_name=avatar_name
        )
        
        return result
        
    except Exception as e:
        error_response = error_handler.handle_avatar_error(
            error=e,
            avatar_id="new",
            operation="registration"
        )
        raise HTTPException(status_code=500, detail=error_response.error_message)

@app.get("/avatar/list")
async def list_avatars(user_id: str = None):
    """Get list of available avatars"""
    if not app_instance.services_ready:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    try:
        avatars = await services["avatar"].get_avatar_list(user_id=user_id)
        return {"avatars": avatars}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/avatar/{avatar_id}")
async def delete_avatar(avatar_id: str, user_id: str):
    """Delete registered avatar"""
    if not app_instance.services_ready:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    try:
        success = await services["avatar"].delete_avatar(avatar_id=avatar_id, user_id=user_id)
        return {"success": success}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/avatar/{avatar_id}/info")
async def get_avatar_info(avatar_id: str):
    """Get avatar information"""
    if not app_instance.services_ready:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    try:
        avatar_info = await services["avatar"].get_avatar_info(avatar_id=avatar_id)
        if avatar_info is None:
            raise HTTPException(status_code=404, detail="Avatar not found")
        return avatar_info
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health", response_model=ServiceHealthResponse)
async def health_check():
    """Service health status"""
    try:
        models_loaded = model_loader is not None and model_loader.models_loaded
        avatar_cache_loaded = face_cache_manager is not None
        active_sessions = resource_manager.get_active_sessions_count() if resource_manager else 0
        
        # Get registered avatars count
        registered_avatars_count = 0
        if services.get("avatar"):
            avatars = await services["avatar"].get_avatar_list()
            registered_avatars_count = len(avatars)
        
        return ServiceHealthResponse(
            service_status="healthy" if app_instance.services_ready else "starting",
            models_loaded=models_loaded,
            avatar_cache_loaded=avatar_cache_loaded,
            gpu_available=True,  # Validated during startup
            active_sessions=active_sessions,
            registered_avatars_count=registered_avatars_count,
            average_response_time=0.0,  # TODO: Implement metrics
            last_health_check=time.time(),
            available_capacity=max(0, settings.max_concurrent_users - active_sessions)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@app.get("/ready")
async def readiness_check():
    """Model loading status"""
    return {
        "ready": app_instance.services_ready,
        "models_loaded": model_loader is not None and model_loader.models_loaded,
        "avatar_cache_ready": face_cache_manager is not None
    }

@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket streaming endpoint"""
    if not app_instance.services_ready:
        await websocket.close(code=1013, reason="Service not ready")
        return
    
    try:
        await websocket_handler.handle_client_connection(websocket)
    except WebSocketDisconnect:
        logging.info("Client disconnected")
    except Exception as e:
        logging.error(f"WebSocket error: {str(e)}")
        await websocket.close(code=1011, reason="Internal error")

if __name__ == "__main__":
    # Setup logging
    setup_logging()
    
    # Run application
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=5002,
        reload=False,
        workers=1
    ) 