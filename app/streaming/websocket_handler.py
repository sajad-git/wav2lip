"""
WebSocket Handler for Avatar Streaming Service.
Manages client connections and binary streaming with avatar selection.
"""

import asyncio
import json
import time
import logging
from typing import Dict, Set, Optional, Any, List
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from fastapi import WebSocket, WebSocketDisconnect
import uuid
import base64

logger = logging.getLogger(__name__)


@dataclass
class ConnectionMetadata:
    """Client connection metadata."""
    
    client_id: str
    connection_time: datetime = field(default_factory=datetime.utcnow)
    last_activity: datetime = field(default_factory=datetime.utcnow)
    selected_avatar_id: Optional[str] = None
    buffer_status: float = 0.5  # 0-1 scale
    quality_settings: Dict[str, Any] = field(default_factory=dict)
    user_agent: str = ""
    capabilities: Dict[str, Any] = field(default_factory=dict)
    bytes_sent: int = 0
    messages_sent: int = 0
    messages_received: int = 0


@dataclass
class BufferStatus:
    """Client buffer status information."""
    
    level: float  # 0-1 scale
    target_level: float = 0.5
    last_update: datetime = field(default_factory=datetime.utcnow)
    chunks_buffered: int = 0
    playback_position: float = 0.0


class MessageValidator:
    """Validates incoming WebSocket messages."""
    
    def __init__(self):
        self.required_fields = {
            "message_type": str,
            "message_id": str,
            "client_id": str,
            "data": dict
        }
        
        self.valid_message_types = {
            "text", "audio", "control", "avatar_selection", "buffer_status"
        }
        
    def validate_message(self, message: Dict[str, Any]) -> tuple[bool, str]:
        """Validate incoming message format."""
        
        # Check required fields
        for field, expected_type in self.required_fields.items():
            if field not in message:
                return False, f"Missing required field: {field}"
            if not isinstance(message[field], expected_type):
                return False, f"Invalid type for field {field}: expected {expected_type.__name__}"
                
        # Check message type
        if message["message_type"] not in self.valid_message_types:
            return False, f"Invalid message type: {message['message_type']}"
            
        # Validate message ID format
        try:
            uuid.UUID(message["message_id"])
        except ValueError:
            return False, "Invalid message_id format"
            
        return True, "Valid"
        
    def validate_avatar_selection(self, data: Dict[str, Any]) -> tuple[bool, str]:
        """Validate avatar selection message."""
        
        if "avatar_id" not in data:
            return False, "Missing avatar_id in avatar selection"
            
        avatar_id = data["avatar_id"]
        if not isinstance(avatar_id, str) or len(avatar_id) < 8:
            return False, "Invalid avatar_id format"
            
        return True, "Valid"
        
    def validate_buffer_status(self, data: Dict[str, Any]) -> tuple[bool, str]:
        """Validate buffer status message."""
        
        required_fields = ["buffer_level", "playback_position"]
        
        for field in required_fields:
            if field not in data:
                return False, f"Missing {field} in buffer status"
                
        buffer_level = data["buffer_level"]
        if not isinstance(buffer_level, (int, float)) or not 0 <= buffer_level <= 1:
            return False, "Invalid buffer_level: must be between 0 and 1"
            
        return True, "Valid"


class FlowController:
    """Controls streaming flow based on client capacity."""
    
    def __init__(self):
        self.client_buffers: Dict[str, BufferStatus] = {}
        self.flow_limits: Dict[str, Dict[str, Any]] = {}
        
    def update_client_buffer(self, client_id: str, buffer_data: Dict[str, Any]):
        """Update client buffer status."""
        
        if client_id not in self.client_buffers:
            self.client_buffers[client_id] = BufferStatus(level=0.5)
            
        buffer_status = self.client_buffers[client_id]
        buffer_status.level = buffer_data.get("buffer_level", buffer_status.level)
        buffer_status.chunks_buffered = buffer_data.get("chunks_buffered", buffer_status.chunks_buffered)
        buffer_status.playback_position = buffer_data.get("playback_position", buffer_status.playback_position)
        buffer_status.last_update = datetime.utcnow()
        
    def should_pause_streaming(self, client_id: str) -> bool:
        """Determine if streaming should be paused."""
        
        if client_id not in self.client_buffers:
            return False
            
        buffer_status = self.client_buffers[client_id]
        
        # Pause if buffer is too full
        if buffer_status.level > 0.9:
            return True
            
        # Pause if buffer updates are stale (client may be disconnected)
        if datetime.utcnow() - buffer_status.last_update > timedelta(seconds=30):
            return True
            
        return False
        
    def get_optimal_chunk_delay(self, client_id: str) -> float:
        """Get optimal delay between chunks."""
        
        if client_id not in self.client_buffers:
            return 0.8  # Default 800ms
            
        buffer_status = self.client_buffers[client_id]
        
        # Adjust delay based on buffer level
        if buffer_status.level < 0.3:
            return 0.5  # Send faster when buffer is low
        elif buffer_status.level > 0.7:
            return 1.2  # Send slower when buffer is high
        else:
            return 0.8  # Normal rate


class AvatarAccessValidator:
    """Validates avatar access permissions."""
    
    def __init__(self):
        self.avatar_permissions: Dict[str, Set[str]] = {}  # avatar_id -> set of allowed client_ids
        self.public_avatars: Set[str] = set()
        
    def add_public_avatar(self, avatar_id: str):
        """Add avatar as publicly accessible."""
        self.public_avatars.add(avatar_id)
        
    def add_private_avatar(self, avatar_id: str, allowed_clients: List[str]):
        """Add avatar with specific client permissions."""
        self.avatar_permissions[avatar_id] = set(allowed_clients)
        
    def can_access_avatar(self, client_id: str, avatar_id: str) -> bool:
        """Check if client can access avatar."""
        
        # Check public avatars
        if avatar_id in self.public_avatars:
            return True
            
        # Check private avatar permissions
        if avatar_id in self.avatar_permissions:
            return client_id in self.avatar_permissions[avatar_id]
            
        return False


class BinaryWebSocketHandler:
    """Handles WebSocket connections with binary streaming and avatar management."""
    
    def __init__(self, avatar_service=None, processing_service=None):
        self.active_connections: Dict[str, WebSocket] = {}
        self.connection_metadata: Dict[str, ConnectionMetadata] = {}
        self.message_validator = MessageValidator()
        self.flow_controller = FlowController()
        self.avatar_validator = AvatarAccessValidator()
        
        # Service dependencies
        self.avatar_service = avatar_service
        self.processing_service = processing_service
        
        # Performance tracking
        self.total_connections = 0
        self.total_disconnections = 0
        self.total_messages_processed = 0
        
    async def connect_client(self, websocket: WebSocket, client_id: str) -> bool:
        """Handle new client connection."""
        
        try:
            await websocket.accept()
            
            # Register client
            self.active_connections[client_id] = websocket
            self.connection_metadata[client_id] = ConnectionMetadata(client_id=client_id)
            self.total_connections += 1
            
            logger.info(f"Client {client_id} connected. Total active: {len(self.active_connections)}")
            
            # Send connection confirmation and available avatars
            await self._send_connection_confirmation(client_id)
            await self._send_available_avatars(client_id)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect client {client_id}: {e}")
            return False
            
    async def disconnect_client(self, client_id: str, reason: str = "Normal closure"):
        """Handle client disconnection."""
        
        if client_id in self.active_connections:
            try:
                websocket = self.active_connections[client_id]
                await websocket.close()
            except Exception as e:
                logger.warning(f"Error closing WebSocket for {client_id}: {e}")
                
            # Clean up
            del self.active_connections[client_id]
            
        if client_id in self.connection_metadata:
            del self.connection_metadata[client_id]
            
        if client_id in self.flow_controller.client_buffers:
            del self.flow_controller.client_buffers[client_id]
            
        self.total_disconnections += 1
        logger.info(f"Client {client_id} disconnected ({reason}). Total active: {len(self.active_connections)}")
        
    async def handle_client_connection(self, websocket: WebSocket, client_id: str):
        """Main connection handler loop."""
        
        try:
            # Connect client
            if not await self.connect_client(websocket, client_id):
                return
                
            # Message handling loop
            while True:
                try:
                    # Receive message
                    message_data = await websocket.receive_text()
                    message = json.loads(message_data)
                    
                    # Update activity timestamp
                    if client_id in self.connection_metadata:
                        self.connection_metadata[client_id].last_activity = datetime.utcnow()
                        self.connection_metadata[client_id].messages_received += 1
                        
                    # Process message
                    await self.process_client_message(message, client_id)
                    self.total_messages_processed += 1
                    
                except WebSocketDisconnect:
                    break
                    
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON from client {client_id}: {e}")
                    await self._send_error_message(client_id, "Invalid JSON format")
                    
                except Exception as e:
                    logger.error(f"Error processing message from {client_id}: {e}")
                    await self._send_error_message(client_id, "Message processing error")
                    
        except Exception as e:
            logger.error(f"Connection handler error for {client_id}: {e}")
            
        finally:
            await self.disconnect_client(client_id, "Connection ended")
            
    async def process_client_message(self, message: Dict[str, Any], client_id: str):
        """Process incoming client message."""
        
        # Validate message format
        is_valid, error_msg = self.message_validator.validate_message(message)
        if not is_valid:
            await self._send_error_message(client_id, f"Message validation failed: {error_msg}")
            return
            
        message_type = message["message_type"]
        message_data = message["data"]
        
        # Route to appropriate handler
        try:
            if message_type == "avatar_selection":
                await self._handle_avatar_selection(message_data, client_id)
            elif message_type == "text":
                await self._handle_text_message(message_data, client_id)
            elif message_type == "audio":
                await self._handle_audio_message(message_data, client_id)
            elif message_type == "control":
                await self._handle_control_message(message_data, client_id)
            elif message_type == "buffer_status":
                await self._handle_buffer_status(message_data, client_id)
            else:
                await self._send_error_message(client_id, f"Unknown message type: {message_type}")
                
        except Exception as e:
            logger.error(f"Error handling {message_type} message from {client_id}: {e}")
            await self._send_error_message(client_id, f"Error processing {message_type} message")
            
    async def _handle_avatar_selection(self, data: Dict[str, Any], client_id: str):
        """Handle avatar selection message."""
        
        # Validate avatar selection
        is_valid, error_msg = self.message_validator.validate_avatar_selection(data)
        if not is_valid:
            await self._send_error_message(client_id, error_msg)
            return
            
        avatar_id = data["avatar_id"]
        
        # Check avatar access permissions
        if not self.avatar_validator.can_access_avatar(client_id, avatar_id):
            await self._send_error_message(client_id, "Access denied for selected avatar")
            return
            
        # Verify avatar exists and is ready
        if self.avatar_service:
            try:
                avatar_info = await self.avatar_service.get_avatar_info(avatar_id)
                if not avatar_info or not avatar_info.get("processing_ready", False):
                    await self._send_error_message(client_id, "Selected avatar is not available or not ready")
                    return
            except Exception as e:
                logger.error(f"Error checking avatar {avatar_id}: {e}")
                await self._send_error_message(client_id, "Error validating avatar")
                return
                
        # Update client metadata
        if client_id in self.connection_metadata:
            self.connection_metadata[client_id].selected_avatar_id = avatar_id
            
        # Send confirmation
        await self._send_message(client_id, {
            "type": "avatar_selected",
            "avatar_id": avatar_id,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        logger.info(f"Client {client_id} selected avatar {avatar_id}")
        
    async def _handle_text_message(self, data: Dict[str, Any], client_id: str):
        """Handle text processing message."""
        
        # Check if avatar is selected
        if client_id not in self.connection_metadata or not self.connection_metadata[client_id].selected_avatar_id:
            await self._send_error_message(client_id, "Please select an avatar first")
            return
            
        avatar_id = self.connection_metadata[client_id].selected_avatar_id
        
        # Add avatar ID to processing request
        processing_data = {
            **data,
            "avatar_id": avatar_id,
            "client_id": client_id
        }
        
        # Start processing
        if self.processing_service:
            try:
                # This should start the processing pipeline and return immediately
                await self.processing_service.process_text_to_avatar(processing_data)
                
                await self._send_message(client_id, {
                    "type": "processing_started",
                    "message": "Text processing started",
                    "timestamp": datetime.utcnow().isoformat()
                })
                
            except Exception as e:
                logger.error(f"Error starting text processing for {client_id}: {e}")
                await self._send_error_message(client_id, "Failed to start text processing")
        else:
            await self._send_error_message(client_id, "Processing service not available")
            
    async def _handle_audio_message(self, data: Dict[str, Any], client_id: str):
        """Handle audio processing message."""
        
        # Check if avatar is selected
        if client_id not in self.connection_metadata or not self.connection_metadata[client_id].selected_avatar_id:
            await self._send_error_message(client_id, "Please select an avatar first")
            return
            
        avatar_id = self.connection_metadata[client_id].selected_avatar_id
        
        # Decode audio data if base64 encoded
        if "audio_data" in data and isinstance(data["audio_data"], str):
            try:
                data["audio_data"] = base64.b64decode(data["audio_data"])
            except Exception as e:
                await self._send_error_message(client_id, "Invalid audio data encoding")
                return
                
        # Add avatar ID to processing request
        processing_data = {
            **data,
            "avatar_id": avatar_id,
            "client_id": client_id
        }
        
        # Start processing
        if self.processing_service:
            try:
                await self.processing_service.process_audio_to_avatar(processing_data)
                
                await self._send_message(client_id, {
                    "type": "processing_started",
                    "message": "Audio processing started",
                    "timestamp": datetime.utcnow().isoformat()
                })
                
            except Exception as e:
                logger.error(f"Error starting audio processing for {client_id}: {e}")
                await self._send_error_message(client_id, "Failed to start audio processing")
        else:
            await self._send_error_message(client_id, "Processing service not available")
            
    async def _handle_control_message(self, data: Dict[str, Any], client_id: str):
        """Handle control message."""
        
        action = data.get("action")
        
        if action == "get_status":
            await self._send_client_status(client_id)
        elif action == "clear_buffer":
            if client_id in self.flow_controller.client_buffers:
                del self.flow_controller.client_buffers[client_id]
            await self._send_message(client_id, {"type": "buffer_cleared"})
        elif action == "get_avatars":
            await self._send_available_avatars(client_id)
        else:
            await self._send_error_message(client_id, f"Unknown control action: {action}")
            
    async def _handle_buffer_status(self, data: Dict[str, Any], client_id: str):
        """Handle buffer status update."""
        
        # Validate buffer status
        is_valid, error_msg = self.message_validator.validate_buffer_status(data)
        if not is_valid:
            await self._send_error_message(client_id, error_msg)
            return
            
        # Update flow controller
        self.flow_controller.update_client_buffer(client_id, data)
        
    async def stream_video_chunk_binary(self, chunk_data: bytes, client_id: str, metadata: Dict[str, Any] = None):
        """Stream binary video chunk to client."""
        
        if client_id not in self.active_connections:
            logger.warning(f"Attempted to stream to disconnected client {client_id}")
            return False
            
        # Check flow control
        if self.flow_controller.should_pause_streaming(client_id):
            logger.debug(f"Streaming paused for client {client_id} due to flow control")
            return False
            
        try:
            websocket = self.active_connections[client_id]
            
            # Create message with binary data
            message = {
                "type": "video_chunk",
                "timestamp": datetime.utcnow().isoformat(),
                "metadata": metadata or {}
            }
            
            # Send JSON metadata first
            await websocket.send_text(json.dumps(message))
            
            # Send binary data
            await websocket.send_bytes(chunk_data)
            
            # Update connection metadata
            if client_id in self.connection_metadata:
                self.connection_metadata[client_id].bytes_sent += len(chunk_data)
                self.connection_metadata[client_id].messages_sent += 1
                
            return True
            
        except Exception as e:
            logger.error(f"Error streaming to client {client_id}: {e}")
            await self.disconnect_client(client_id, "Streaming error")
            return False
            
    async def _send_connection_confirmation(self, client_id: str):
        """Send connection confirmation to client."""
        
        await self._send_message(client_id, {
            "type": "connected",
            "client_id": client_id,
            "timestamp": datetime.utcnow().isoformat(),
            "server_capabilities": {
                "binary_streaming": True,
                "avatar_selection": True,
                "flow_control": True
            }
        })
        
    async def _send_available_avatars(self, client_id: str):
        """Send list of available avatars to client."""
        
        try:
            if self.avatar_service:
                avatars = await self.avatar_service.get_avatar_list()
                # Filter avatars based on access permissions
                accessible_avatars = [
                    avatar for avatar in avatars 
                    if self.avatar_validator.can_access_avatar(client_id, avatar.get("avatar_id", ""))
                ]
            else:
                accessible_avatars = []
                
            await self._send_message(client_id, {
                "type": "available_avatars",
                "avatars": accessible_avatars,
                "timestamp": datetime.utcnow().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error sending avatars to {client_id}: {e}")
            
    async def _send_client_status(self, client_id: str):
        """Send client status information."""
        
        if client_id in self.connection_metadata:
            metadata = self.connection_metadata[client_id]
            status = {
                "type": "client_status",
                "client_id": client_id,
                "connected_since": metadata.connection_time.isoformat(),
                "last_activity": metadata.last_activity.isoformat(),
                "selected_avatar": metadata.selected_avatar_id,
                "messages_sent": metadata.messages_sent,
                "messages_received": metadata.messages_received,
                "bytes_sent": metadata.bytes_sent
            }
        else:
            status = {
                "type": "client_status",
                "client_id": client_id,
                "status": "not_found"
            }
            
        await self._send_message(client_id, status)
        
    async def _send_message(self, client_id: str, message: Dict[str, Any]):
        """Send JSON message to client."""
        
        if client_id not in self.active_connections:
            return False
            
        try:
            websocket = self.active_connections[client_id]
            await websocket.send_text(json.dumps(message))
            
            if client_id in self.connection_metadata:
                self.connection_metadata[client_id].messages_sent += 1
                
            return True
            
        except Exception as e:
            logger.error(f"Error sending message to {client_id}: {e}")
            await self.disconnect_client(client_id, "Send error")
            return False
            
    async def _send_error_message(self, client_id: str, error_message: str):
        """Send error message to client."""
        
        await self._send_message(client_id, {
            "type": "error",
            "message": error_message,
            "timestamp": datetime.utcnow().isoformat()
        })
        
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection statistics."""
        
        active_count = len(self.active_connections)
        
        # Calculate uptime for active connections
        current_time = datetime.utcnow()
        uptimes = []
        
        for metadata in self.connection_metadata.values():
            uptime = (current_time - metadata.connection_time).total_seconds()
            uptimes.append(uptime)
            
        avg_uptime = sum(uptimes) / len(uptimes) if uptimes else 0
        
        return {
            "active_connections": active_count,
            "total_connections": self.total_connections,
            "total_disconnections": self.total_disconnections,
            "total_messages_processed": self.total_messages_processed,
            "average_uptime_seconds": avg_uptime
        } 