"""
Connection Management Component
Manages WebSocket connection lifecycle and client session state
"""

import asyncio
import logging
import time
import uuid
from typing import Dict, Optional, Set, List, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from fastapi import WebSocket, WebSocketDisconnect

from ..models.response_models import ServiceHealthResponse
from ..utils.error_handler import GlobalErrorHandler


class ConnectionState(Enum):
    """WebSocket connection state"""
    CONNECTING = "connecting"
    CONNECTED = "connected"
    AUTHENTICATED = "authenticated"
    ACTIVE = "active"
    IDLE = "idle"
    DISCONNECTING = "disconnecting"
    DISCONNECTED = "disconnected"
    ERROR = "error"


@dataclass
class ClientSession:
    """Client session information"""
    client_id: str
    websocket: WebSocket
    connection_time: float
    last_activity: float
    state: ConnectionState = ConnectionState.CONNECTING
    
    # Avatar context
    selected_avatar_id: Optional[str] = None
    avatar_access_permissions: Set[str] = field(default_factory=set)
    
    # Processing context
    current_request_id: Optional[str] = None
    processing_active: bool = False
    
    # Quality and preferences
    quality_settings: Optional[Dict[str, Any]] = None
    language_preference: str = "fa"  # Default to Persian
    
    # Metrics
    messages_sent: int = 0
    messages_received: int = 0
    bytes_sent: int = 0
    bytes_received: int = 0
    connection_errors: int = 0
    
    # Heartbeat
    last_heartbeat: float = 0.0
    heartbeat_interval: float = 30.0  # seconds
    missed_heartbeats: int = 0
    max_missed_heartbeats: int = 3


@dataclass
class ConnectionMetrics:
    """Global connection metrics"""
    total_connections: int = 0
    active_connections: int = 0
    peak_connections: int = 0
    total_messages: int = 0
    total_bytes_transferred: int = 0
    connection_errors: int = 0
    disconnections: int = 0
    
    # Timing metrics
    average_connection_duration: float = 0.0
    average_message_rate: float = 0.0
    
    # Performance metrics
    message_processing_time: float = 0.0
    websocket_latency: float = 0.0


class ConnectionManager:
    """WebSocket connection lifecycle manager"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Connection tracking
        self.active_connections: Dict[str, ClientSession] = {}
        self.connection_history: List[str] = []  # Recent connection IDs
        
        # Metrics and monitoring
        self.metrics = ConnectionMetrics()
        self.error_handler = GlobalErrorHandler()
        
        # Connection limits and policies
        self.max_connections = 100
        self.max_idle_time = 300.0  # 5 minutes
        self.heartbeat_enabled = True
        
        # Event callbacks
        self.connect_callbacks: List[Callable] = []
        self.disconnect_callbacks: List[Callable] = []
        self.message_callbacks: List[Callable] = []
        self.error_callbacks: List[Callable] = []
        
        # Background tasks
        self.monitoring_active = False
        self.monitoring_task: Optional[asyncio.Task] = None
        self.cleanup_task: Optional[asyncio.Task] = None
    
    async def initialize(self) -> None:
        """Initialize connection manager"""
        self.monitoring_active = True
        
        # Start background monitoring tasks
        self.monitoring_task = asyncio.create_task(self._monitor_connections())
        self.cleanup_task = asyncio.create_task(self._cleanup_inactive_connections())
        
        self.logger.info("✅ Connection manager initialized")
    
    async def shutdown(self) -> None:
        """Shutdown connection manager"""
        self.monitoring_active = False
        
        # Cancel monitoring tasks
        if self.monitoring_task and not self.monitoring_task.done():
            self.monitoring_task.cancel()
        if self.cleanup_task and not self.cleanup_task.done():
            self.cleanup_task.cancel()
        
        # Disconnect all active connections
        await self._disconnect_all_clients()
        
        self.logger.info("🛑 Connection manager shut down")
    
    async def connect_client(self, websocket: WebSocket, 
                           client_id: Optional[str] = None) -> str:
        """Handle new client connection"""
        try:
            # Generate client ID if not provided
            if not client_id:
                client_id = f"client_{uuid.uuid4().hex[:8]}"
            
            # Check connection limits
            if len(self.active_connections) >= self.max_connections:
                self.logger.warning(f"Connection limit reached, rejecting client {client_id}")
                await websocket.close(code=1008, reason="Server capacity reached")
                raise ConnectionError("Maximum connections exceeded")
            
            # Accept WebSocket connection
            await websocket.accept()
            
            # Create client session
            current_time = time.time()
            session = ClientSession(
                client_id=client_id,
                websocket=websocket,
                connection_time=current_time,
                last_activity=current_time,
                last_heartbeat=current_time
            )
            
            # Register client
            self.active_connections[client_id] = session
            self.connection_history.append(client_id)
            
            # Update metrics
            self.metrics.total_connections += 1
            self.metrics.active_connections = len(self.active_connections)
            self.metrics.peak_connections = max(
                self.metrics.peak_connections, 
                self.metrics.active_connections
            )
            
            # Update session state
            session.state = ConnectionState.CONNECTED
            
            # Notify connection callbacks
            await self._notify_connect_callbacks(client_id, session)
            
            self.logger.info(f"✅ Client {client_id} connected successfully")
            return client_id
            
        except Exception as e:
            self.logger.error(f"Failed to connect client: {str(e)}")
            self.metrics.connection_errors += 1
            await self.error_handler.handle_connection_error(e, client_id)
            raise
    
    async def disconnect_client(self, client_id: str, 
                              reason: str = "Normal closure") -> None:
        """Handle client disconnection"""
        if client_id not in self.active_connections:
            self.logger.warning(f"Attempted to disconnect unknown client {client_id}")
            return
        
        try:
            session = self.active_connections[client_id]
            
            # Update session state
            session.state = ConnectionState.DISCONNECTING
            
            # Notify disconnect callbacks
            await self._notify_disconnect_callbacks(client_id, session, reason)
            
            # Close WebSocket connection if still active
            if session.websocket:
                try:
                    await session.websocket.close(code=1000, reason=reason)
                except Exception as e:
                    self.logger.debug(f"WebSocket already closed for {client_id}: {str(e)}")
            
            # Update metrics
            connection_duration = time.time() - session.connection_time
            self._update_disconnection_metrics(session, connection_duration)
            
            # Remove from active connections
            del self.active_connections[client_id]
            self.metrics.active_connections = len(self.active_connections)
            
            self.logger.info(f"🔌 Client {client_id} disconnected: {reason}")
            
        except Exception as e:
            self.logger.error(f"Error disconnecting client {client_id}: {str(e)}")
            # Force removal from active connections
            if client_id in self.active_connections:
                del self.active_connections[client_id]
                self.metrics.active_connections = len(self.active_connections)
    
    async def send_message(self, client_id: str, message: Any, 
                          message_type: str = "text") -> bool:
        """Send message to specific client"""
        if client_id not in self.active_connections:
            self.logger.warning(f"Attempted to send message to unknown client {client_id}")
            return False
        
        session = self.active_connections[client_id]
        
        try:
            # Update activity timestamp
            session.last_activity = time.time()
            
            # Send message based on type
            if message_type == "text":
                await session.websocket.send_text(message)
            elif message_type == "json":
                await session.websocket.send_json(message)
            elif message_type == "bytes":
                await session.websocket.send_bytes(message)
            else:
                raise ValueError(f"Unsupported message type: {message_type}")
            
            # Update metrics
            session.messages_sent += 1
            if isinstance(message, (str, bytes)):
                session.bytes_sent += len(message)
            
            self.metrics.total_messages += 1
            
            self.logger.debug(f"Sent {message_type} message to client {client_id}")
            return True
            
        except WebSocketDisconnect:
            self.logger.info(f"Client {client_id} disconnected during message send")
            await self.disconnect_client(client_id, "Client disconnected")
            return False
        except Exception as e:
            self.logger.error(f"Failed to send message to client {client_id}: {str(e)}")
            session.connection_errors += 1
            await self.error_handler.handle_message_error(e, client_id, message)
            return False
    
    async def broadcast_message(self, message: Any, message_type: str = "text",
                              exclude_clients: Optional[Set[str]] = None) -> int:
        """Broadcast message to all connected clients"""
        exclude_clients = exclude_clients or set()
        successful_sends = 0
        
        # Get list of clients to avoid modification during iteration
        client_ids = list(self.active_connections.keys())
        
        for client_id in client_ids:
            if client_id not in exclude_clients:
                success = await self.send_message(client_id, message, message_type)
                if success:
                    successful_sends += 1
        
        self.logger.info(f"Broadcast message to {successful_sends}/{len(client_ids)} clients")
        return successful_sends
    
    def get_client_session(self, client_id: str) -> Optional[ClientSession]:
        """Get client session information"""
        return self.active_connections.get(client_id)
    
    def update_client_activity(self, client_id: str) -> None:
        """Update client activity timestamp"""
        if client_id in self.active_connections:
            session = self.active_connections[client_id]
            session.last_activity = time.time()
            
            # Update state if idle
            if session.state == ConnectionState.IDLE:
                session.state = ConnectionState.ACTIVE
    
    def set_client_avatar(self, client_id: str, avatar_id: str) -> bool:
        """Set selected avatar for client"""
        if client_id not in self.active_connections:
            return False
        
        session = self.active_connections[client_id]
        session.selected_avatar_id = avatar_id
        session.last_activity = time.time()
        
        self.logger.info(f"Client {client_id} selected avatar {avatar_id}")
        return True
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection statistics"""
        current_time = time.time()
        
        # Calculate active session stats
        session_durations = []
        idle_clients = 0
        
        for session in self.active_connections.values():
            duration = current_time - session.connection_time
            session_durations.append(duration)
            
            if current_time - session.last_activity > 60:  # 1 minute idle
                idle_clients += 1
        
        avg_duration = sum(session_durations) / len(session_durations) if session_durations else 0
        
        return {
            "total_connections": self.metrics.total_connections,
            "active_connections": self.metrics.active_connections,
            "peak_connections": self.metrics.peak_connections,
            "idle_clients": idle_clients,
            "average_session_duration": avg_duration,
            "total_messages": self.metrics.total_messages,
            "total_bytes_transferred": self.metrics.total_bytes_transferred,
            "connection_errors": self.metrics.connection_errors,
            "disconnections": self.metrics.disconnections
        }
    
    async def _monitor_connections(self) -> None:
        """Monitor connection health and heartbeats"""
        while self.monitoring_active:
            try:
                current_time = time.time()
                clients_to_disconnect = []
                
                for client_id, session in self.active_connections.items():
                    # Check for idle connections
                    idle_time = current_time - session.last_activity
                    if idle_time > self.max_idle_time:
                        session.state = ConnectionState.IDLE
                    
                    # Check heartbeat if enabled
                    if self.heartbeat_enabled:
                        heartbeat_age = current_time - session.last_heartbeat
                        if heartbeat_age > session.heartbeat_interval * 2:
                            session.missed_heartbeats += 1
                            
                            if session.missed_heartbeats >= session.max_missed_heartbeats:
                                self.logger.warning(f"Client {client_id} missed too many heartbeats")
                                clients_to_disconnect.append((client_id, "Heartbeat timeout"))
                
                # Disconnect problematic clients
                for client_id, reason in clients_to_disconnect:
                    await self.disconnect_client(client_id, reason)
                
                # Sleep before next check
                await asyncio.sleep(10.0)  # Check every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Error in connection monitoring: {str(e)}")
                await asyncio.sleep(5.0)
    
    async def _cleanup_inactive_connections(self) -> None:
        """Clean up inactive connections and stale data"""
        while self.monitoring_active:
            try:
                current_time = time.time()
                
                # Limit connection history size
                if len(self.connection_history) > 1000:
                    self.connection_history = self.connection_history[-500:]
                
                # Update connection metrics
                total_duration = 0
                for session in self.active_connections.values():
                    total_duration += current_time - session.connection_time
                
                if self.active_connections:
                    self.metrics.average_connection_duration = (
                        total_duration / len(self.active_connections)
                    )
                
                # Sleep before next cleanup
                await asyncio.sleep(60.0)  # Cleanup every minute
                
            except Exception as e:
                self.logger.error(f"Error in connection cleanup: {str(e)}")
                await asyncio.sleep(30.0)
    
    async def _disconnect_all_clients(self) -> None:
        """Disconnect all active clients"""
        client_ids = list(self.active_connections.keys())
        
        for client_id in client_ids:
            await self.disconnect_client(client_id, "Server shutdown")
        
        self.logger.info(f"Disconnected {len(client_ids)} clients")
    
    def _update_disconnection_metrics(self, session: ClientSession, duration: float) -> None:
        """Update metrics for client disconnection"""
        self.metrics.disconnections += 1
        self.metrics.total_bytes_transferred += session.bytes_sent + session.bytes_received
        
        # Update average connection duration
        total_duration = self.metrics.average_connection_duration * (self.metrics.disconnections - 1)
        self.metrics.average_connection_duration = (total_duration + duration) / self.metrics.disconnections
    
    # Event callback management
    def register_connect_callback(self, callback: Callable) -> None:
        """Register callback for client connections"""
        self.connect_callbacks.append(callback)
    
    def register_disconnect_callback(self, callback: Callable) -> None:
        """Register callback for client disconnections"""
        self.disconnect_callbacks.append(callback)
    
    def register_message_callback(self, callback: Callable) -> None:
        """Register callback for message events"""
        self.message_callbacks.append(callback)
    
    def register_error_callback(self, callback: Callable) -> None:
        """Register callback for error events"""
        self.error_callbacks.append(callback)
    
    async def _notify_connect_callbacks(self, client_id: str, session: ClientSession) -> None:
        """Notify connection callbacks"""
        for callback in self.connect_callbacks:
            try:
                await callback(client_id, session)
            except Exception as e:
                self.logger.error(f"Error in connect callback: {str(e)}")
    
    async def _notify_disconnect_callbacks(self, client_id: str, session: ClientSession, reason: str) -> None:
        """Notify disconnection callbacks"""
        for callback in self.disconnect_callbacks:
            try:
                await callback(client_id, session, reason)
            except Exception as e:
                self.logger.error(f"Error in disconnect callback: {str(e)}")
    
    async def send_heartbeat(self, client_id: str) -> bool:
        """Send heartbeat to specific client"""
        if client_id not in self.active_connections:
            return False
        
        session = self.active_connections[client_id]
        
        try:
            heartbeat_message = {
                "type": "heartbeat",
                "timestamp": time.time(),
                "server_time": time.time()
            }
            
            success = await self.send_message(client_id, heartbeat_message, "json")
            
            if success:
                session.last_heartbeat = time.time()
                session.missed_heartbeats = 0
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to send heartbeat to {client_id}: {str(e)}")
            return False
    
    async def handle_heartbeat_response(self, client_id: str, response: Dict[str, Any]) -> None:
        """Handle heartbeat response from client"""
        if client_id not in self.active_connections:
            return
        
        session = self.active_connections[client_id]
        session.last_heartbeat = time.time()
        session.missed_heartbeats = 0
        
        # Calculate WebSocket latency if timestamp provided
        if "client_timestamp" in response:
            server_time = time.time()
            client_time = response["client_timestamp"]
            latency = server_time - client_time
            self.metrics.websocket_latency = latency
        
        self.logger.debug(f"Received heartbeat response from {client_id}")
    
    def get_health_status(self) -> ServiceHealthResponse:
        """Get connection manager health status"""
        active_count = len(self.active_connections)
        
        # Determine health status
        if active_count == 0:
            status = "healthy"
        elif active_count < self.max_connections * 0.8:
            status = "healthy"
        elif active_count < self.max_connections * 0.95:
            status = "degraded"
        else:
            status = "error"
        
        return ServiceHealthResponse(
            service_status=status,
            models_loaded=True,  # Would be checked from model loader
            avatar_cache_loaded=True,  # Would be checked from avatar cache
            gpu_available=True,  # Would be checked from GPU manager
            active_sessions=active_count,
            registered_avatars_count=0,  # Would be provided by avatar service
            average_response_time=self.metrics.websocket_latency,
            last_health_check=time.time(),
            available_capacity=self.max_connections - active_count
        ) 