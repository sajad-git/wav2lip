# Avatar Streaming Service API Documentation

## Overview

The Avatar Streaming Service provides real-time AI avatar generation with Persian language support, cold model loading, and avatar registration capabilities. This API enables text-to-avatar and audio-to-avatar processing with optimized performance through pre-loaded models and cached avatar face data.

## Base URL

- **Development**: `http://localhost:5002`
- **Production**: `https://your-domain.com`

## Authentication

Currently, the service uses session-based authentication. Future versions will support API key authentication.

## Content Types

- **Request**: `application/json` for REST endpoints, `multipart/form-data` for file uploads
- **Response**: `application/json` for REST endpoints, binary for WebSocket streams
- **WebSocket**: Binary frames for video chunks, JSON for control messages

---

## Avatar Management Endpoints

### Register New Avatar

**POST** `/avatar/register`

Register a new avatar with face detection and caching.

**Request**:
```http
Content-Type: multipart/form-data

file: [avatar file - image or video]
avatar_name: string (required)
user_id: string (required)
description: string (optional)
tags: string[] (optional)
```

**Response**:
```json
{
  "avatar_id": "uuid-string",
  "registration_status": "success|failed",
  "face_detection_results": {
    "faces_detected": 1,
    "primary_face_confidence": 0.95,
    "face_consistency_score": 0.92,
    "bounding_boxes": [[x, y, w, h]],
    "quality_metrics": {
      "face_clarity": 0.9,
      "lighting_quality": 0.85,
      "processing_ready": true
    }
  },
  "quality_assessment": {
    "overall_score": 0.9,
    "recommendations": ["Good quality avatar ready for processing"]
  },
  "processing_time": 3.2,
  "cache_status": "created",
  "errors": [],
  "warnings": []
}
```

**Status Codes**:
- `200`: Avatar registered successfully
- `400`: Invalid file format or validation failed
- `413`: File too large
- `422`: Face detection failed
- `500`: Server error

### List Avatars

**GET** `/avatar/list?user_id={user_id}`

Retrieve list of available avatars for a user.

**Parameters**:
- `user_id` (optional): Filter by user ID

**Response**:
```json
{
  "avatars": [
    {
      "avatar_id": "uuid-string",
      "name": "My Avatar",
      "file_format": "jpg",
      "file_size": 2048576,
      "resolution": [512, 512],
      "frame_count": 1,
      "registration_date": "2024-01-15T10:30:00Z",
      "last_used": "2024-01-16T14:20:00Z",
      "usage_count": 25,
      "face_quality_score": 0.9,
      "processing_ready": true,
      "cache_size": 1024,
      "owner_id": "user123"
    }
  ],
  "total_count": 1
}
```

### Get Avatar Info

**GET** `/avatar/{avatar_id}/info`

Retrieve detailed information about a specific avatar.

**Response**:
```json
{
  "avatar_id": "uuid-string",
  "name": "My Avatar",
  "file_format": "jpg",
  "file_size": 2048576,
  "resolution": [512, 512],
  "frame_count": 1,
  "registration_date": "2024-01-15T10:30:00Z",
  "last_used": "2024-01-16T14:20:00Z",
  "usage_count": 25,
  "face_quality_score": 0.9,
  "processing_ready": true,
  "cache_size": 1024,
  "owner_id": "user123",
  "face_detection_summary": {
    "faces_detected": 1,
    "primary_face_confidence": 0.95,
    "face_consistency_score": 0.92
  },
  "processing_stats": {
    "average_processing_time": 0.15,
    "cache_hit_rate": 0.98,
    "total_chunks_processed": 150
  }
}
```

### Delete Avatar

**DELETE** `/avatar/{avatar_id}`

Remove an avatar and its associated data.

**Request Body**:
```json
{
  "user_id": "user123",
  "confirm": true
}
```

**Response**:
```json
{
  "success": true,
  "message": "Avatar deleted successfully",
  "cleaned_up": {
    "files_removed": 3,
    "cache_cleared": true,
    "database_updated": true
  }
}
```

---

## Processing Endpoints

### Process Avatar Request

**POST** `/avatar/process`

Process text or audio input with selected avatar.

**Request**:
```json
{
  "input_type": "text|audio",
  "content": "سلام، چطور هستید؟",
  "avatar_id": "uuid-string",
  "language": "fa",
  "quality": "fast|balanced|high",
  "client_id": "client-session-id"
}
```

**Response**:
```json
{
  "request_id": "uuid-string",
  "status": "processing",
  "total_chunks": 3,
  "estimated_duration": 15.5,
  "first_chunk_eta": 0.3,
  "avatar_id": "uuid-string",
  "face_cache_hit": true,
  "websocket_url": "ws://localhost:5002/stream",
  "session_id": "session-uuid"
}
```

---

## System Endpoints

### Health Check

**GET** `/health`

Check service health and model status.

**Response**:
```json
{
  "service_status": "healthy",
  "models_loaded": true,
  "avatar_cache_loaded": true,
  "gpu_available": true,
  "active_sessions": 2,
  "registered_avatars_count": 15,
  "average_response_time": 0.25,
  "last_health_check": "2024-01-16T15:30:00Z",
  "available_capacity": 3
}
```

### Service Ready

**GET** `/ready`

Check if service is ready to accept requests.

**Response**:
```json
{
  "ready": true,
  "model_loading_complete": true,
  "avatar_cache_ready": true,
  "startup_time": "2024-01-16T15:00:00Z",
  "initialization_duration": 12.5
}
```

### Performance Metrics

**GET** `/metrics`

Retrieve service performance metrics.

**Response**:
```json
{
  "processing_metrics": {
    "average_chunk_time": 0.15,
    "total_requests_processed": 1250,
    "average_queue_time": 0.05,
    "cache_hit_rate": 0.95
  },
  "resource_metrics": {
    "gpu_memory_used": 15.2,
    "gpu_utilization": 0.75,
    "system_memory_used": 8.5,
    "cpu_utilization": 0.45
  },
  "avatar_metrics": {
    "registered_avatars": 15,
    "cached_avatars": 12,
    "cache_hit_rate": 0.98,
    "average_registration_time": 3.2
  }
}
```

---

## WebSocket Streaming

### Connection

**WebSocket** `/stream`

Establish WebSocket connection for real-time streaming.

**Connection Headers**:
```
Upgrade: websocket
Connection: Upgrade
Sec-WebSocket-Key: [key]
Sec-WebSocket-Version: 13
```

### Message Types

#### Avatar Selection
```json
{
  "type": "avatar_selection",
  "avatar_id": "uuid-string",
  "client_id": "client-session-id"
}
```

#### Audio Input
```json
{
  "type": "audio_input",
  "audio_data": "base64-encoded-audio",
  "metadata": {
    "format": "wav",
    "sample_rate": 16000,
    "duration": 5.2
  },
  "avatar_id": "uuid-string"
}
```

#### Text Input
```json
{
  "type": "text_input",
  "text": "متن فارسی برای پردازش",
  "language": "fa",
  "avatar_id": "uuid-string"
}
```

#### Video Chunk Response
Binary frame with metadata header:
```
[4 bytes: metadata_length]
[metadata_length bytes: JSON metadata]
[remaining bytes: video data]
```

Metadata structure:
```json
{
  "chunk_id": "uuid-string",
  "sequence_number": 1,
  "total_chunks": 3,
  "frame_count": 25,
  "duration": 1.0,
  "timestamp": 1.0,
  "avatar_id": "uuid-string",
  "sync_info": {
    "audio_offset": 0.0,
    "video_offset": 0.0
  }
}
```

---

## Error Handling

### Error Response Format

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid avatar file format",
    "category": "validation",
    "avatar_related": true,
    "recovery_suggestion": "Please upload a supported image or video format",
    "retry_possible": true,
    "fallback_available": false,
    "timestamp": "2024-01-16T15:30:00Z",
    "request_id": "uuid-string"
  }
}
```

### Error Codes

- `VALIDATION_ERROR`: Input validation failed
- `AVATAR_NOT_FOUND`: Specified avatar doesn't exist
- `FACE_DETECTION_FAILED`: No face detected in avatar
- `PROCESSING_ERROR`: Error during avatar processing
- `GPU_ERROR`: GPU processing error
- `MODEL_ERROR`: Model inference error
- `CACHE_ERROR`: Avatar cache error
- `RATE_LIMIT_EXCEEDED`: Too many requests
- `SERVICE_UNAVAILABLE`: Service temporarily unavailable

---

## Rate Limiting

- **Avatar Registration**: 10 uploads per user per hour
- **Processing Requests**: 100 requests per user per minute
- **WebSocket Connections**: 3 concurrent connections per user

---

## File Upload Limits

- **Maximum File Size**: 50MB (development), 100MB (production)
- **Supported Formats**: JPG, JPEG, PNG, GIF, MP4, MOV
- **Image Resolution**: Minimum 256x256, Maximum 2048x2048
- **Video Duration**: Maximum 30 seconds

---

## Persian Language Support

The service includes optimized support for Persian (Farsi) language:

- **Text Processing**: RTL text handling, Unicode normalization
- **Speech Recognition**: Persian-optimized Whisper models
- **Text-to-Speech**: Persian pronunciation optimization
- **Character Encoding**: Full UTF-8 support with Persian character set

---

## Performance Specifications

- **Model Loading**: Once at startup (5-10 seconds)
- **Avatar Registration**: 2-5 seconds per avatar
- **First Chunk Latency**: <500ms
- **Chunk Processing**: <150ms per 5-15 second segment
- **Face Cache Access**: <10ms per retrieval
- **Concurrent Users**: Up to 5 users (production)

---

## SDK Examples

### JavaScript/Node.js

```javascript
// Avatar Registration
const formData = new FormData();
formData.append('file', avatarFile);
formData.append('avatar_name', 'My Avatar');
formData.append('user_id', 'user123');

const response = await fetch('/avatar/register', {
  method: 'POST',
  body: formData
});

const result = await response.json();

// WebSocket Streaming
const ws = new WebSocket('ws://localhost:5002/stream');

ws.onopen = () => {
  ws.send(JSON.stringify({
    type: 'avatar_selection',
    avatar_id: 'avatar-uuid',
    client_id: 'client-id'
  }));
};

ws.onmessage = (event) => {
  // Handle video chunks
  const buffer = event.data;
  // Process binary video data
};
```

### Python

```python
import requests
import websocket

# Avatar Registration
with open('avatar.jpg', 'rb') as f:
    files = {'file': f}
    data = {
        'avatar_name': 'My Avatar',
        'user_id': 'user123'
    }
    response = requests.post('/avatar/register', files=files, data=data)

# WebSocket Connection
def on_message(ws, message):
    # Handle binary video chunks
    pass

ws = websocket.WebSocketApp("ws://localhost:5002/stream",
                          on_message=on_message)
ws.run_forever()
``` 