# Soniox WebSocket Keepalive Timeout Fix

## Problem
The Soniox STT service was experiencing repeated WebSocket errors:
```
sent 1011 (internal error) keepalive ping timeout; no close frame received
```

This indicated that:
1. WebSocket connections were timing out on keepalive pings
2. Proper close frames weren't being sent during disconnection
3. No timeout handling for long-running operations
4. Connection failures weren't properly handled or recovered from

## Root Causes Identified
1. **Missing timeout configuration**: The original `connect()` call had no timeout parameters
2. **No keepalive monitoring**: No mechanism to detect stalled connections
3. **Poor error recovery**: Errors weren't tracked or handled gracefully
4. **Abrupt disconnection**: Connection close lacked proper WebSocket protocol compliance

## Solutions Implemented

### 1. **Timeout Configuration**
```python
WEBSOCKET_TIMEOUT = 30  # seconds for individual operations
```
- Added explicit timeout to connection establishment
- Set `close_timeout` parameter in `connect()` for graceful shutdown
- Added timeout to `recv()` calls in message loop

### 2. **Connection Health Monitoring**
```python
self.last_message_time = time.time()  # Track last message received
self.consecutive_errors = 0            # Count consecutive errors
```
- Track when last message was received
- Count consecutive errors to trigger reconnection
- Allow detection of stalled connections

### 3. **Proper Timeout Handling**
In `_receive_messages()`:
```python
try:
    message = self.ws.recv(timeout=WEBSOCKET_TIMEOUT)
except TimeoutError:
    # Check if this is a genuine timeout or keepalive issue
    time_since_last = time.time() - self.last_message_time
    if time_since_last > WEBSOCKET_TIMEOUT * 2:
        logger.warning(f"WebSocket timeout - no messages for {time_since_last:.1f}s")
        raise ConnectionError("WebSocket timeout - no keepalive response")
    continue
```
- Distinguishes between temporary network hiccups and genuine failures
- Automatically reconnects when keepalive fails

### 4. **Graceful Disconnection**
```python
def disconnect(self):
    # Send empty string to signal end of audio
    try:
        self.ws.send("")
    except Exception as e:
        logger.debug(f"Error sending disconnect signal: {e}")
    
    # Close connection gracefully with proper close frame
    try:
        self.ws.close(code=1000, reason="Client disconnect")
    except Exception as e:
        logger.debug(f"Error sending close frame: {e}")
```
- Sends proper WebSocket close frame (code 1000 = normal closure)
- Ensures "no close frame received" error doesn't occur

### 5. **Exception Handling**
- Specific handling for `ConnectionClosed`, `TimeoutError`, and `ConnectionError`
- Reset error counter on successful operations
- Trigger reconnection after `MAX_CONSECUTIVE_ERRORS` (5)

### 6. **Thread-Safe Operations**
```python
self.connection_lock = threading.Lock()
```
- Ensures connection/disconnection operations don't race
- Prevents concurrent access issues

### 7. **Connection Status Checking**
```python
def is_connected(self) -> bool:
    """Check if WebSocket is connected"""
    return self.ws is not None and self.running and not self.is_reconnecting
```
- Provides reliable way to check connection status
- Can be used by calling code to determine if reconnection is needed

## Configuration Constants
```python
WEBSOCKET_TIMEOUT = 30              # timeout for individual operations
KEEPALIVE_INTERVAL = 5              # keepalive check interval
CONNECTION_RETRY_DELAY = 1          # initial retry delay
MAX_RETRY_DELAY = 10                # max retry delay
MAX_CONSECUTIVE_ERRORS = 5          # errors before reconnection
```

## Benefits
✅ **Timeout protection** - Operations won't hang indefinitely  
✅ **Keepalive monitoring** - Detects and recovers from stalled connections  
✅ **Graceful shutdown** - Proper WebSocket close frames sent  
✅ **Error recovery** - Automatic reconnection on repeated failures  
✅ **Better logging** - Clear error messages for debugging  
✅ **Thread safety** - No race conditions during connection changes  

## Testing Recommendations
1. Test with long audio streams to verify keepalive handling
2. Test network interruptions and recovery
3. Verify proper WebSocket close frames are sent
4. Monitor for memory leaks during reconnection cycles
5. Test callback error handling doesn't break the service

## Future Improvements
- Implement exponential backoff for connection retries
- Add metrics/monitoring for connection health
- Implement connection pooling for multiple concurrent sessions
- Add circuit breaker pattern for failing endpoints
