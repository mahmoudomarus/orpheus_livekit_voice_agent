import asyncio
import io
import logging
import aiohttp
import numpy as np
from typing import Any, Dict, Optional, Callable, List, AsyncIterator
import os
import functools
import time
import uuid
import ctypes
import hashlib

# Define capabilities class
class TTSCapabilities:
    def __init__(self, streaming=False):
        self.streaming = streaming

# Audio frame class to match Livekit's expectations
class AudioFrame:
    """
    A class that represents a frame of audio data with specific properties such as sample rate,
    number of channels, and samples per channel.
    """
    
    def __init__(self, data, sample_rate, num_channels=1, samples_per_channel=None):
        """
        Initialize an AudioFrame instance.
        
        Args:
            data: The raw audio data, should be a numpy array of int16 values
            sample_rate: The sample rate of the audio in Hz
            num_channels: The number of audio channels (e.g., 1 for mono, 2 for stereo)
            samples_per_channel: The number of samples per channel
        """
        if isinstance(data, np.ndarray):
            # Make sure data is int16
            if data.dtype != np.int16:
                data = data.astype(np.int16)
            
            # Store as numpy array
            self._data = data
        else:
            # Convert to numpy array if not already
            self._data = np.frombuffer(data, dtype=np.int16)
            
        self.sample_rate = sample_rate
        self.num_channels = num_channels
        
        # Calculate samples per channel if not provided
        if samples_per_channel is None:
            self.samples_per_channel = len(self._data) // num_channels
        else:
            self.samples_per_channel = samples_per_channel
            
        # Make the frame attribute reference self for compatibility with Livekit
        self.frame = self
        
        # LiveKit seems to require this to be bytes not numpy array
        self.data = self._data.tobytes()
    
    @property
    def duration(self):
        """Returns the duration of the audio frame in seconds."""
        return self.samples_per_channel / self.sample_rate
        
    def _proto_info(self):
        """
        Returns a protocol buffer compatible representation of the audio frame.
        This method is required by Livekit's AudioSource.capture_frame() method.
        """
        # Import here to avoid circular imports
        from livekit.rtc._proto import audio_frame_pb2
        
        # Create the proper protobuf object
        audio_info = audio_frame_pb2.AudioFrameBufferInfo()
        
        # Store the raw bytes data in a buffer
        import ctypes
        
        # Create a C buffer from our bytes data
        data_bytes = self.data
        buffer = (ctypes.c_byte * len(data_bytes)).from_buffer_copy(data_bytes)
        
        # Set the data pointer to the buffer's address
        audio_info.data_ptr = ctypes.addressof(buffer)
        
        # Set the properties
        audio_info.sample_rate = self.sample_rate
        audio_info.num_channels = self.num_channels
        audio_info.samples_per_channel = self.samples_per_channel
        
        # We need to keep a reference to the buffer to prevent it from being garbage collected
        self._buffer_ref = buffer
        
        return audio_info

# Error class (matching Livekit's format)
class Error:
    pass

# Metrics class to match Livekit's expectations
class TTSMetrics:
    def __init__(self, characters_count, duration, audio_duration):
        self.request_id = str(uuid.uuid4())
        self.timestamp = time.time()
        self.ttfb = 0.1  # Time to first byte - estimate
        self.duration = duration
        self.audio_duration = audio_duration
        self.cancelled = False
        self.characters_count = characters_count
        self.label = "orpheus"
        self.streamed = False
        self.error = None

# Stream class for yielding audio chunks
class AudioStream:
    """Asynchronous iterator for audio chunks."""
    
    def __init__(self, audio_data, sample_rate, chunk_size=2048):  # Increased chunk size for better performance
        self.audio_data = audio_data
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.position = 0
    
    def __aiter__(self):
        return self
    
    async def __anext__(self):
        if self.position >= len(self.audio_data):
            raise StopAsyncIteration
        
        end = min(self.position + self.chunk_size, len(self.audio_data))
        chunk = self.audio_data[self.position:end]
        self.position = end
        
        # Return properly structured AudioFrame
        return AudioFrame(chunk, self.sample_rate)
    
    async def aclose(self):
        """Close the audio stream."""
        self.position = len(self.audio_data)  # Mark as complete

# Instead of importing TTS, let's create a compatible interface
class TTS:
    """
    Custom TTS implementation for Orpheus using Baseten API.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        voice: str = "tara",
        max_tokens: int = 10000,
        endpoint_url: str = #replace it with your endpoint from baseten
        sample_rate: int = 24000,
        cache_size: int = 100,  # Cache size for storing common phrases
    ):
        """
        Initialize the Orpheus TTS.
        
        Args:
            api_key: Orpheus API key. If not provided, it will be loaded from the ORPHEUS_API_KEY environment variable.
            voice: Voice to use (e.g., "tara")
            max_tokens: Maximum number of tokens to generate
            endpoint_url: Orpheus endpoint URL
            sample_rate: Audio sample rate
            cache_size: Number of phrases to cache
        """
        self.api_key = api_key
        self.voice = voice
        self.max_tokens = max_tokens
        self.endpoint_url = endpoint_url
        self.sample_rate = sample_rate
        self.cache_size = cache_size
        
        # Add attributes required by Livekit
        self.capabilities = TTSCapabilities(streaming=False)
        self.num_channels = 1  # Mono audio
        self.model = "orpheus"
        self.voice_id = voice
        
        # Event emitter functionality
        self._event_handlers = {}
        
        # Create a cache of common phrases for faster responses
        # Using a dictionary with the MD5 hash of the text as the key
        self._cache = {}
        self._cache_order = []  # LRU tracking
        
        # Create a shared session for all requests
        self._session = None
        
        if self.api_key is None:
            self.api_key = os.environ.get("ORPHEUS_API_KEY")
            if self.api_key is None:
                raise ValueError("ORPHEUS_API_KEY environment variable not set")
    
    # Event emitter methods
    def on(self, event_name: str, callback: Optional[Callable] = None) -> Any:
        """Register an event handler. Can be used as a decorator."""
        # If used as decorator with no arguments
        if callback is None:
            def decorator(func: Callable) -> Callable:
                if event_name not in self._event_handlers:
                    self._event_handlers[event_name] = []
                self._event_handlers[event_name].append(func)
                return func
            return decorator
        # If used as a method call
        else:
            if event_name not in self._event_handlers:
                self._event_handlers[event_name] = []
            self._event_handlers[event_name].append(callback)
            return callback
    
    def off(self, event_name: str, callback: Optional[Callable] = None) -> None:
        """Remove an event handler."""
        if callback is None:
            self._event_handlers[event_name] = []
        else:
            if event_name in self._event_handlers:
                self._event_handlers[event_name] = [
                    cb for cb in self._event_handlers[event_name] if cb != callback
                ]
    
    def emit(self, event_name: str, *args: Any, **kwargs: Any) -> None:
        """Emit an event."""
        if event_name in self._event_handlers:
            for callback in self._event_handlers[event_name]:
                callback(*args, **kwargs)
    
    def _add_to_cache(self, text, audio_data):
        """Add synthesized audio to the cache."""
        # Create a hash of the text as the cache key
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
        
        # If the cache is full, remove the oldest entry
        if len(self._cache) >= self.cache_size:
            oldest_key = self._cache_order.pop(0)
            del self._cache[oldest_key]
        
        # Add the new entry
        self._cache[text_hash] = audio_data
        self._cache_order.append(text_hash)
    
    def _get_from_cache(self, text):
        """Get synthesized audio from the cache if available."""
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
        
        if text_hash in self._cache:
            # Move this entry to the end of the LRU list
            self._cache_order.remove(text_hash)
            self._cache_order.append(text_hash)
            
            return self._cache[text_hash]
        
        return None
    
    async def _get_session(self):
        """Get or create an aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session
    
    async def aclose(self) -> None:
        """Required by Livekit StreamAdapter to close the TTS."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
    
    async def _preprocess_text(self, text):
        """Preprocess text for better synthesis results and performance."""
        # Split long text into sentences for better streaming
        if len(text) > 100:
            import re
            sentences = re.split(r'(?<=[.!?])\s+', text)
            return sentences
        return [text]

    async def synthesize(self, text: str) -> AsyncIterator[AudioFrame]:
        """
        Synthesize text to audio. Returns an async iterator that yields audio frames.
        
        Args:
            text: Text to synthesize
            
        Returns:
            Async iterator yielding AudioFrame objects
        """
        logger = logging.getLogger(__name__)
        logger.info(f"Synthesizing text with Orpheus TTS: {text[:50]}...")
        
        sentences = await self._preprocess_text(text)
        
        # For metrics calculation
        start_time = time.time()
        all_audio_bytes = bytearray()
        
        for sentence in sentences:
            # Check if this sentence is in the cache
            cached_audio = self._get_from_cache(sentence)
            
            if cached_audio is not None:
                # Use cached audio
                logger.info(f"Using cached audio for: {sentence[:20]}...")
                audio_data = cached_audio
            else:
                # Get audio from Orpheus API
                session = await self._get_session()
                
                headers = {
                    "Authorization": f"Api-Key {self.api_key}",
                    "Content-Type": "application/json",
                }
                
                data = {
                    "voice": self.voice,
                    "prompt": sentence,
                    "max_tokens": min(self.max_tokens, len(sentence) * 10)  # Scale tokens based on text length
                }
                
                async with session.post(
                    self.endpoint_url,
                    json=data,
                    headers=headers
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Error from Orpheus API: {error_text}")
                        raise Exception(f"Error from Orpheus API: {error_text}")
                    
                    content = await response.read()
                    audio_data = np.frombuffer(content, dtype=np.int16)
                    
                    # Add to cache for future use
                    self._add_to_cache(sentence, audio_data)
            
            # Accumulate all audio data for metrics
            all_audio_bytes.extend(audio_data.tobytes())
            
            # Stream this sentence's audio
            stream = AudioStream(audio_data, self.sample_rate, chunk_size=2048)
            async for frame in stream:
                yield frame
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Estimate audio duration based on sample rate and number of samples
        # Assuming 16-bit audio which is 2 bytes per sample
        num_samples = len(all_audio_bytes) / 2  # 2 bytes per sample for 16-bit audio
        audio_duration = num_samples / self.sample_rate
        
        # Emit metrics for consumption by StreamAdapter using a proper Metrics object
        metrics = TTSMetrics(
            characters_count=len(text),
            duration=duration,
            audio_duration=audio_duration
        )
        self.emit("metrics_collected", metrics) 
