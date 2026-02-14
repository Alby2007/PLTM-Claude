#!/usr/bin/env python3
"""
Direct test of Whisper transcription to verify it works.
Run this to test if Whisper can transcribe a simple audio signal.
"""

import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# Generate a simple test audio signal (5 seconds of 440Hz tone)
sample_rate = 16000
duration = 5.0
t = np.linspace(0, duration, int(sample_rate * duration))
audio = np.sin(2 * np.pi * 440 * t).astype(np.float32) * 0.3

print(f"Audio shape: {audio.shape}")
print(f"Audio dtype: {audio.dtype}")
print(f"Audio range: [{audio.min():.3f}, {audio.max():.3f}]")

# Load Whisper
print("\nLoading Whisper tiny model...")
processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
print("Whisper loaded successfully")

# Process audio
print("\nProcessing audio...")
input_features = processor(
    audio,
    sampling_rate=sample_rate,
    return_tensors="pt"
).input_features

print(f"Input features shape: {input_features.shape}")

# Generate transcription
print("\nGenerating transcription...")
forced_decoder_ids = processor.get_decoder_prompt_ids(
    language="en",
    task="transcribe"
)

predicted_ids = model.generate(
    input_features,
    forced_decoder_ids=forced_decoder_ids,
    max_length=448,
    num_beams=1,
)

# Decode
transcription_raw = processor.batch_decode(predicted_ids)[0]
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

print(f"\nRaw output: '{transcription_raw}'")
print(f"Decoded output: '{transcription}'")
print(f"Transcription length: {len(transcription)}")

if transcription.strip():
    print(f"\n✓ SUCCESS: Whisper produced output: '{transcription.strip()}'")
else:
    print("\n✗ ISSUE: Whisper produced empty transcription")
    print("This is expected for a pure tone - Whisper needs actual speech")
    print("\nTry recording actual speech and saving as .wav, then modify this script to load it")
