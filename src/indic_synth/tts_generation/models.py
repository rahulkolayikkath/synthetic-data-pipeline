"""tts_generation.models — the IndicF5 voice-cloning wrapper (§4.5).

Wraps ai4bharat/IndicF5, a zero-shot voice-cloning TTS (F5-TTS based). Inference
takes a target text, a reference audio path, and the reference transcript, and
produces audio at 24 kHz. Uses the model card's default inference parameters.
"""
