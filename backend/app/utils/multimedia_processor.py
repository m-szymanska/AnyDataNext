"""
Utilities for processing multimedia files (audio/video) to create datasets.
"""
import asyncio
import os
import json
import time
import logging
import traceback
import subprocess # For calling ffmpeg, whisper
from pathlib import Path
from typing import Optional, List, Dict, Any
import zipfile
import shutil # For checking ffmpeg availability
import mlx_whisper # Import the library
from concurrent.futures import ThreadPoolExecutor # For running sync whisper in async

# Assuming ConnectionManager is defined in app.py or accessible elsewhere
# We need a way to pass it or a reference for broadcasting progress
# from ..app import ConnectionManager # This might cause circular imports, pass instance instead

from .logging import setup_logging

logger = setup_logging()

# --- Helper: Check for Dependencies ---
def check_dependencies():
    """Checks for required external command-line tools."""
    if not shutil.which("ffmpeg"):
        raise RuntimeError("FFmpeg not found. Please install ffmpeg and ensure it's in the system PATH.")
    # Basic check if mlx_whisper is importable (pip install worked)
    try:
        import mlx_whisper
    except ImportError:
        raise RuntimeError("mlx-whisper package not installed. Please run 'pip install mlx-whisper'.")
    # We might want a more robust check later, e.g., attempting a model load.
    logger.info("Dependencies (ffmpeg, mlx-whisper) seem available.")


# --- Helper: Run FFmpeg Command --- 
async def run_ffmpeg(
    command: List[str],
    job_id: str,
    error_message: str = "FFmpeg command failed"
) -> bool:
    """
    Runs an ffmpeg command using subprocess asynchronously.
    Logs output and errors.
    Returns True on success (exit code 0), False otherwise.
    """
    command_str = " ".join(command)
    logger.info(f"[{job_id}] Running FFmpeg: {command_str}")
    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await process.communicate()
    
    if process.returncode != 0:
        stderr_decoded = stderr.decode('utf-8', errors='ignore').strip()
        logger.error(f"[{job_id}] {error_message}: {stderr_decoded}")
        logger.error(f"[{job_id}] FFmpeg command was: {command_str}")
        return False
    else:
        stdout_decoded = stdout.decode('utf-8', errors='ignore').strip()
        if stdout_decoded:
             logger.debug(f"[{job_id}] FFmpeg stdout: {stdout_decoded}")
        stderr_decoded = stderr.decode('utf-8', errors='ignore').strip()
        if stderr_decoded: # FFmpeg often outputs info to stderr
            logger.debug(f"[{job_id}] FFmpeg stderr: {stderr_decoded}")
        logger.info(f"[{job_id}] FFmpeg command completed successfully.")
        return True


# --- Segmentation Logic --- 
def segment_transcription(
    words: List[Dict[str, Any]],
    job_id: str, # Add job_id for logging context
    max_duration: float = 20.0, # Max duration in seconds
    max_pause: float = 1.0,     # Max pause between words to force split
    sentence_end_chars: str = ".?!",
) -> List[Dict[str, Any]]:
    """
    Segments a list of transcribed words with timestamps into sentence-like chunks.

    Args:
        words: List of word dictionaries [{'word': str, 'start': float, 'end': float}, ...].
        max_duration: Maximum duration of a segment in seconds.
        max_pause: Maximum pause duration between words to trigger a segment split.
        sentence_end_chars: Characters indicating potential sentence ends.

    Returns:
        List of segment dictionaries [{'text': str, 'start_time': float, 'end_time': float, 'words': list}, ...].
    """
    segments = []
    if not words:
        return segments

    current_segment_words = []
    
    for i, word_info in enumerate(words):
        word_text = word_info.get("word", "").strip()
        start_time = word_info.get("start")
        end_time = word_info.get("end")

        if start_time is None or end_time is None or end_time <= start_time: # Added explicit end > start check
            logger.warning(f"[{job_id}] Skipping word due to missing or invalid timestamp: {word_info}")
            continue
            
        current_segment_words.append(word_info)
        
        segment_start_time = current_segment_words[0]['start']
        segment_end_time = current_segment_words[-1]['end']
        
        # Check segment duration only if timestamps are valid
        segment_duration = 0
        if segment_start_time is not None and segment_end_time is not None and segment_end_time > segment_start_time:
            segment_duration = segment_end_time - segment_start_time
        else:
            # This case should ideally not happen due to the check above, but as a safeguard:
             logger.warning(f"[{job_id}] Invalid timestamps during segment duration calculation: Start={segment_start_time}, End={segment_end_time}")

        # Check for segment termination conditions
        end_segment = False
        split_reason = None # For logging
        
        # 1. Sentence ending punctuation
        if word_text and word_text[-1] in sentence_end_chars:
            end_segment = True
            split_reason = f"punctuation ('{word_text[-1]}')"
            
        # 2. Maximum duration exceeded (only trigger if not already ending)
        if not end_segment and segment_duration > max_duration:
            # If the segment has only one word and it already exceeds max_duration, 
            # we have to keep it, otherwise we might lose words.
            if len(current_segment_words) > 1:
                end_segment = True
                split_reason = f"max_duration ({segment_duration:.2f}s > {max_duration}s)"
            else:
                 logger.warning(f"[{job_id}] Single word segment exceeds max_duration ({segment_duration:.2f}s). Keeping word: '{word_text}'")

            
        # 3. Pause duration exceeded (only trigger if not already ending)
        if not end_segment and i + 1 < len(words):
            next_word_info = words[i+1]
            next_start_time = next_word_info.get("start")
            # Check validity of next word's start time
            if next_start_time is not None and next_start_time > end_time:
                 pause_duration = next_start_time - end_time
                 if pause_duration > max_pause:
                     end_segment = True
                     split_reason = f"max_pause ({pause_duration:.2f}s > {max_pause}s)"
            elif next_start_time is not None: # Handles cases like zero pause or overlapping words from Whisper
                 logger.debug(f"[{job_id}] Non-positive pause detected or invalid next_start_time: Current_End={end_time}, Next_Start={next_start_time}")
                     
        # 4. Is it the last word? (always end segment if it's the last word)
        if i == len(words) - 1:
            end_segment = True
            if not split_reason: # Avoid overriding previous reason
                 split_reason = "last_word"

        if end_segment and current_segment_words:
            segment_text = "".join([w.get("word", "") for w in current_segment_words]).strip()
            final_start = current_segment_words[0]['start']
            final_end = current_segment_words[-1]['end']

            # Final validation before adding
            if final_start is None or final_end is None or final_start >= final_end:
                # Log detailed info about why segment creation is skipped
                logger.warning(f"[{job_id}] Skipping segment creation due to invalid final timestamps: Start={final_start}, End={final_end}, Reason: {split_reason}, Text='{segment_text[:50]}...'")
                skipped_segments += 1
                continue

            logger.debug(f"[{job_id}] Ending segment. Reason: {split_reason}. Duration: {(final_end-final_start):.2f}s. Words: {len(current_segment_words)}.")
            segments.append({
               "text": segment_text,
               "start_time": final_start,
               "end_time": final_end,
               "words": list(current_segment_words), 
            })

            current_segment_words = [] 
            
    return segments


# --- create_audio_text_dataset function (Modified Whisper call) ---
async def create_audio_text_dataset(
    job_id: str,
    input_file_path: Path,
    output_dir: Path, 
    language: Optional[str] = None,
    model_name: str = "large-v3", # Add model selection
    websocket_manager: Optional[Any] = None # Pass the ConnectionManager instance
) -> str:
    """
    Main pipeline function to process an audio or video file.
    Steps include: audio extraction, transcription with timestamps, 
    segmentation, audio chunking, metadata generation, and zipping.

    Args:
        job_id: The ID for this processing job.
        input_file_path: Path to the input audio/video file.
        output_dir: Directory where intermediate files and the final zip should be stored.
        language: Optional language hint for Whisper. If None, whisper attempts auto-detection.
        model_name: Name of the Whisper model to use (e.g., 'tiny', 'base', 'small', 'medium', 'large-v3').
        websocket_manager: Instance of ConnectionManager for sending progress updates.

    Returns:
        Path to the final generated ZIP file.
        
    Raises:
        Exception: If any critical step in the pipeline fails.
    """
    
    start_time = time.time()
    logger.info(f"[{job_id}] Starting audio dataset creation for: {input_file_path.name} using model '{model_name}'")

    # --- Check Dependencies --- 
    try:
        check_dependencies()
    except RuntimeError as dep_error:
        logger.error(f"[{job_id}] Dependency check failed: {dep_error}")
        raise # Re-raise to stop processing

    # --- Helper function to send progress --- 
    async def send_progress(status: str, progress: int, details: Optional[Dict] = None):
        if websocket_manager:
            message = {
                "job_id": job_id,
                "type": "job_update",
                "status": status,
                "progress": progress
            }
            if details:
                message.update(details)
            await websocket_manager.broadcast(message)
        else:
            logger.warning(f"[{job_id}] WebSocket manager not provided, cannot send progress.")

    # --- Create subdirectories for intermediate files --- 
    audio_extract_dir = output_dir / "temp_audio"
    audio_output_dir = output_dir / "audio_chunks"
    metadata_output_dir = output_dir / "metadata"
    # Ensure all directories exist
    audio_extract_dir.mkdir(exist_ok=True)
    audio_output_dir.mkdir(exist_ok=True)
    metadata_output_dir.mkdir(exist_ok=True)

    # Define path for the extracted/converted WAV file
    extracted_audio_path = audio_extract_dir / f"{input_file_path.stem}_16khz_mono.wav"

    try:
        # --- Step 1: Extract Audio & Convert to WAV --- 
        await send_progress("Preparing Audio (ffmpeg)...", 10)
        logger.info(f"[{job_id}] Step 1: Extracting/Converting audio to 16kHz mono WAV...")
        
        ffmpeg_command_extract = [
            "ffmpeg",
            "-i", str(input_file_path), # Input file
            "-vn",                     # Disable video recording
            "-acodec", "pcm_s16le",   # Set audio codec to 16-bit PCM (standard WAV)
            "-ar", "16000",             # Set audio sampling rate to 16kHz
            "-ac", "1",                 # Set number of audio channels to 1 (mono)
            "-y",                      # Overwrite output file if it exists
            str(extracted_audio_path)  # Output file path
        ]
        
        success = await run_ffmpeg(ffmpeg_command_extract, job_id, "Audio extraction/conversion failed")
        if not success or not extracted_audio_path.is_file():
             # Error is already logged by run_ffmpeg
             raise RuntimeError("Failed to extract or convert audio track using FFmpeg.")
             
        logger.info(f"[{job_id}] Audio prepared: {extracted_audio_path.name}")

        # --- Step 2: Transcription with Timestamps (Async Whisper) --- 
        await send_progress(f"Transcribing ({model_name})...", 25)
        logger.info(f"[{job_id}] Step 2: Starting Whisper MLX transcription (model: {model_name}) in background thread...")
        
        transcription_result_raw = None
        loop = asyncio.get_running_loop()
        try:
            transcription_start_time = time.time()
            
            # Run the synchronous mlx_whisper.transcribe in a thread pool executor
            with ThreadPoolExecutor() as pool:
                transcription_result_raw = await loop.run_in_executor(
                    pool,
                    mlx_whisper.transcribe, # The function to run
                    str(extracted_audio_path), # Positional arguments for the function
                    model_name, # Corresponds to path_or_hf_repo
                    # --- Keyword arguments for mlx_whisper.transcribe --- 
                    # language=language, # Optional: Pass language hint 
                    # word_timestamps=True # This seems to be default or implicitly True in newer versions? Let's rely on output check.
                    # verbose=False 
                )
            # Check if word_timestamps is directly in the result, if not, try segments (older versions?)    
            if transcription_result_raw and "word_timestamps" not in transcription_result_raw and "segments" in transcription_result_raw:
                logger.info(f"[{job_id}] 'word_timestamps' key not found directly, attempting to extract from segments.")
                # Attempt to reconstruct word timestamps from segments if the main key is missing
                all_words = []
                for segment in transcription_result_raw.get("segments", []):
                    all_words.extend(segment.get("words", []))
                if all_words:
                     transcription_result_raw["word_timestamps"] = all_words
                else:
                     logger.warning(f"[{job_id}] Could not reconstruct word timestamps from segments.")
            
            transcription_duration = time.time() - transcription_start_time
            logger.info(f"[{job_id}] Whisper MLX transcription thread finished in {transcription_duration:.2f}s.")

        except Exception as whisper_error:
            logger.error(f"[{job_id}] Whisper MLX transcription failed: {whisper_error}", exc_info=True)
            raise RuntimeError(f"Whisper MLX transcription failed: {whisper_error}") from whisper_error
            
        if not transcription_result_raw or ("segments" not in transcription_result_raw and "word_timestamps" not in transcription_result_raw):
             logger.error(f"[{job_id}] Whisper MLX returned unexpected or empty result: {transcription_result_raw}")
             raise ValueError("Transcription failed or returned empty/invalid result structure.")

        detected_language = transcription_result_raw.get("language", language)
        logger.info(f"[{job_id}] Detected language: {detected_language}")

        # Prefer 'word_timestamps' key if present, otherwise reconstructed 'all_words'
        word_level_timestamps = transcription_result_raw.get("word_timestamps", [])
            
        if not word_level_timestamps:
            logger.error(f"[{job_id}] No word-level timestamps were generated or extracted by Whisper MLX.")
            # Allow proceeding but with zero segments if transcription otherwise succeeded but no words found
            # raise ValueError("Transcription succeeded but failed to produce word-level timestamps.")

        logger.info(f"[{job_id}] Transcription processing complete. Found {len(word_level_timestamps)} words.")
        transcription_words = word_level_timestamps

        # --- Step 3: Semantic Segmentation --- 
        await send_progress("Segmenting Text...", 60)
        logger.info(f"[{job_id}] Step 3: Segmenting transcription...")
        
        segmentation_start_time = time.time()
        # Pass job_id for logging within segmentation function
        segmented_data = segment_transcription(transcription_words, job_id=job_id)
        segmentation_duration = time.time() - segmentation_start_time
        logger.info(f"[{job_id}] Segmentation finished in {segmentation_duration:.2f}s.")

        if not segmented_data:
            logger.warning(f"[{job_id}] Segmentation resulted in zero segments. Check transcription output and segmentation logic.")
        
        logger.info(f"[{job_id}] Segmentation complete: {len(segmented_data)} segments generated.")

        # --- Step 4 & 5: Cut Audio Chunks & Generate Metadata --- 
        await send_progress(f"Processing Segments (0/{len(segmented_data)})...", 75)
        logger.info(f"[{job_id}] Step 4/5: Cutting audio chunks (ffmpeg) and generating metadata...")
        metadata_records = []
        total_segments = len(segmented_data)
        
        if total_segments == 0:
            logger.warning(f"[{job_id}] No segments to process after segmentation. Skipping chunking and metadata steps.")
        else:
            for i, segment in enumerate(segmented_data):
                segment_start = segment["start_time"]
                segment_end = segment["end_time"]
                segment_text = segment["text"]
                if segment_start >= segment_end or (segment_end - segment_start) < 0.01: 
                    logger.warning(f"[{job_id}] Skipping invalid segment {i}: start={segment_start}, end={segment_end}")
                    continue
                
                chunk_filename = f"segment_{i:05d}.wav"
                chunk_output_path = audio_output_dir / chunk_filename

                ffmpeg_command_cut = [
                    "ffmpeg",
                    "-i", str(extracted_audio_path),
                    "-ss", str(segment_start),
                    "-to", str(segment_end),
                    "-acodec", "pcm_s16le",
                    "-ar", "16000",
                    "-ac", "1",
                    "-y",
                    str(chunk_output_path)
                ]
                
                cut_success = await run_ffmpeg(ffmpeg_command_cut, job_id, f"Audio cutting failed for segment {i}")
                
                if not cut_success or not chunk_output_path.is_file():
                     logger.warning(f"[{job_id}] Failed to create audio chunk: {chunk_filename}")
                     continue

                metadata_record = {
                    "text": segment_text,
                    "start_time": round(segment_start, 3),
                    "end_time": round(segment_end, 3),
                    "duration": round(segment_end - segment_start, 3),
                    "audio_chunk_path": f"audio_chunks/{chunk_filename}", 
                    "source_file": input_file_path.name,
                    "segment_index": i,
                    "total_segments": total_segments,
                    "language": detected_language or "unknown"
                }
                metadata_records.append(metadata_record)
                
                if (i + 1) % 10 == 0 or (i + 1) == total_segments:
                     progress = 75 + int(((i + 1) / total_segments) * 20) # Progress from 75% to 95%
                     await send_progress(f"Processing Segments ({i+1}/{total_segments})...", progress)

            logger.info(f"[{job_id}] Audio chunking and metadata generation complete.")

        # Save the main metadata file (e.g., dataset.jsonl)
        metadata_file_path = metadata_output_dir / "dataset.jsonl"
        try:
            with metadata_file_path.open('w', encoding='utf-8') as f_meta:
                for record in metadata_records:
                    f_meta.write(json.dumps(record, ensure_ascii=False) + '\n')
        except Exception as save_error:
            logger.error(f"[{job_id}] Failed to save metadata file: {save_error}")
            raise
        logger.info(f"[{job_id}] Metadata saved to {metadata_file_path.name}")

        # --- Step 6: Package Results (ZIP) --- 
        await send_progress("Packaging Results (ZIP)...", 98)
        logger.info(f"[{job_id}] Step 6: Creating ZIP archive...")
        final_zip_path = output_dir / f"audio_dataset_{job_id}.zip"
        try:
            with zipfile.ZipFile(final_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                metadata_file_path = metadata_output_dir / "dataset.jsonl"
                if metadata_records:
                    try:
                        with metadata_file_path.open('w', encoding='utf-8') as f_meta:
                            for record in metadata_records:
                                f_meta.write(json.dumps(record, ensure_ascii=False) + '\n')
                        zipf.write(metadata_file_path, arcname=metadata_file_path.name)
                        logger.info(f"[{job_id}] Metadata saved to {metadata_file_path.name} and added to ZIP.")
                    except Exception as save_error:
                        logger.error(f"[{job_id}] Failed to save metadata file: {save_error}")
                        # Decide if this error should halt zip creation
                else:
                     logger.info(f"[{job_id}] No metadata records, skipping metadata file saving/zipping.")

                # Add audio chunks only if they exist
                added_chunks_count = 0
                if audio_output_dir.exists():
                    for chunk_file in audio_output_dir.glob("*.wav"):
                        if chunk_file.is_file():
                             zipf.write(chunk_file, arcname=f"audio_chunks/{chunk_file.name}")
                             added_chunks_count += 1
                        else:
                             logger.warning(f"[{job_id}] Skipping non-file item during zipping: {chunk_file}")
                logger.info(f"[{job_id}] Added {added_chunks_count} audio chunks to ZIP.")
            
            # --- Cleanup --- 
            try:
                logger.info(f"[{job_id}] Cleaning up intermediate files...")
                if audio_extract_dir.exists(): shutil.rmtree(audio_extract_dir)
                if audio_output_dir.exists(): shutil.rmtree(audio_output_dir)
                if metadata_output_dir.exists(): shutil.rmtree(metadata_output_dir)
            except OSError as cleanup_error:
                 logger.warning(f"[{job_id}] Failed to clean up some intermediate files: {cleanup_error}")
            
        except Exception as zip_error:
            logger.error(f"[{job_id}] Failed to create or cleanup after ZIP file: {zip_error}")
            raise

        logger.info(f"[{job_id}] ZIP archive created: {final_zip_path.name}")
        # Step 7 (Making results downloadable) is handled by the API endpoint returning the file

        total_duration = time.time() - start_time
        logger.info(f"[{job_id}] Audio dataset creation finished in {total_duration:.2f} seconds.")
        
        return str(final_zip_path)

    except Exception as e:
        logger.error(f"[{job_id}] Pipeline failed: {type(e).__name__}: {e}", exc_info=True)
        # Send final error status via WebSocket
        await send_progress("Error", 100, details={"error": f"{type(e).__name__}: {e}"})
        # Re-raise the exception to be caught by the background task handler in app.py
        raise
