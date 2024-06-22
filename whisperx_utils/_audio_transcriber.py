import gc
import os
import warnings

import numpy as np
import torch

from whisperx.alignment import align, load_align_model
from whisperx.asr import load_model
from whisperx.audio import load_audio
from whisperx.diarize import DiarizationPipeline, assign_word_speakers
from whisperx.utils import (
    LANGUAGES,
    TO_LANGUAGE_CODE,
    get_writer,
)
from typing import Optional, TYPE_CHECKING, TypeAlias
from typing_extensions import Literal

if TYPE_CHECKING:
    from whisperx.asr import FasterWhisperPipeline, TranscriptionResult

# fmt: off
WhisperModelType: TypeAlias = Literal["tiny", "tiny.en", "base", "base.en", "small", "small.en",
                                      "medium","medium.en","large-v1","large-v2","large-v3","large"
                                ]
OutputFormatType: TypeAlias = Literal["all", "srt", "vtt", "txt", "tsv", "json", "aud"]
LanguageType: TypeAlias = Literal["en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr",
                                 "pl", "ca", "nl", "ar", "sv", "it", "id", "hi", "fi", "vi",
                                 "he", "uk", "el", "ms", "cs", "ro", "da", "hu", "ta", "no",
                                 "th", "ur", "hr", "bg", "lt", "la", "mi", "ml", "cy", "sk",
                                 "te", "fa", "lv", "bn", "sr", "az", "sl", "kn", "et", "mk",
                                 "br", "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw",
                                 "gl", "mr", "pa", "si", "km", "sn", "yo", "so", "af", "oc",
                                 "ka", "be", "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo",
                                 "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my", "bo", "tl",
                                 "mg", "as", "tt", "haw", "ln", "ha", "ba", "jw", "su", "yue"
                                ]
# fmt: on


def transcribe_audio_file(
    audio_path: str,
    model: WhisperModelType = "small",
    model_dir: Optional[str] = None,
    device: Literal["auto", "cpu", "cuda"] = "auto",
    device_index: int = 0,
    batch_size: int = 8,
    compute_type: Literal["float16", "float32", "int8"] = "float16",
    output_dir: str = ".",
    output_format: OutputFormatType = "all",
    verbose: bool = True,
    task: Literal["transcribe", "translate"] = "transcribe",
    language: Optional[LanguageType] = None,
    align_model: Optional[str] = None,
    interpolate_method: Literal["nearest", "linear", "ignore"] = "nearest",
    no_align: bool = False,
    return_char_alignments: bool = False,
    vad_onset: float = 0.500,
    vad_offset: float = 0.363,
    chunk_size: int = 30,
    diarize: bool = False,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
    temperature: float = 0,
    best_of: Optional[int] = 5,
    beam_size: Optional[int] = 5,
    patience: float = 1.0,
    length_penalty: float = 1.0,
    suppress_tokens: str = "-1",
    suppress_numerals: bool = False,
    initial_prompt: Optional[str] = None,
    condition_on_previous_text: bool = False,
    fp16: bool = True,
    temperature_increment_on_fallback: Optional[float] = 0.2,
    compression_ratio_threshold: Optional[float] = 2.4,
    logprob_threshold: Optional[float] = -1.0,
    no_speech_threshold: Optional[float] = 0.6,
    max_line_width: Optional[int] = None,
    max_line_count: Optional[int] = None,
    highlight_words: bool = False,
    segment_resolution: Literal["sentence", "chunk"] = "sentence",
    threads: Optional[int] = 0,
    hf_token: Optional[str] = None,
    print_progress: bool = False,
):
    """
    Transcribe and process an audio file using WhisperX.

    Args:
        audio_path (str):
            Path to the audio file.
        model (WhisperModelType, optional):
            Name of the Whisper model to use. Defaults to "small".
        model_dir (Optional[str], optional):
            The path to save model files; uses ~/.cache/whisper by default. Defaults to None.
        device (Literal["auto","cpu", "cuda"], optional):
            Device to use for PyTorch inference. Defaults to "auto".
        device_index (int, optional):
            Device index to use for FasterWhisper inference. Defaults to 0.
        batch_size (int, optional):
            The preferred batch size for inference. Defaults to 8.
        compute_type (Literal["float16", "float32", "int8"], optional):
            Compute type for computation. Defaults to "float16".
        output_dir (str, optional):
            Directory to save outputs. Defaults to ".".
        output_format (OutputFormatType, optional):
            Format of the output file; if not specified, all available formats will be produced. Defaults to "all".
        verbose (bool, optional):
            Whether to print out progress and debug messages. Defaults to True.
        task (Literal["transcribe", "translate"], optional):
            Whether to perform X->X speech recognition ("transcribe") or X->English translation ("translate"). Defaults to "transcribe".
        language (Optional[LanguageType], optional):
            Language spoken in the audio. Defaults to None.
        align_model (Optional[str], optional):
            Name of phoneme-level ASR model to do alignment. Defaults to None.
        interpolate_method (Literal["nearest", "linear", "ignore"], optional):
            Method for assigning timestamps to non-aligned words, or merging them into neighboring. Defaults to "nearest".
        no_align (bool, optional):
            Do not perform phoneme alignment. Defaults to False.
        return_char_alignments (bool, optional):
            Return character-level alignments in the output JSON file. Defaults to False.
        vad_onset (float, optional):
            Onset threshold for VAD (see pyannote.audio); reduce this if speech is not being detected. Defaults to 0.500.
        vad_offset (float, optional):
            Offset threshold for VAD (see pyannote.audio); reduce this if speech is not being detected. Defaults to 0.363.
        chunk_size (int, optional):
            Chunk size for merging VAD segments. Defaults to 30.
        diarize (bool, optional):
            Apply diarization to assign speaker labels to each segment/word. Defaults to False.
        min_speakers (Optional[int], optional):
            Minimum number of speakers in the audio file. Defaults to None.
        max_speakers (Optional[int], optional):
            Maximum number of speakers in the audio file. Defaults to None.
        temperature (float, optional):
            Temperature to use for sampling. Defaults to 0.
        best_of (Optional[int], optional):
            Number of candidates when sampling with non-zero temperature. Defaults to 5.
        beam_size (Optional[int], optional):
            Number of beams in beam search (applies only with zero temperature). Defaults to 5.
        patience (float, optional):
            Optional patience value to use in beam decoding; the default (1.0) is equivalent to conventional beam search. Defaults to 1.0.
        length_penalty (float, optional):
            Optional token length penalty coefficient (alpha); uses simple length normalization by default. Defaults to 1.0.
        suppress_tokens (str, optional):
            Comma-separated list of token IDs to suppress during sampling; "-1" will suppress most special characters except common punctuations. Defaults to "-1".
        suppress_numerals (bool, optional):
            Whether to suppress numeric symbols and currency symbols during sampling. Defaults to False.
        initial_prompt (Optional[str], optional):
            Optional text to provide as a prompt for the first window. Defaults to None.
        condition_on_previous_text (bool, optional):
            If True, provide the previous output of the model as a prompt for the next window; disabling may make the text inconsistent across windows, but the model becomes less prone to getting stuck in a failure loop. Defaults to False.
        fp16 (bool, optional):
            Whether to perform inference in FP16. Defaults to True.
        temperature_increment_on_fallback (Optional[float], optional):
            Temperature to increase when falling back if decoding fails. Defaults to 0.2.
        compression_ratio_threshold (Optional[float], optional):
            If the gzip compression ratio is higher than this value, treat the decoding as failed. Defaults to 2.4.
        logprob_threshold (Optional[float], optional):
            If the average log probability is lower than this value, treat the decoding as failed. Defaults to -1.0.
        no_speech_threshold (Optional[float], optional):
            If the probability of the token is higher than this value AND the decoding has failed due to `logprob_threshold`, consider the segment as silence. Defaults to 0.6.
        max_line_width (Optional[int], optional):
            The maximum number of characters in a line before breaking the line. Defaults to None.
        max_line_count (Optional[int], optional):
            The maximum number of lines in a segment. Defaults to None.
        highlight_words (bool, optional):
            Highlight words in the output. Defaults to False.
        segment_resolution (Literal["sentence", "chunk"], optional):
            Segment resolution for splitting long chunks. Defaults to "sentence".
        threads (Optional[int], optional):
            Number of threads to use for inference. Defaults to 0.
        hf_token (Optional[str], optional):
            HuggingFace API token. Defaults to None.
        print_progress (bool, optional):
            Print progress information. Defaults to False.

    """

    os.makedirs(output_dir, exist_ok=True)
    # デバイスがautoに設定されていた場合には、GPUを優先使用する
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if task == "translate":
        no_align = True

    if language is not None:
        language = language.lower()
        if language not in LANGUAGES:
            if language in TO_LANGUAGE_CODE:
                language = TO_LANGUAGE_CODE[language]
            else:
                raise ValueError(f"Unsupported language: {language}")

    if model.endswith(".en") and language != "en":
        if language is not None:
            warnings.warn(
                f"{model} is an English-only model but received '{language}'; using English instead."
            )
        language = "en"
    # default to loading english if not specified
    align_language = language if language is not None else "en"

    if (increment := temperature_increment_on_fallback) is not None:
        temperature = tuple(np.arange(temperature, 1.0 + 1e-6, increment))
    else:
        temperature = [temperature]

    faster_whisper_threads = 4
    if threads > 0:
        torch.set_num_threads(threads)
        faster_whisper_threads = threads

    asr_options = {
        "beam_size": beam_size,
        "patience": patience,
        "length_penalty": length_penalty,
        "temperatures": temperature,
        "compression_ratio_threshold": compression_ratio_threshold,
        "log_prob_threshold": logprob_threshold,
        "no_speech_threshold": no_speech_threshold,
        "condition_on_previous_text": condition_on_previous_text,
        "initial_prompt": initial_prompt,
        "suppress_tokens": [int(x) for x in suppress_tokens.split(",")],
        "suppress_numerals": suppress_numerals,
    }

    writer = get_writer(output_format, output_dir)
    if no_align:
        if highlight_words:
            raise ValueError(
                "NO_ALIGNが指定されている場合には、HIGHLIGHT_WORDSは使用できません"
            )
        if max_line_count is not None:
            raise ValueError(
                "NO_ALIGNが指定されている場合には、MAX_LINE_COUNTは使用できません"
            )
        if max_line_width is not None:
            raise ValueError(
                "NO_ALIGNが指定されている場合には、MAX_LINE_WIDTHは使用できません"
            )

    if max_line_count and not max_line_width:
        warnings.warn("MAX_LINE_COUNT has no effect without MAX_LINE_WIDTH")
    writer_args = {
        "highlight_words": highlight_words,
        "max_line_count": max_line_count,
        "max_line_width": max_line_width,
    }

    model: "FasterWhisperPipeline" = load_model(
        model,
        device=device,
        device_index=device_index,
        download_root=model_dir,
        compute_type=compute_type,
        language=language,
        asr_options=asr_options,
        vad_options={"vad_onset": vad_onset, "vad_offset": vad_offset},
        task=task,
        threads=faster_whisper_threads,
    )

    audio = load_audio(audio_path)
    print(">>Performing transcription...")
    result: "TranscriptionResult" = model.transcribe(
        audio,
        batch_size=batch_size,
        chunk_size=chunk_size,
        print_progress=print_progress,
    )

    # Unload Whisper and VAD
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # Part 2: Align Loop
    if not no_align:
        align_model, align_metadata = load_align_model(
            align_language, device, model_name=align_model
        )
        if align_model is not None and len(result["segments"]) > 0:
            if result.get("language", "en") != align_metadata["language"]:
                print(
                    f"New language found ({result['language']})! Previous was ({align_metadata['language']}), loading new alignment model for new language..."
                )
                align_model, align_metadata = load_align_model(
                    result["language"], device
                )
            print(">>Performing alignment...")
            result = align(
                result["segments"],
                align_model,
                align_metadata,
                audio,
                device,
                interpolate_method=interpolate_method,
                return_char_alignments=return_char_alignments,
                print_progress=print_progress,
            )

        del align_model
        gc.collect()
        torch.cuda.empty_cache()

    if diarize:
        if hf_token is None:
            print(
                "Warning, no `hf_token` used, needs to be saved in environment variable, otherwise will throw error loading diarization model..."
            )
        print(">>Performing diarization...")
        diarize_model = DiarizationPipeline(use_auth_token=hf_token, device=device)
        diarize_segments = diarize_model(
            audio_path, min_speakers=min_speakers, max_speakers=max_speakers
        )
        result = assign_word_speakers(diarize_segments, result)

    # >> Write
    result["language"] = align_language
    writer(result, audio_path, writer_args)
    return result
