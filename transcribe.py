'''
File: transcribe.py
Author: Amber Charlotte Converse
Purpose: Model file for automatic transcription of Spanish audio to segment IPA transcription aligned in a TextGrid file to
    facilitate manual phonemic transcription.

    Usage:

    python3 transcribe.py <audio file> -o <output dir>
'''
import os
import argparse
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import subprocess
import librosa

def load_audio(file_path, target_sr=16000):
    '''
    Load audio array from file
    '''
    waveform, sample_rate = librosa.load(file_path, sr=target_sr)  # Automatically resamples
    return waveform  # Librosa returns a NumPy array

def transcribe_audio(audio_path, model_name="openai/whisper-large-v3"):
    '''
    Transcribe audio using Whisper.
    '''
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "mps" if torch.backends.mps.is_available() and not device == "cuda" else "cpu"
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)
    
    # Load and process audio
    audio_array = load_audio(audio_path)
    input_features = processor(audio_array, return_tensors="pt", sampling_rate=16000).input_features.to(device)
    
    # Generate transcription
    with torch.no_grad():
        predicted_ids = model.generate(input_features)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    
    return transcription

def generate_ipa_transcription(text, g2p_model_path="spanish_4_3_2.fst"):
    '''
    Convert transcribed text to IPA using Phonetisaurus.
    '''
    ipa_transcription = []
    for word in text.split():
        command = ["phonetisaurus-g2pfst", f"--model={g2p_model_path}", f"--word={word}"]
        result = subprocess.run(command, capture_output=True, text=True)
        ipa_transcription.append(' '.join(result.stdout.strip().split()[2:]))
    return " ".join(ipa_transcription)

def align_with_mfa(audio_path, transcript_path, dictionary_path, acoustic_model_path, output_dir):
    '''
    Run MFA to align words and phonemes with audio.
    '''
    os.makedirs(output_dir, exist_ok=True)
    command = [
        "mfa", "align", audio_path, transcript_path, dictionary_path, acoustic_model_path, output_dir, "--output_format", "textgrid"
    ]
    subprocess.run(command)
    return os.path.join(output_dir, "aligned.TextGrid")

def transcribe(audio_file, dictionary_path, acoustic_model_path, output_dir):
    '''
    Run complete speech -> transcription pipeline
    '''

    txt_transcription = transcribe_audio(audio_file)
    print("Text transcription:", txt_transcription)
    ipa_transcription = generate_ipa_transcription(txt_transcription)
    print("IPA Transcription:", ipa_transcription)

    # Save text transcription for MFA
    with open("transcript.txt", "w", encoding="utf-8") as f:
        f.write(txt_transcription)

    # Align with MFA
    textgrid_file = align_with_mfa(audio_file, "transcript.txt", dictionary_path, acoustic_model_path, output_dir)
    print("Alignment saved to:", textgrid_file)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        prog="Transcribe",
        description="Model file for automatic transcription of Spanish audio to segment IPA transcription aligned in a TextGrid file to facilitate manual phonemic transcription.")

    parser.add_argument("audio_file")
    parser.add_argument("-o", "--output_dir", default="aligned_output")

    args = parser.parse_args()

    dictionary_path = "spanish_mfa.dict"
    acoustic_model_path = "spanish_mfa.zip"

    transcribe(args.audio_file, dictionary_path, acoustic_model_path, args.output_dir)
    