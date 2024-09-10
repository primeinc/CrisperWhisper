from cog import BasePredictor, Input, Path
import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np
import subprocess
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

def adjust_pauses_for_hf_pipeline_output(pipeline_output, split_threshold=0.12):
    """
    Adjust pause timings by distributing pauses up to the threshold evenly between adjacent words.
    """
    adjusted_chunks = pipeline_output["chunks"].copy()

    for i in range(len(adjusted_chunks) - 1):
        current_chunk = adjusted_chunks[i]
        next_chunk = adjusted_chunks[i + 1]

        current_start, current_end = current_chunk["timestamp"]
        next_start, next_end = next_chunk["timestamp"]
        pause_duration = next_start - current_end

        if pause_duration > 0:
            distribute = min(pause_duration / 2, split_threshold / 2)

            # Adjust current chunk end time
            adjusted_chunks[i]["timestamp"] = (current_start, current_end + distribute)

            # Adjust next chunk start time
            adjusted_chunks[i + 1]["timestamp"] = (next_start - distribute, next_end)

    pipeline_output["chunks"] = adjusted_chunks
    return pipeline_output


class Predictor(BasePredictor):
    def setup(self):
        """Load the CrisperWhisper model into memory"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        print(f"[INFO] Device: {self.device}")
        print(f"[INFO] Torch dtype: {self.torch_dtype}")

        self.model_id = "nyrahealth/CrisperWhisper"
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_id, torch_dtype=self.torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        )
        self.model.to(self.device)

        self.processor = AutoProcessor.from_pretrained(self.model_id)

        print(f"[INFO] Model Version: {self.model.config._name_or_path}")
        print(f"[INFO] Model Hash: {getattr(self.model.config, 'model_hash', 'Unknown')}")

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            chunk_length_s=30,
            batch_size=16,
            return_timestamps='word',
            torch_dtype=self.torch_dtype,
            device=self.device,
        )
        print(f"[INFO] Pipeline initialized.")

    def predict(self, audio: Path = Input(description="Audio file to transcribe")) -> str:
        """Run a prediction on the CrisperWhisper model"""
        try:
            print(f"[INFO] Loading audio file: {audio}")
            waveform, sample_rate = torchaudio.load(audio)

            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            if sample_rate != 16000:
                waveform = T.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)

            waveform = waveform.squeeze().cpu().numpy()

            hf_pipeline_output = self.pipe(waveform)

            crisper_whisper_result = adjust_pauses_for_hf_pipeline_output(hf_pipeline_output)
            
            # Return the transcription
            transcription = " ".join([chunk['text'] for chunk in crisper_whisper_result["chunks"]])
            print(f"[INFO] Transcription successful.")
            return transcription

        except Exception as e:
            print(f"[ERROR] Error during transcription: {str(e)}")
            return f"Error during transcription: {str(e)}"
        
    def post_process(self):
        """Run some final diagnostics and log the package versions."""
        # Print the installed packages
        print(f"[INFO] Running `pip freeze` to list installed packages:")
        pip_freeze_output = subprocess.run(["pip", "freeze"], capture_output=True, text=True)
        print(pip_freeze_output.stdout)

        # Print model details
        print(f"[INFO] Model details: {self.model.config}")