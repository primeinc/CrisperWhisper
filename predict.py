from cog import BasePredictor, Input, Path
import torch
import torchaudio
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
            if pause_duration > split_threshold:
                distribute = split_threshold / 2
            else:
                distribute = pause_duration / 2

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

        self.model_id = "nyrahealth/CrisperWhisper"
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_id, torch_dtype=self.torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        )
        self.model.to(self.device)

        self.processor = AutoProcessor.from_pretrained(self.model_id)

        # Initialize the pipeline
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

    def predict(self, audio: Path = Input(description="Audio file to transcribe")) -> str:
        """Run a prediction on the CrisperWhisper model"""
        try:
            # Load the audio file
            waveform, _ = torchaudio.load(audio)

            # Pass the waveform to the pipeline
            hf_pipeline_output = self.pipe(waveform)
            
            # Adjust the pauses and return the refined result
            crisper_whisper_result = adjust_pauses_for_hf_pipeline_output(hf_pipeline_output)
            return crisper_whisper_result

        except Exception as e:
            return f"Error during transcription: {str(e)}"