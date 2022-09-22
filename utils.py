import json
import logging
import pickle
import shutil
import subprocess
from logging.handlers import RotatingFileHandler
from pathlib import Path
from shutil import rmtree

import ctc_segmentation
import shlex
import librosa
import numpy as np
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2Tokenizer, BertTokenizerFast

from config import *


def init_logger():
    # The ctc_segmentation module, which runs in a subprocess, calls
    # logging.debug(), which in turn calls logging.basicConfig(), which adds a
    # handler to the root logger if none exists. This is explicitly suggested
    # against in python docs here: https://docs.python.org/3/library/logging.html#logging.basicConfig
    # In this case, it results in duplicate logging messages from an extra
    # handler if we init logger with custom names other than the root logger.
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        fmt = '[%(asctime)s] - %(levelname)s - %(message)s',
        datefmt = "%Y-%m-%d %H:%M:%S"
    )
    fh = RotatingFileHandler(LOG_FILE_PATH, maxBytes=2e7, backupCount=5)
    fh.setFormatter(formatter)
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # set numba logger to warning to prevent flooding messages when calling librosa.load()
    logging.getLogger('numba').setLevel(logging.WARNING)
    return logger

def clear_media_folder():
    logger = logging.getLogger()
    logger.info("Cleaning temp files...")
    for path in MEDIA_PATH.glob("**/*"):
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            rmtree(path)

def execute_cmd(cmd: str) -> bytes:
    args = shlex.split(cmd)
    p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    if p.returncode != 0:
        raise RuntimeError("ffmpeg", out, err)
    return out

def do_asr(
    audio_file : str | Path, 
    processor : Wav2Vec2Processor,
    model : Wav2Vec2ForCTC,
) -> str:
    try:
        sample_rate = 16000
        audio, _ = librosa.load(audio_file, sr=sample_rate)
        features = processor(
            audio, 
            sampling_rate=sample_rate,
            return_tensors="pt", 
            padding="longest"
        )
        with torch.no_grad():
            logits = model(features.input_values.to(DEVICE)).logits.cpu()[0]

        predicted_ids = torch.argmax(logits, dim=-1)
        asr_text = processor.decode(predicted_ids)
        return asr_text
    except Exception as e:
        raise e

def do_alignment(
    audio_file : str | Path, 
    transcript : str,
    processor : Wav2Vec2Processor,
    tokenizer : Wav2Vec2Tokenizer,
    model : Wav2Vec2ForCTC,
) -> list[dict]:
    try:
        audio, _ = librosa.load(audio_file, sr=16000)
        features = processor(
            audio, 
            sampling_rate=16000, 
            return_tensors="pt", 
            padding="longest"
        )
        with torch.no_grad():
            logits = model(features.input_values.to(DEVICE)).logits.cpu()[0]
            probs = torch.nn.functional.log_softmax(logits,dim=-1)

        # Tokenize transcripts
        transcripts = transcript.split()
        vocab = tokenizer.get_vocab()
        inv_vocab = {v:k for k,v in vocab.items()}
        unk_id = vocab["<unk>"]
        tokens = []
        for transcript in transcripts:
            assert len(transcript) > 0
            tok_ids = tokenizer(transcript.lower())['input_ids']
            tok_ids = np.array(tok_ids,dtype=np.int32)
            tokens.append(tok_ids[tok_ids != unk_id])
        
        # Do align
        char_list = [inv_vocab[i] for i in range(len(inv_vocab))]
        config = ctc_segmentation.CtcSegmentationParameters(char_list=char_list)
        config.index_duration = audio.shape[0] / probs.size()[0] / 16000
        
        ground_truth_mat, utt_begin_indices = ctc_segmentation.prepare_token_list(config, tokens)
        timings, char_probs, state_list = ctc_segmentation.ctc_segmentation(config, probs.numpy(), ground_truth_mat)
        segments = ctc_segmentation.determine_utterance_segments(config, utt_begin_indices, char_probs, timings, transcripts)
        return [{"text" : t, "start" : p[0], "end" : p[1], "conf" : np.exp(p[2])} for t,p in zip(transcripts, segments)]
    except Exception as e:
        raise e

def data_defence(video_id: str, defence_methods: list[str]) -> bool:
    logger = logging.getLogger()
    modified_video_path = MEDIA_PATH / video_id / "modified_video.mp4"
    defended_video_path = MEDIA_PATH / video_id / "defended_video.mp4"
    defended_video_tmp = MEDIA_PATH / video_id / "defended_video_tmp.mp4"
    defended = False
    shutil.copyfile(modified_video_path, defended_video_path)
    for method in defence_methods:
        if method == "a_denoise":
            logger.info("Audio Denoising...")
            cmd = f"ffmpeg -i {defended_video_path} -af afftdn=nr=40:nf=-20:tn=1 -c:v copy -y {defended_video_tmp}"
            execute_cmd(cmd)
            shutil.copyfile(defended_video_tmp, defended_video_path)
            defended_video_tmp.unlink()
            defended = True
        elif method == "v_reconstruct":
            logger.info("Video MCI...")
            cmd = f"ffmpeg -i {defended_video_path} -vf blackframe=0,metadata=select:key=lavfi.blackframe.pblack:value=90:function=less,minterpolate=mi_mode=mci -c:a copy -y {defended_video_tmp}"
            execute_cmd(cmd)
            shutil.copyfile(defended_video_tmp, defended_video_path)
            defended_video_tmp.unlink()
            defended = True
    return defended

def feature_defence(video_id: str, defence_methods: list[str], data_defended: bool, word_ids: list[int]) -> bool:
    modified_feature = MEDIA_PATH / video_id / "feat_modified.pkl"
    defended_feature = MEDIA_PATH / video_id / "feat_defended.pkl"
    video_edit_file = MEDIA_PATH / video_id / "edit_video.json"
    audio_edit_file = MEDIA_PATH / video_id / "edit_audio.json"
    defended = False
    if not data_defended:
        shutil.copyfile(modified_feature, defended_feature)
    for method in defence_methods:
        if method == "f_interpol":
            need_vf_defend = False if "v_reconstruct" in defence_methods else True
            with open(defended_feature, "rb") as f:
                feat = pickle.load(f)
            with open(video_edit_file, "r") as f:
                video_edit = json.load(f)
            with open(audio_edit_file, "r") as f:
                audio_edit = json.load(f)
            v_edit_ids = [v[0] for v in video_edit]
            a_edit_ids = [a[0] for a in audio_edit]
            v_edit_mask = []
            a_edit_mask = []
            for v in word_ids:
                if v in v_edit_ids:
                    v_edit_mask.append(1)
                else:
                    v_edit_mask.append(0)
                if v in a_edit_ids:
                    a_edit_mask.append(1)
                else:
                    a_edit_mask.append(0)
            # audio interpolation
            i, start_idx, end_idx = 0, -1, -1
            while i < len(a_edit_mask):
                if a_edit_mask[i] == 1:
                    start_idx = i - 1
                    while i < len(a_edit_mask) and a_edit_mask[i] == 1:
                        i += 1
                    end_idx = i
                    start_f = feat['audio'][start_idx]
                    end_f = feat['audio'][end_idx]
                    delta = end_f - start_f
                    for idx in range(1, end_idx - start_idx):
                        feat['audio'][start_idx + idx] = start_f + delta * idx / float(end_idx - start_idx)
                i += 1
            # video interpolation
            if need_vf_defend:
                i, start_idx, end_idx = 0, -1, -1
                while i < len(v_edit_mask):
                    if v_edit_mask[i] == 1:
                        start_idx = i - 1
                        while i < len(v_edit_mask) and v_edit_mask[i] == 1:
                            i += 1
                        end_idx = i
                        start_f = feat['vision'][start_idx]
                        end_f = feat['vision'][end_idx]
                        delta = end_f - start_f
                        for idx in range(1, end_idx - start_idx):
                            feat['vision'][start_idx + idx] = start_f + delta * idx / float(end_idx - start_idx)
                    i += 1
            defended = True
    return defended

def get_word_ids(text_file : str, tokenizer : BertTokenizerFast) -> list[int]:
    text = open(text_file, "r").read()
    encoding = tokenizer(text.split(), is_split_into_words=True)
    return encoding.word_ids()