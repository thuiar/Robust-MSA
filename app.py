import datetime
import json
import logging
import os
import random
import shutil
from pathlib import Path

# import transformers
from flask import Flask, request
from flask_cors import CORS
from MMSA import MMSA_test, get_config_regression
from MSA_FET import FeatureExtractionTool, get_default_config
from transformers import (Wav2Vec2CTCTokenizer, Wav2Vec2ForCTC,
                          Wav2Vec2Processor, BertTokenizerFast)

from config import *
from utils import *

os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES

app = Flask(__name__)
app.config.from_object(APP_SETTINGS)
CORS(app, supports_credentials=True)

WAV2VEC_PROCESSER = Wav2Vec2Processor.from_pretrained(WAV2VEC_MODEL_NAME)
WAV2VEC_TOKENIZER = Wav2Vec2CTCTokenizer.from_pretrained(WAV2VEC_MODEL_NAME)
WAV2VEC_MODEL = Wav2Vec2ForCTC.from_pretrained(WAV2VEC_MODEL_NAME).to(DEVICE)
cfg = get_default_config('bert+opensmile+openface')
cfg['text']['device'] = 'cuda:3'
cfg['video']['fps'] = 30
cfg['video']['args'] = {
    "hogalign": False,
    "simalign": False,
    "nobadaligned": False,
    "landmark_2D": True,
    "landmark_3D": False,
    "pdmparams": False,
    "head_pose": False,
    "action_units": True,
    "gaze": False,
    "tracked": False
}
FET = FeatureExtractionTool(cfg)
BERT_TOKENIZER = BertTokenizerFast.from_pretrained("bert-base-uncased")

logger = logging.getLogger()

@app.route('/test')
def test():
    logger.info("API called: /test")
    return {"code": SUCCESS_CODE, "msg": "success"}

@app.route('/uploadVideo', methods=['POST'])
def upload_video():
    logger.info("API called: /uploadVideo")
    try:
        file = request.files.get('video')
        now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        video_id = now + str(random.randint(0, 999)).rjust(3, '0')
        save_path = Path(MEDIA_PATH) / video_id
        save_path.mkdir(parents=True, exist_ok=False)
        original_file = save_path / f"original_{file.filename}"
        file.save(original_file)
        # convert to mp4 using ffmpeg
        logger.info("Converting video to mp4...")
        cmd = f"ffmpeg -i {original_file} -c:v libx264 -c:a aac -y {save_path / 'raw_video.mp4'}"
        execute_cmd(cmd)
        logger.info("Conversion done.")
    except Exception as e:
        logger.exception(e)
        return {"code": ERROR_CODE, "msg": str(e)}
    return {"code": SUCCESS_CODE, "msg": "success", "id": video_id, "path": str(save_path)}

@app.route('/callASR', methods=['POST'])
def call_ASR():
    logger.info("API called: /callASR")
    try:
        video_id = json.loads(request.get_data())['videoID']
        video_file = Path(MEDIA_PATH) / video_id / "raw_video.mp4"
        # extract audio
        audio_save_path = Path(MEDIA_PATH) / video_id / "audio.wav"
        logger.info("Extracting audio...")
        cmd = f"ffmpeg -i {video_file} -vn -acodec pcm_s16le -ac 1 -y {audio_save_path}"
        execute_cmd(cmd)
        logger.info("Extraction done.")
        # do ASR
        logger.info("Doing ASR...")
        transcript = do_asr(audio_save_path, WAV2VEC_PROCESSER, WAV2VEC_MODEL)
        logger.info("ASR done.")
    except Exception as e:
        logger.exception(e)
        return {"code": ERROR_CODE, "msg": str(e)}
    return {"code": SUCCESS_CODE, "msg": "success", "result": transcript}

@app.route('/uploadTranscript', methods=['POST'])
def upload_transcript():
    logger.info("API called: /uploadTranscript")
    try:
        data = json.loads(request.get_data())
        transcript = data['transcript']
        video_id = data['videoID']
        audio_save_path = MEDIA_PATH / video_id / "audio.wav"
        text_save_path = MEDIA_PATH / video_id / "transcript.txt"
        video_path = MEDIA_PATH / video_id / "raw_video.mp4"
        annotated_video_path = MEDIA_PATH / video_id / "annotated_video.mp4"
        with open(text_save_path, 'w') as f:
            f.write(transcript)
        # do alignment
        logger.info("Doing alignment...")
        aligned_results = do_alignment(audio_save_path, transcript, WAV2VEC_PROCESSER, WAV2VEC_TOKENIZER, WAV2VEC_MODEL)
        with open(MEDIA_PATH / video_id / "aligned_results.json", 'w') as f:
            json.dump(aligned_results, f)
        # new_transcript = " ".join([x['text'] for x in aligned_results])
        # with open(text_save_path, 'w') as f:
        #     f.write(new_transcript)
        logger.info("Alignment done.")
        # annotate video with OpenFace
        # logger.info("Annotating video with OpenFace...")
        # res = openfaceExtractor.get_annotated_video(str(video_path), str(annotated_video_path))
        # logger.debug(res) # BUG: OpenFace won't set return code >0 on error
        # logger.info("Annotation done.")    
    except Exception as e:
        logger.exception(e)
        return {"code": ERROR_CODE, "msg": str(e)}
    return {"code": SUCCESS_CODE, "msg": "success", "align": aligned_results}

@app.route('/videoEditAligned', methods=['POST'])
def edit_video_aligned():
    logger.info("API called: /videoEditAligned")
    try:
        data = json.loads(request.get_data())
        video_id = data['videoID']
        words = data['words']
        raw_video_path = MEDIA_PATH / video_id / "raw_video.mp4"
        modified_video_path = MEDIA_PATH / video_id / "modified_video.mp4"
        modified_video_tmp = MEDIA_PATH / video_id / "modified_video_tmp.mp4"
        shutil.copyfile(raw_video_path, modified_video_path)
        with open(MEDIA_PATH / video_id / "aligned_results.json") as f:
            aligned_results = json.load(f)
        # re-organize
        video_modify, audio_modify, noise, volume, text_modify = [], [], [], [], []
        for word in words:
            if word[1] == 'v':
                video_modify.append([word[0], word[2], word[3]])
            elif word[1] == 'a':
                if 'noise' in word[2]:
                    noise.append(word[2])
                    volume.append(str(word[3]))
                else:
                    audio_modify.append([word[0], word[2], word[3]])
            elif word[1] == 't':
                text_modify.append([word[0], word[2], word[3]])
        # edit video
        if len(video_modify) > 0:
            logger.info("Editing video...")
            cmd = f"ffmpeg -i {modified_video_path} -filter_complex \"[0:v]"
            sigma_dict = {
                'low': 5,
                'medium': 10,
                'high': 20
            }
            for i, v in enumerate(video_modify):
                word_id = v[0]
                start = aligned_results[word_id]['start']
                end = aligned_results[word_id]['end']
                if v[1] == 'blank':
                    if i != 0:
                        cmd += f'[v{i-1}]'
                    cmd += f"drawbox=enable='between(t,{start},{end})':color=black:t=fill[v{i}];"
                elif v[1] == 'gblur':
                    if i != 0:
                        cmd += f'[v{i-1}]'
                    cmd += f"gblur=sigma={sigma_dict[v[2]]}:enable='between(t,{start},{end})'[v{i}];"
            cmd = cmd[:-1] + f"\" -map \"[v{len(video_modify)-1}]\" -map 0:a -c:a copy -y {modified_video_tmp}"
            logger.debug(cmd)
            execute_cmd(cmd)
            shutil.copyfile(modified_video_tmp, modified_video_path)
            modified_video_tmp.unlink()
        # edit audio
        if len(audio_modify) > 0:
            logger.info("Editing audio...")
            cmd = f"ffmpeg -i {modified_video_path} -filter_complex \"[0:a]"
            for i, a in enumerate(audio_modify):
                word_id = a[0]
                if a[1] == 'mute':
                    start = aligned_results[word_id]['start']
                    end = aligned_results[word_id]['end']
                    if i != 0:
                        cmd += f'[v{i-1}]'
                    cmd += f"volume=enable='between(t,{start},{end})':volume=0[a{i}];"
            cmd = cmd[:-1] + f"\" -map 0:v -map \"[a{len(audio_modify)-1}]\" -c:v copy -y {modified_video_tmp}"
            logger.debug(cmd)
            execute_cmd(cmd)
            shutil.copyfile(modified_video_tmp, modified_video_path)
            modified_video_tmp.unlink()
        # add noise
        if len(noise) > 0:
            logger.info("Adding noise...")
            cmd = f"ffmpeg -i {modified_video_path} "
            for i in range(len(noise)):
                noise_path = Path(__file__).parent / "assets" / "noise" / f"{noise[i][6:]}.wav"
                cmd += f"-i {noise_path} "
            cmd += f"-filter_complex \"[0:a]aresample=44100[0a];"
            for i in range(len(noise)):
                cmd += f"[{i+1}:a]aresample=44100[{i+1}a];"
            cmd += "[0a]"
            for i in range(len(noise)):
                cmd += f"[{i+1}a]"
            cmd += f"amix=inputs={len(noise)+1}:duration=shortest:weights='1 {' '.join(volume)}':dropout_transition=0:normalize=0[aout]\" -map 0:v -map \"[aout]\" -c:v copy -y {modified_video_tmp}"
            logger.debug(cmd)
            execute_cmd(cmd)
            shutil.copyfile(modified_video_tmp, modified_video_path)
            modified_video_tmp.unlink()
        # edit text
        for t in text_modify:
            word_id = t[0]
            if t[1] == 'replace':
                aligned_results[word_id]['word'] = t[2]
            elif t[1] == 'remove':
                aligned_results[word_id]['word'] = '[UNK]' # ???
        transcript = " ".join([w['text'] for w in aligned_results])
        with open(MEDIA_PATH / video_id / "transcript_modified.txt", 'w') as f:
            f.write(transcript)
        with open(MEDIA_PATH / video_id / "aligned_modified.json", 'w') as f:
            json.dump(aligned_results, f)
    except Exception as e:
        logger.exception(e)
        return {"code": ERROR_CODE, "msg": str(e)}
    return {"code": SUCCESS_CODE, "msg": "success"}

@app.route('/runMSAAligned', methods=['POST'])
def run_msa_aligned():
    logger.info("API called: /runMSAAligned")
    try:
        data = json.loads(request.get_data())
        video_id = data['videoID']
        models = data['models']
        defence = data['defence']
        video_original = MEDIA_PATH / video_id / "raw_video.mp4"
        video_modified = MEDIA_PATH / video_id / "modified_video.mp4"
        video_defended = MEDIA_PATH / video_id / "defended_video.mp4"
        feat_original = MEDIA_PATH / video_id / "feat_original.pkl"
        feat_modified = MEDIA_PATH / video_id / "feat_modified.pkl"
        feat_defended = MEDIA_PATH / video_id / "feat_defended.pkl"
        trans_original = MEDIA_PATH / video_id / "transcript.txt"
        trans_modified = MEDIA_PATH / video_id / "transcript_modified.txt"
        weights_root = Path(__file__).parent / "assets" / "weights"
        
        # data defence
        logger.info("Data-level Defence...")
        data_defence(video_id, defence)
        # original
        logger.info("Extracting features from original video...")
        f_original = FET.run_single(video_original, feat_original, text_file=trans_original)
        # modified
        logger.info("Extracting features from defended video...")
        f_defended = FET.run_single(video_defended, feat_modified, text_file=trans_modified)
        # feature defence
        logger.info("Feature-level Defence...")
        feature_defence(video_id, defence)
        # explain features
        word_ids_original = get_word_ids(trans_original, BERT_TOKENIZER)
        last_word_id = -1
        remove_ids = []
        for i, v in enumerate(word_ids_original):
            if v is None: # start or end token
                continue
            if v == last_word_id: # same word, multiple tokens
                remove_ids.append(i)
            last_word_id = v
        for m in ['audio', 'vision', 'text']:
            f_original[m] = np.delete(f_original[m], remove_ids, axis=0) # remove repeated feature
            f_original[m] = f_original[m][1:-1] # remove start and end token
        
        word_ids_modified = get_word_ids(trans_modified, BERT_TOKENIZER)
        last_word_id = -1
        remove_ids = []
        for i, v in enumerate(word_ids_modified):
            if v is None:
                continue
            if v == last_word_id:
                remove_ids.append(i)
            last_word_id = v
        for m in ['audio', 'vision', 'text']:
            f_defended[m] = np.delete(f_defended[m], remove_ids, axis=0)
            f_defended[m] = f_defended[m][1:-1]
        
        feat = {
            "original": {
                "loudness": f_original['audio'][:, 0].tolist(),
                "alphaRatio": f_original['audio'][:, 1].tolist(),
                "mfcc1": f_original['audio'][:, 6].tolist(),
                "mfcc2": f_original['audio'][:, 7].tolist(),
                "mfcc3": f_original['audio'][:, 8].tolist(),
                "mfcc4": f_original['audio'][:, 9].tolist(),
                "pitch": f_original['audio'][:, 10].tolist(),
                "HNR": f_original['audio'][:, 13].tolist(),
                "F1frequency": f_original['audio'][:, 16].tolist(),
                "F1bandwidth": f_original['audio'][:, 17].tolist(),
                "F1amplitude": f_original['audio'][:, 18].tolist(),
                "AU01_r": f_original['vision'][:, -35].tolist(),
                "AU02_r": f_original['vision'][:, -34].tolist(),
                "AU04_r": f_original['vision'][:, -33].tolist(),
                "AU05_r": f_original['vision'][:, -32].tolist(),
                "AU06_r": f_original['vision'][:, -31].tolist(),
                "AU07_r": f_original['vision'][:, -30].tolist(),
                "AU09_r": f_original['vision'][:, -29].tolist(),
                "AU10_r": f_original['vision'][:, -28].tolist(),
                "AU12_r": f_original['vision'][:, -27].tolist(),
                "AU14_r": f_original['vision'][:, -26].tolist(),
                "AU15_r": f_original['vision'][:, -25].tolist(),
                "AU17_r": f_original['vision'][:, -24].tolist(),
                "AU20_r": f_original['vision'][:, -23].tolist(),
                "AU23_r": f_original['vision'][:, -22].tolist(),
                "AU25_r": f_original['vision'][:, -21].tolist(),
                "AU26_r": f_original['vision'][:, -20].tolist(),
                "AU45_r": f_original['vision'][:, -19].tolist(),
            },
            "modified": {
                "loudness": f_defended['audio'][:, 0].tolist(),
                "alphaRatio": f_defended['audio'][:, 1].tolist(),
                "mfcc1": f_defended['audio'][:, 6].tolist(),
                "mfcc2": f_defended['audio'][:, 7].tolist(),
                "mfcc3": f_defended['audio'][:, 8].tolist(),
                "mfcc4": f_defended['audio'][:, 9].tolist(),
                "pitch": f_defended['audio'][:, 10].tolist(),
                "HNR": f_defended['audio'][:, 13].tolist(),
                "F1frequency": f_defended['audio'][:, 16].tolist(),
                "F1bandwidth": f_defended['audio'][:, 17].tolist(),
                "F1amplitude": f_defended['audio'][:, 18].tolist(),
                "AU01_r": f_defended['vision'][:, -35].tolist(),
                "AU02_r": f_defended['vision'][:, -34].tolist(),
                "AU04_r": f_defended['vision'][:, -33].tolist(),
                "AU05_r": f_defended['vision'][:, -32].tolist(),
                "AU06_r": f_defended['vision'][:, -31].tolist(),
                "AU07_r": f_defended['vision'][:, -30].tolist(),
                "AU09_r": f_defended['vision'][:, -29].tolist(),
                "AU10_r": f_defended['vision'][:, -28].tolist(),
                "AU12_r": f_defended['vision'][:, -27].tolist(),
                "AU14_r": f_defended['vision'][:, -26].tolist(),
                "AU15_r": f_defended['vision'][:, -25].tolist(),
                "AU17_r": f_defended['vision'][:, -24].tolist(),
                "AU20_r": f_defended['vision'][:, -23].tolist(),
                "AU23_r": f_defended['vision'][:, -22].tolist(),
                "AU25_r": f_defended['vision'][:, -21].tolist(),
                "AU26_r": f_defended['vision'][:, -20].tolist(),
                "AU45_r": f_defended['vision'][:, -19].tolist(),
            }
        }
        # run MSA models
        logger.info("Running MSA models...")
        res = {
            "original": {},
            "modified": {}
        }
        for m in models:
            config = get_config_regression(m, "mosei")
            r = MMSA_test(config, weights_root / f"{m}-mosei.pth", feat_original, gpu_id=3)
            res["original"][m] = float(r)
            r = MMSA_test(config, weights_root / f"{m}-mosei.pth", feat_defended, gpu_id=3)
            res["modified"][m] = float(r)

    except Exception as e:
        logger.exception(e)
        return {"code": ERROR_CODE, "msg": str(e)}
    return {"code": SUCCESS_CODE, "msg": "success", "result": res, "feature": feat}
