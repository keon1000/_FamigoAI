# -*- coding: utf-8 -*-
"""
Face Kiosk (Streamlit)
- 얼굴 감지/임베딩으로 사용자 식별
- Silero VAD로 자동 녹음 시작/종료
- Whisper로 ASR
- (선택) Kokoro TTS
- RAG 응답 + 요약/저장

주의:
1) 이 파일은 반드시 UTF-8로 저장하세요.
2) 외부 라이브러리(whisper, facenet-pytorch, mediapipe, streamlit, sounddevice, soundfile, playsound3 등)가 설치되어 있어야 합니다.
3) 마이크/카메라 접근 권한이 필요합니다.
"""

import os
import time
import ssl
import threading
from enum import Enum
from collections import deque

import cv2
import numpy as np
import sounddevice as sd
import soundfile as sf
import torch
import torchvision.transforms as transforms
from PIL import Image
import mediapipe as mp
import streamlit as st
from facenet_pytorch import InceptionResnetV1
from playsound3 import playsound
import gc

# 프로젝트 서비스 (사용자 DB 조회, 요약 저장, RAG 호출)
from services.rag.conversation_summarizer import summarize_and_store
from services.rag.rag_connecter import get_rag_response
from sqlalchemy.orm import Session
from db.session import SessionLocal
from db.models import User


# =========================
# 간단한 DB 헬퍼
# =========================
def get_user_id_by_name(name: str) -> int | None:
    """입력된 name에 해당하는 user_id 반환 (없으면 None)."""
    if not name:
        return None
    with SessionLocal() as session:
        user = session.query(User).filter(User.name == name).first()
        return user.user_id if user else None


# OpenCV 내부 스레드 제한(환경에 따라 과도한 스레딩 방지)
cv2.setNumThreads(1)

# =========================
# 경로/디렉터리
# =========================
# 일부 환경에서 torch.hub SSL 검증 문제 회피(내부망/사설 인증서 등)
ssl._create_default_https_context = ssl._create_unverified_context

TEMP_AUDIO_DIR = "audio"
os.makedirs(TEMP_AUDIO_DIR, exist_ok=True)

# =========================
# 기능 토글/기본값
# =========================
ENABLE_TTS = True  # Kokoro TTS 사용 여부
DEFAULT_TTS_LANG = "a"  # Kokoro lang_code
DEFAULT_TTS_VOICE = "af_heart"  # Kokoro voice (예: 'af_heart')
DEFAULT_TTS_SR = 24000  # Kokoro 샘플레이트


# =========================
# 캐시된 모델 로더 (Streamlit 캐시)
# =========================
@st.cache_resource
def get_whisper_model(model_name="base.en", device=None):
    """Whisper 모델 로드(캐시)."""
    import whisper
    return whisper.load_model(model_name) if device is None else whisper.load_model(model_name, device=device)


@st.cache_resource
def get_facenet_model():
    """FaceNet 임베딩 모델 로드(캐시)."""
    return InceptionResnetV1(pretrained='vggface2').eval()


@st.cache_resource
def get_silero_vad_bundle():
    """Silero VAD 모델과 유틸 로드(캐시)."""
    model, utils = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        trust_repo=True
    )
    return model, utils


@st.cache_resource
def get_face_detector():
    """Mediapipe 얼굴 검출기(캐시)."""
    return mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)


@st.cache_resource
def get_kokoro_pipeline():
    """Kokoro TTS 파이프라인(캐시). 실패 시 None 반환."""
    if not ENABLE_TTS:
        return None
    try:
        from kokoro import KPipeline
        pipeline = KPipeline(lang_code=DEFAULT_TTS_LANG)
        return pipeline
    except Exception as e:
        print(f"[TTS] Kokoro pipeline init failed: {e}")
        return None


# =========================
# VAD Recorder (공유 모델 사용)
# =========================
class VADRecorder:
    """Silero VAD로 사용자의 발화를 자동으로 감지하여 wav 파일로 저장."""

    def __init__(self, model=None, utils=None):
        if model is None or utils is None:
            model, utils = get_silero_vad_bundle()
        self.model = model
        self.utils = utils

        # utils는 버전에 따라 dict 또는 tuple일 수 있음 -> 양쪽을 모두 지원
        try:
            VADIterator = utils['VADIterator']  # 신버전(dict)
        except Exception:
            VADIterator = utils[3]              # 구버전(tuple)
        self.vad_iterator = VADIterator(self.model)

        # 파라미터
        self.SAMPLE_RATE = 16000
        self.BUFFER_SIZE = self.SAMPLE_RATE * 60  # 1분 버퍼
        self.THRESHOLD = 0.6                      # 발화 판단 임계값(EMA된 확률)
        self.MIN_DURATION = 0.5                   # 최소 발화 길이(초)
        self.MARGIN = 1                           # 양 옆 여유 구간(초)
        self.SILENCE_TIME = 1                     # 발화 종료 판단용 무음 길이(초)

        self.reset_state()

    def reset_state(self):
        """상태 초기화."""
        self.audio_buffer = deque(maxlen=self.BUFFER_SIZE)
        self.is_speaking = False
        self.speech_start_sample = None
        self.sample_counter = 0
        self.silence_counter = 0
        self.ema_speech_prob = 0.0
        self.saved_filename = None

    def _save_audio_segment(self, start_sample, end_sample):
        """버퍼에서 지정 구간을 잘라 wav로 저장."""
        audio_array = np.array(list(self.audio_buffer), dtype=np.int16)
        start = max(0, start_sample - int(self.MARGIN * self.SAMPLE_RATE))
        end = min(len(audio_array), end_sample + int(self.MARGIN * self.SAMPLE_RATE))
        segment = audio_array[start:end]
        if len(segment) / self.SAMPLE_RATE < self.MIN_DURATION:
            print("[VAD] Segment too short, skip.")
            return
        filename = f"{TEMP_AUDIO_DIR}/speech_{time.strftime('%Y%m%d_%H%M%S')}.wav"
        sf.write(filename, segment, self.SAMPLE_RATE)
        print(f"[VAD] Audio saved: {filename}")
        self.saved_filename = filename

    def _callback(self, indata, frames, time_info, status):
        """SoundDevice 입력 콜백. 발화 감지/종료 처리."""
        if status:
            print("[VAD] status:", status)
        if self.saved_filename:
            return  # 이미 저장 완료되면 더 처리하지 않음

        # float(-1~1) -> int16
        audio_int16 = (indata * 32768).astype(np.int16).flatten()
        self.audio_buffer.extend(audio_int16)

        if len(audio_int16) < 512:
            return

        # 발화 확률 추정(EMA)
        audio_tensor = torch.from_numpy(audio_int16).float()
        speech_prob = self.vad_iterator.model(audio_tensor, self.SAMPLE_RATE).item()
        self.ema_speech_prob = 0.9 * self.ema_speech_prob + 0.1 * speech_prob

        if self.ema_speech_prob > self.THRESHOLD:
            # 발화 시작/연장
            if not self.is_speaking:
                self.is_speaking = True
                self.speech_start_sample = self.sample_counter
            self.silence_counter = 0
        else:
            # 발화 종료 후보
            if self.is_speaking:
                self.silence_counter += frames / self.SAMPLE_RATE
                if self.silence_counter >= self.SILENCE_TIME:
                    self.is_speaking = False
                    speech_end_sample = self.sample_counter
                    duration = (speech_end_sample - self.speech_start_sample) / self.SAMPLE_RATE
                    if duration >= self.MIN_DURATION:
                        self._save_audio_segment(self.speech_start_sample, speech_end_sample)

        self.sample_counter += frames

    def record(self, timeout=10):
        """timeout초 동안 마이크를 모니터링하여 발화 구간을 자동 저장."""
        self.reset_state()
        stream = sd.InputStream(
            callback=self._callback,
            channels=1,
            samplerate=self.SAMPLE_RATE,
            blocksize=512
        )
        with stream:
            print("[VAD] Listening for speech...")
            start_time = time.time()
            while time.time() - start_time < timeout:
                if self.saved_filename:
                    break
                sd.sleep(100)
        print("[VAD] Finished listening.")
        return self.saved_filename


def listen_and_record_speech(timeout=10, model=None, utils=None):
    """메인 스레드에서 준비한 VAD 모델/유틸을 사용하여 발화 녹음."""
    if model is None or utils is None:
        raise RuntimeError("VAD model/utils must be provided from main thread")
    recorder = VADRecorder(model=model, utils=utils)
    return recorder.record(timeout=timeout)


# =========================
# 상태 정의
# =========================
class State(Enum):
    IDLE = 0       # 사용자 대기
    USER_CHECK = 1 # 얼굴로 사용자 확인
    ENROLL = 2     # 신규 사용자 등록
    WELCOME = 3    # 인사 + VAD 준비
    ASR = 4        # 녹음 파일을 ASR 처리
    BYE = 5        # 종료 + 요약 저장


# =========================
# 전역 상태/플래그
# =========================
FACE_DETECTED = False
USER_EXIST = False
ENROLL_SUCCESS = False
VAD = False
BYE_EXIST = False
TIMER_EXPIRED = False  # WELCOME/BYE 타이머

# 공유 데이터
sh_face_crop = None
sh_bbox = None
sh_embedding = None
sh_current_user = None
sh_audio_file = None
sh_tts_file = None
sh_message = "Initializing..."
sh_color = (255, 255, 0)  # BGR
sh_timer_end = 0

SESSION_USER = None
USER_SWITCHED = False

# 세션 한정 그룹명 (WELCOME~BYE 사이 유지)
sh_session_group = None

# 비동기 작업 플래그
VAD_TASK_STARTED = False
VAD_TASK_RUNNING = False
ASR_TASK_STARTED = False
ASR_TASK_RUNNING = False
ASR_TEXT = None

# 대화 로그(간단한 리스트)
sh_transcript = []

# DB/스레시홀드
DB_PATH = "faces_db.npy"
SIM_THRESHOLD = 0.5

# BBOX 평균화(스무딩)
BBOX_AVG_N = 5
_bbox_history = deque(maxlen=BBOX_AVG_N)


# =========================
# 얼굴 DB 유틸(name_list + embeddings)
# =========================
def load_db():
    """간단한 로컬 npy 기반 DB 로드."""
    if os.path.exists(DB_PATH):
        data = np.load(DB_PATH, allow_pickle=True).item()
        name = data["name_list"]
        embs = data["embeddings"]
        return name, embs
    else:
        return [], np.empty((0, 512))


def save_db(name_list, embeddings):
    """간단한 로컬 npy 기반 DB 저장."""
    np.save(DB_PATH, {"name_list": name_list, "embeddings": embeddings})


def find_match(embedding, name_list, embeddings):
    """코사인 유사도로 가장 가까운 사용자 찾기."""
    if len(embeddings) == 0:
        return None, 0.0
    sims = [np.dot(embedding, emb) / (np.linalg.norm(embedding) * np.linalg.norm(emb)) for emb in embeddings]
    max_idx = int(np.argmax(sims))
    if sims[max_idx] >= SIM_THRESHOLD:
        return name_list[max_idx], float(sims[max_idx])
    else:
        return None, float(sims[max_idx])


def detect_user_change():
    """
    세션 도중 사용자 교체 감지:
    현재 SESSION_USER와 프레임의 얼굴 임베딩 매칭 결과가 다르면 USER_SWITCHED=True.
    """
    global USER_SWITCHED, sh_message, sh_color

    if SESSION_USER is None:
        return  # 아직 세션 시작 전

    if sh_face_crop is None or sh_face_crop.size == 0:
        return  # 얼굴 미검출 상태는 교체로 보지 않음

    face_pil = Image.fromarray(cv2.cvtColor(sh_face_crop, cv2.COLOR_BGR2RGB))
    face_tensor = preprocess(face_pil).unsqueeze(0)
    with torch.no_grad():
        emb = resnet(face_tensor)[0].cpu().numpy()
    emb = emb / np.linalg.norm(emb)

    match_name, sim = find_match(emb, name_list, embeddings)

    # 다른 사용자로 인식되면 교체로 판단
    if match_name and match_name != SESSION_USER:
        USER_SWITCHED = True
        sh_message = f"사용자가 {match_name}님으로 바뀌었습니다. 세션을 종료합니다..."
        sh_color = (255, 0, 0)


def _clip_bbox(x, y, w, h, iw, ih):
    """bbox를 프레임 경계 안으로 클리핑."""
    x = max(0, min(x, iw - 1))
    y = max(0, min(y, ih - 1))
    w = max(1, min(w, iw - x))
    h = max(1, min(h, ih - y))
    return x, y, w, h


def update_face_detection():
    """얼굴 탐지 + 최근 N프레임 평균 bbox 반영."""
    global FACE_DETECTED, sh_face_crop, sh_bbox, sh_frame, _bbox_history, BBOX_AVG_N

    # deque maxlen 동기화
    if _bbox_history.maxlen != BBOX_AVG_N:
        _bbox_history = deque(list(_bbox_history), maxlen=BBOX_AVG_N)

    image = sh_frame
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_image)

    ih, iw, _ = image.shape
    det_count = len(results.detections) if (results and results.detections) else 0

    if det_count == 1:
        FACE_DETECTED = True
        det = results.detections[0]
        bboxC = det.location_data.relative_bounding_box
        x = int(bboxC.xmin * iw)
        y = int(bboxC.ymin * ih)
        w = int(bboxC.width * iw)
        h = int(bboxC.height * ih)
        x, y, w, h = _clip_bbox(x, y, w, h, iw, ih)

        _bbox_history.append((x, y, w, h))
        xs = np.mean([b[0] for b in _bbox_history])
        ys = np.mean([b[1] for b in _bbox_history])
        ws = np.mean([b[2] for b in _bbox_history])
        hs = np.mean([b[3] for b in _bbox_history])
        xa, ya, wa, ha = _clip_bbox(int(xs), int(ys), int(ws), int(hs), iw, ih)

        sh_bbox = (xa, ya, wa, ha)
        # ROI는 copy로 뽑아 사용(원본 프레임 덮어쓰기 방지)
        sh_face_crop = image[ya:ya + ha, xa:xa + wa].copy()
        if sh_face_crop.size == 0:
            FACE_DETECTED = False
            _bbox_history.clear()
            sh_bbox = None
            sh_face_crop = None
    else:
        FACE_DETECTED = False
        _bbox_history.clear()
        sh_bbox = None
        sh_face_crop = None

    # 임시 객체 해제
    del rgb_image, results
    return det_count


# =========================
# Whisper ASR
# =========================
def asr_from_wav(file_path: str) -> str:
    """Whisper로 wav 파일을 텍스트로 변환."""
    result = whisper_model.transcribe(file_path, language='en', fp16=False)
    return result['text']


# =========================
# TTS (Kokoro)
# =========================
def build_tts_reply_text(asr_text: str, user: str | None) -> str:
    """
    간단한 TTS 응답 텍스트 구성.
    '안녕' 또는 'bye' 포함 시 작별 인사, 아니면 ASR/응답 내용을 읽어줌.
    """
    t = "".join((asr_text or "").split()).lower()
    if ("안녕" in t) or ("bye" in t):
        return f"Good bye, {user if user else ''}."
    return f"{(user + ' ') if user else ''}said that {asr_text}"


def synthesize_tts_kokoro(text: str) -> str | None:
    """Kokoro로 합성하여 wav 저장 및 재생. 실패 시 None."""
    if kokoro_pipeline is None or not ENABLE_TTS:
        return None
    try:
        chunks = []
        for _, _, audio in kokoro_pipeline(text, voice=DEFAULT_TTS_VOICE):
            chunks.append(audio)
        if not chunks:
            return None
        audio = chunks[0]
        out_path = f"{TEMP_AUDIO_DIR}/tts_{int(time.time())}.wav"
        sf.write(out_path, audio, DEFAULT_TTS_SR)
        playsound(out_path)
        return out_path
    except Exception as e:
        print(f"[TTS] synthesis failed: {e}")
        return None


# =========================
# 상태 진입 함수(각 화면 로직)
# =========================
def enter_idle():
    """IDLE: 얼굴이 없으면 대기 메시지."""
    global sh_message, sh_color
    det_count = update_face_detection()
    if not FACE_DETECTED:
        if det_count > 1:
            sh_message = f"{det_count}명 얼굴 감지됨. 한 명만 카메라에 들어와 주세요."
            sh_color = (0, 0, 255)
        else:
            sh_message = "사용자를 기다리는 중..."
            sh_color = (255, 255, 0)


def enter_user_check():
    """USER_CHECK: 얼굴 임베딩 추출 후 DB에서 매칭."""
    global USER_EXIST, sh_embedding, sh_current_user, sh_message, sh_color
    update_face_detection()
    if sh_face_crop is None:
        return
    face_pil = Image.fromarray(cv2.cvtColor(sh_face_crop, cv2.COLOR_BGR2RGB))
    face_tensor = preprocess(face_pil).unsqueeze(0)
    with torch.no_grad():
        embedding = resnet(face_tensor)[0].cpu().numpy()
    sh_embedding = embedding / np.linalg.norm(embedding)

    match_name, sim = find_match(sh_embedding, name_list, embeddings)
    if match_name:
        USER_EXIST = True
        sh_current_user = match_name
        sh_message = f"식별 중... {match_name} ({sim:.2f})"
        sh_color = (0, 255, 0)
    else:
        USER_EXIST = False
        sh_message = "등록되지 않은 사용자입니다. 오른쪽 패널에서 등록해주세요."
        sh_color = (0, 255, 255)
    gc.collect()


def enter_enroll(key=None):
    """ENROLL: 프레임에서 얼굴이 1개일 때만 등록 유도."""
    global sh_message, sh_color
    det_count = update_face_detection()
    if not FACE_DETECTED:
        if det_count > 1:
            sh_message = f"{det_count}명 얼굴 감지됨. 한 명만 카메라에 들어와 주세요."
            sh_color = (0, 0, 255)
        else:
            sh_message = "등록을 위해 얼굴을 카메라 정면에 맞춰주세요."
            sh_color = (255, 255, 0)
    else:
        sh_message = "미등록 사용자입니다. 오른쪽 패널에서 이름을 입력해 등록하세요."
        sh_color = (0, 255, 255)


def enter_welcome():
    """WELCOME: 인사 + VAD 비동기 시작(타이머 경과 후)."""
    global VAD, sh_audio_file, TIMER_EXPIRED, sh_message, sh_color
    update_face_detection()
    detect_user_change()

    sh_message = f"Hi, {sh_current_user}!"
    sh_color = (0, 255, 0)

    TIMER_EXPIRED = (time.time() > sh_timer_end)
    if TIMER_EXPIRED and not VAD_TASK_STARTED:
        start_vad_async(timeout=5)


def enter_asr():
    """ASR: 저장된 오디오가 있으면 Whisper 비동기 시작."""
    update_face_detection()
    detect_user_change()

    if sh_audio_file and not ASR_TASK_STARTED:
        start_asr_async(sh_audio_file)


def enter_bye():
    """BYE: 인사 후 일정 시간 경과 시 요약/저장 트리거."""
    global TIMER_EXPIRED, sh_message, sh_color
    update_face_detection()
    sh_message = f"Bye, {sh_current_user}!"
    sh_color = (255, 0, 255)
    TIMER_EXPIRED = (time.time() > sh_timer_end)


# =========================
# 비동기 작업(스레드)
# =========================
def start_vad_async(timeout=5):
    """VAD 녹음을 비동기로 시작."""
    global VAD_TASK_STARTED, VAD_TASK_RUNNING
    if VAD_TASK_RUNNING:
        return
    VAD_TASK_STARTED = True
    VAD_TASK_RUNNING = True

    def _worker():
        global sh_audio_file, VAD, VAD_TASK_RUNNING
        try:
            filename = listen_and_record_speech(timeout=timeout, model=vad_model, utils=vad_utils)
            if filename:
                sh_audio_file = filename
                VAD = True
            else:
                VAD = False
        finally:
            VAD_TASK_RUNNING = False

    threading.Thread(target=_worker, daemon=True).start()


def start_asr_async(file_path: str):
    """Whisper로 ASR + (선택) Kokoro로 음성 합성 + RAG 연동."""
    global ASR_TASK_STARTED, ASR_TASK_RUNNING
    if ASR_TASK_RUNNING:
        return
    ASR_TASK_STARTED = True
    ASR_TASK_RUNNING = True

    def _worker():
        global ASR_TEXT, BYE_EXIST, ASR_TASK_RUNNING, sh_tts_file
        try:
            text = asr_from_wav(file_path)
            ASR_TEXT = text or ""
            print("[ASR] ", ASR_TEXT)
            t = "".join(ASR_TEXT.split()).lower()
            BYE_EXIST = ("안녕" in t) or ("bye" in t)
            if BYE_EXIST:
                print("[ASR] BYE detected in ASR result.")
                tts_text = build_tts_reply_text(ASR_TEXT, sh_current_user)
                sh_tts_file = synthesize_tts_kokoro(tts_text)
            else:
                # RAG 호출
                answer = get_rag_response(
                    user_id=get_user_id_by_name(SESSION_USER),
                    query=str(ASR_TEXT),
                    target="team",
                    group_name=sh_session_group,
                )

                # 대화 로그 업데이트
                sh_transcript.append({"role": "user", "content": ASR_TEXT})
                sh_transcript.append({"role": "assistant", "content": answer})

                # 응답 낭독(TTS)
                tts_text = build_tts_reply_text(answer, sh_current_user)
                sh_tts_file = synthesize_tts_kokoro(tts_text)
        finally:
            ASR_TASK_RUNNING = False

    threading.Thread(target=_worker, daemon=True).start()


# =========================
# 상태 전이 & 디스패처
# =========================
def state_transition(current_state: State) -> State:
    """현재 상태와 플래그에 따라 다음 상태를 결정."""
    global sh_embedding, name_list, embeddings

    if current_state == State.IDLE:
        return State.USER_CHECK if FACE_DETECTED else State.IDLE

    elif current_state == State.USER_CHECK:
        if USER_EXIST:
            return State.WELCOME if sh_session_group else State.USER_CHECK
        else:
            return State.ENROLL

    elif current_state == State.ENROLL:
        if ENROLL_SUCCESS:
            name_list, embeddings = load_db()
            return State.USER_CHECK
        return State.IDLE if not FACE_DETECTED else State.ENROLL

    elif current_state == State.WELCOME:
        if USER_SWITCHED:
            return State.BYE
        if not (time.time() > sh_timer_end):
            return State.WELCOME
        if VAD:
            return State.ASR
        if VAD_TASK_RUNNING:
            return State.WELCOME
        return State.IDLE

    elif current_state == State.ASR:
        if USER_SWITCHED:
            return State.BYE
        if ASR_TASK_RUNNING:
            return State.ASR
        if ASR_TASK_STARTED and not ASR_TASK_RUNNING:
            return State.BYE if BYE_EXIST else State.IDLE
        return State.ASR

    elif current_state == State.BYE:
        if TIMER_EXPIRED:
            # 종료 시점에 대화 요약/저장 (에러는 콘솔로)
            try:
                res = summarize_and_store(
                    me_id=get_user_id_by_name(SESSION_USER),
                    messages=sh_transcript,
                    visibility="group"
                )
                print(f"(conversation saved) text_ref={res.get('text_ref')} id={res.get('embedding_id')}")
            except Exception as e:
                print(f"[Save Conversation] failed: {e}")
            return State.IDLE
        return State.BYE

    return current_state


def call_state_fn(state: State, key):
    """현재 상태에 해당하는 진입 함수를 호출."""
    if state == State.IDLE:
        enter_idle()
    elif state == State.USER_CHECK:
        enter_user_check()
    elif state == State.ENROLL:
        enter_enroll(key)
    elif state == State.WELCOME:
        enter_welcome()
    elif state == State.ASR:
        enter_asr()
    elif state == State.BYE:
        enter_bye()


# =========================
# 모델 초기화(캐시된 로더 사용)
# =========================
print("Loading models (cached)...")
resnet = get_facenet_model()
face_detection = get_face_detector()
whisper_model = get_whisper_model("base.en")
vad_model, vad_utils = get_silero_vad_bundle()
kokoro_pipeline = get_kokoro_pipeline()

# 얼굴 전처리 파이프라인(FaceNet 입력 규격)
preprocess = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

name_list, embeddings = load_db()
print("Models loaded (using cache).")


# =========================
# Streamlit UI & 메인 루프
# =========================
st.set_page_config(page_title="Face Kiosk", layout="wide")
st.title("👤 Face Kiosk with State UI")

col_video, col_ui = st.columns([3, 2], vertical_alignment="top")

# Camera / Options
with col_video:
    st.subheader("📷 Camera")
    cam_index = st.number_input("Camera index", min_value=0, max_value=10, value=0, step=1)
    width = st.slider("Frame width", 320, 1920, 640, step=10)
    bbox_avg_n_ui = st.slider("BBOX smoothing (frames)", 1, 30, 5, help="최근 N프레임의 얼굴 bbox 평균 적용.")
    run = st.toggle("Run camera", value=True)
    frame_slot = st.empty()

# UI placeholders
with col_ui:
    st.subheader("👥 Group")
    group_ui = st.empty()  # 세션용 그룹 입력 placeholder
    st.subheader("🧭 State Panel")
    state_badge = st.empty()
    message_slot = st.empty()
    usercheck_slot = st.empty()
    enroll_slot = st.empty()
    welcome_slot = st.empty()
    asr_slot = st.empty()
    bye_slot = st.empty()
    audio_slot = st.empty()
    debug_slot = st.expander("Debug", expanded=False)
    with debug_slot:
        debug_ph = st.empty()

# 그룹 입력 상태키 관리
if "group_key_counter" not in st.session_state:
    st.session_state.group_key_counter = 1
if "current_group_key" not in st.session_state:
    st.session_state.current_group_key = f"usercheck_group_input_{st.session_state.group_key_counter}"

# 그룹 입력 UI: 세션 중에는 잠금, BYE 후 재입력
with group_ui.container():
    _gkey = st.session_state.current_group_key
    _gval = (st.session_state.get(_gkey, "") or "").strip()
    st.text_input(
        "그룹명 (BYE 이후에만 다시 입력 가능)",
        key=_gkey,
        placeholder="예: slpr",
        disabled=bool(_gval),
    )
    st.caption(f"현재 그룹: {_gval or '-'}")

# 세션 상태
GROUP_INPUT_COUNTER = 0
CURRENT_GROUP_KEY = None

# Camera 핸들
if "cap" not in st.session_state:
    st.session_state.cap = None


def open_camera(index: int, target_w: int):
    """카메라 오픈 + 해상도 설정."""
    cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_w)
    return cap


# ENROLL UI lifecycle flags & unique keys
ENROLL_UI_BUILT = False
enroll_face_ph = None
enroll_form_counter = 0
current_enroll_form_key = None
current_enroll_name_key = None

# WELCOME 관련
WELCOME_KEY = None
WELCOME_KEY_COUNTER = 0
WELCOME_UI_BUILT = False
welcome_group_ph = None
welcome_status_ph = None
welcome_progress_ph = None

# 초기 상태
state = State.IDLE
st.caption("Starting state machine...")


def ui_enroll_submit(new_name: str):
    """등록 버튼 눌렀을 때 처리(이름만 저장, 얼굴 임베딩은 현재 프레임에서 사용)."""
    global ENROLL_SUCCESS, USER_EXIST, name_list, embeddings, sh_current_user

    if not new_name or new_name.strip() == "":
        st.warning("이름은 필수입니다.")
        return
    if sh_embedding is None or len(sh_embedding) == 0:
        st.error("얼굴 임베딩이 준비되지 않았습니다. 카메라 정면에서 얼굴을 맞춰주세요.")
        return
    if any(n == new_name for n in name_list):
        st.warning("이미 존재하는 이름입니다. 다른 이름을 입력하세요.")
        return

    name_list.append(new_name)
    if embeddings.size:
        embeddings = np.vstack([embeddings, sh_embedding])
    else:
        embeddings = np.array([sh_embedding])

    save_db(name_list, embeddings)

    sh_current_user = new_name
    ENROLL_SUCCESS = True
    USER_EXIST = True

    st.success(f"등록 완료: {new_name}")
    print("[DB Updated] ", name_list, embeddings.shape)


def render_state_panel(current_state: State):
    """오른쪽 상태 패널 렌더링."""
    global ENROLL_UI_BUILT, enroll_face_ph, current_enroll_name_key, current_enroll_form_key
    global sh_session_group

    state_badge.markdown(f"**Current State:** :blue[{current_state.name}]")

    # 상태별 슬롯 초기화
    if current_state != State.USER_CHECK:
        if usercheck_slot is not None:
            usercheck_slot.empty()
    if current_state != State.ENROLL:
        enroll_slot.empty()
    if current_state != State.WELCOME:
        welcome_slot.empty()
    if current_state != State.ASR:
        asr_slot.empty()
    if current_state != State.BYE:
        bye_slot.empty()

    message_slot.markdown(f"**Message:** {sh_message}")

    # USER_CHECK
    if current_state == State.USER_CHECK:
        usercheck_slot.empty()
        with usercheck_slot.container():
            st.info("사용자 확인 단계입니다. 상단 Group 입력란에 그룹명을 먼저 입력하세요(BYE 전까지 유지).")

    # ENROLL
    if current_state == State.ENROLL:
        if current_enroll_form_key is None or current_enroll_name_key is None:
            ts = int(time.time() * 1000)
            current_enroll_form_key = f"form_enroll_{ts}"
            current_enroll_name_key = f"enroll_name_{ts}"

        if not ENROLL_UI_BUILT:
            ENROLL_UI_BUILT = True
            with enroll_slot.container():
                st.info("미등록 사용자입니다. 아래 폼으로 등록하세요.")
                globals()['enroll_face_ph'] = st.empty()
                with st.form(key=current_enroll_form_key, clear_on_submit=False):
                    new_name = st.text_input("이름", key=current_enroll_name_key)
                    submitted = st.form_submit_button("등록하기", use_container_width=True)
                if submitted:
                    ui_enroll_submit(new_name)

        if enroll_face_ph is not None:
            if sh_face_crop is not None and sh_face_crop.size != 0:
                face_rgb = cv2.cvtColor(sh_face_crop, cv2.COLOR_BGR2RGB)
                enroll_face_ph.image(face_rgb, caption="등록할 얼굴", use_container_width=True)
            else:
                enroll_face_ph.warning("얼굴이 보이지 않습니다. 카메라 정면에 한 명만 들어오세요.")
    else:
        if ENROLL_UI_BUILT:
            ENROLL_UI_BUILT = False
            globals()['enroll_face_ph'] = None

    # WELCOME
    if current_state == State.WELCOME:
        welcome_slot.empty()
        with welcome_slot.container():
            if sh_session_group:
                st.caption(f"(세션 그룹: {sh_session_group})")
            if not (time.time() > sh_timer_end):
                remain = max(0.0, sh_timer_end - time.time())
                st.success(f"Hi, **{sh_current_user}**! 곧 녹음을 시작합니다.")
                st.progress(min(max(1.0 - (remain / 2.0), 0.0), 1.0), text="Greeting...")
            else:
                if VAD_TASK_RUNNING:
                    st.info("마이크 대기 중...")
                elif VAD:
                    st.success("녹음이 완료되어 ASR로 이동합니다.")
                else:
                    st.warning("녹음이 시작되지 않았습니다. 잠시 후 다시 시도합니다.")

    # ASR
    if current_state == State.ASR:
        asr_slot.empty()
        with asr_slot.container():
            if ASR_TASK_RUNNING:
                st.info("Whisper로 변환 중...")
            elif ASR_TEXT is not None:
                st.write("**ASR 결과:** ", ASR_TEXT)
                st.write(f"**BYE 검출:** {'예' if BYE_EXIST else '아니오'}")
                if sh_tts_file:
                    audio_slot.audio(sh_tts_file)
            else:
                st.write("대기 중...")

    # BYE
    if current_state == State.BYE:
        bye_slot.empty()
        with bye_slot.container():
            st.warning(f"Bye, **{sh_current_user}**!")
            if sh_session_group:
                st.caption(f"(세션 그룹: {sh_session_group})")
            remain = max(0.0, sh_timer_end - time.time())
            pct = min(max(1.0 - (remain / 2.0), 0.0), 1.0)
            st.progress(pct, text="Ending...")


# ========= run =========
if run:
    # 카메라 열기
    if st.session_state.cap is None or not st.session_state.cap.isOpened():
        st.session_state.cap = open_camera(int(cam_index), int(width))
        if not st.session_state.cap.isOpened():
            st.error("카메라를 열 수 없습니다. 장치를 확인하거나 다른 인덱스를 시도하세요.")
            st.stop()

    # 초기 상태
    state = State.IDLE
    frame_i = 0

    while run:
        t0 = time.time()
        frame_i += 1

        # bbox 스무딩 길이 동기화
        BBOX_AVG_N = int(bbox_avg_n_ui)

        success, sh_frame = st.session_state.cap.read()
        if not success:
            st.error("프레임을 읽지 못했습니다.")
            break

        key = None  # 키 입력 없음(streamlit 환경)

        # 현재 그룹명은 텍스트박스의 최신 값
        _gkey = st.session_state.current_group_key
        sh_session_group = (st.session_state.get(_gkey, "") or "").strip() or None

        # 상태 호출 + 전이
        call_state_fn(state, key)
        new_state = state_transition(state)

        if new_state != state:
            print(f"State Change: {state.name} -> {new_state.name}")

            # ENROLL 진입 시 UI 상태 초기화
            if new_state == State.ENROLL and state != State.ENROLL:
                ENROLL_SUCCESS = False
                USER_EXIST = False
                ENROLL_UI_BUILT = False
                enroll_form_counter += 1
                current_enroll_form_key = f"form_enroll_{enroll_form_counter}"
                current_enroll_name_key = f"enroll_name_{enroll_form_counter}"

            # WELCOME 진입 시 타이머/VAD 상태 초기화 + 세션 사용자 고정
            if new_state == State.WELCOME:
                sh_timer_end = time.time() + 2.0
                VAD = False
                VAD_TASK_STARTED = False
                VAD_TASK_RUNNING = False
                sh_audio_file = None
                sh_tts_file = None

                SESSION_USER = sh_current_user
                USER_SWITCHED = False

            # ASR 진입 시 상태 초기화
            if new_state == State.ASR:
                ASR_TEXT = None
                BYE_EXIST = False
                ASR_TASK_STARTED = False
                ASR_TASK_RUNNING = False

            # BYE 진입 시 타이머
            if new_state == State.BYE:
                sh_timer_end = time.time() + 2.0

            # BYE -> IDLE 로 돌아갈 때: 그룹/세션 사용자 해제 + 그룹 입력 재활성화
            if state == State.BYE and new_state == State.IDLE:
                sh_session_group = None
                SESSION_USER = None
                USER_SWITCHED = False

                # 다음 세션을 위해 새로운 그룹 입력 키 발급
                new_key = f"usercheck_group_input_{st.session_state.group_key_counter + 1}"

                # 이전 그룹 입력 키들 제거(새 키만 유지)
                for k in list(st.session_state.keys()):
                    if k.startswith("usercheck_group_input_") and k != new_key:
                        st.session_state.pop(k, None)

                st.session_state.group_key_counter += 1
                st.session_state.current_group_key = new_key

                # 그룹 입력칸 재생성을 위해 rerun
                st.rerun()

            state = new_state

        # 상태 패널 렌더링
        render_state_panel(state)

        # 오버레이(메시지/박스) 그리기
        display_frame = sh_frame
        if sh_bbox:
            x, y, w, h = sh_bbox
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), sh_color, 2)
            cv2.putText(display_frame, sh_message, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, sh_color, 2)
        else:
            cv2.putText(display_frame, sh_message, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, sh_color, 2)

        # 스트림릿에 출력(리사이즈)
        frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        h0, w0, _ = display_frame.shape
        new_h = int(h0 * (width / w0))
        resized = cv2.resize(display_frame, (int(width), new_h))
        frame_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        frame_slot.image(
            frame_rgb,
            channels="RGB",
            caption="Live",
            use_container_width=True,
            output_format="JPEG",
        )
        del resized, frame_rgb, display_frame

        # 디버그 정보
        debug_ph.json({
            "FACE_DETECTED": FACE_DETECTED,
            "USER_EXIST": USER_EXIST,
            "ENROLL_SUCCESS": ENROLL_SUCCESS,
            "VAD": VAD,
            "VAD_TASK_RUNNING": VAD_TASK_RUNNING,
            "ASR_TASK_RUNNING": ASR_TASK_RUNNING,
            "BYE_EXIST": BYE_EXIST,
            "TIMER_EXPIRED": TIMER_EXPIRED,
            "current_user": sh_current_user,
            "session_group": sh_session_group,  # 세션 한정 그룹명(저장은 안 함)
            "audio_file": sh_audio_file,
            "tts_file": sh_tts_file,
            "bbox_avg_n": BBOX_AVG_N,
            "len(_bbox_history)": len(_bbox_history),
            "id(whisper_model)": id(whisper_model),
            "id(resnet)": id(resnet),
            "id(face_detection)": id(face_detection),
        })

        # 주기적 GC + CUDA 캐시 정리
        if frame_i % 30 == 0:
            gc.collect()
            frame_i = 0
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # FPS 제한(30fps 기준)
        elapsed = time.time() - t0
        sleep_for = max(0.0, (1 / 30) - elapsed)
        time.sleep(sleep_for)

    # cleanup
    if st.session_state.cap is not None:
        st.session_state.cap.release()
        st.session_state.cap = None
    cv2.destroyAllWindows()
    frame_slot.empty()
    st.info("카메라를 종료했습니다.")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
