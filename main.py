import os
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf


# FastAPI 앱 생성
app = FastAPI()
# CORS 미들웨어 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 출처 허용. 특정 출처만 허용하려면 ["http://localhost", "http://example.com"] 형식으로 지정
    allow_credentials=True,
    allow_methods=["*"],  # 모든 HTTP 메서드 허용 (GET, POST, OPTIONS 등)
    allow_headers=["*"],  # 모든 헤더 허용
)
# 요청 바디 모델 정의
class Message(BaseModel):
    msg: str

# 모델 디렉토리 경로
model_dir = "intent_classifier_model"

# 모델 및 토크나이저 로드
model = TFBertForSequenceClassification.from_pretrained(model_dir)
tokenizer = BertTokenizer.from_pretrained(model_dir)

# 레이블 클래스 로드 (allow_pickle=True)
label_classes = np.load('label_classes.npy', allow_pickle=True)

# 의도별 응답 정의
responses = {
    "날씨정보": "현재 날씨 정보를 알려드리겠습니다.",
    "교통정보": "현재 교통 상황을 안내해 드립니다.",
    "메뉴추천": "[메뉴추천] 서비스로 이동합니다.",
    "주변추천": "주변 추천 장소를 안내해 드립니다.",
    "매장정보": "매장 정보를 확인해 드리겠습니다.",
    "화상통화 연결": "화상통화를 연결합니다.",
    "상담원 연결": "상담원을 연결해 드리겠습니다.",
    "하이패스": "하이패스 정보를 안내해 드립니다.",
    "지역 특산품": "지역 특산품 정보를 알려드리겠습니다.",
    "이벤트": "현재 진행 중인 이벤트 정보를 안내해 드립니다.",
    "정비": "정비 관련 정보를 제공해 드리겠습니다.",
    "전기차 충전": "전기차 충전소 정보를 안내해 드립니다.",
    "교통상황": "현재 교통 상황을 알려드리겠습니다.",
    "긴급상황": "긴급 상황 대처 방법을 안내해 드립니다.",
    "비상상황": "비상 상황 정보를 안내해 드립니다.",
    "숙박": "숙박 가능한 장소를 안내해 드리겠습니다.",
    "메뉴추가": "메뉴 추가 방법을 안내해 드립니다.",
    "메뉴삭제": "메뉴 삭제 방법을 안내해 드립니다.",
    "메뉴수정": "메뉴 수정 방법을 안내해 드립니다.",
    "결제방법안내": "결제 방법을 안내해 드리겠습니다.",
    "주유소": "주유소 정보를 안내해 드립니다.",
    "결제오류": "결제 오류 해결 방법을 안내해 드립니다.",
    "결제방법선택": "결제 방법을 선택해 주세요."
}

# 입력된 텍스트의 의도 예측
def predict_intent(text):
    # 텍스트 전처리
    encoding = tokenizer.encode_plus(
        text,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='tf'
    )

    # 모델 예측
    predictions = model(encoding['input_ids'], attention_mask=encoding['attention_mask'])[0]
    predicted_class_id = tf.argmax(predictions, axis=1).numpy()[0]

    # 예측된 클래스 ID에 따라 적절한 의도를 반환
    return label_classes[predicted_class_id]

# API 라우트 정의
@app.post("/chat")
async def chat_endpoint(message: Message):
    user_input = message.msg
    intent = predict_intent(user_input)
    response = responses.get(intent, "죄송합니다. 이해하지 못했습니다.")
    return {"response": response}

# 서버 실행
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
