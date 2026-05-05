import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json
import os

# 1. 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. 클래스 로드
@st.cache_data
def load_classes():
    if not os.path.exists('class_names.json'):
        st.error("class_names.json 파일이 없습니다! 학습 코드에서 먼저 저장해주세요.")
        return []
    with open('class_names.json', 'r') as f:
        return json.load(f)

class_names = load_classes()
num_classes = len(class_names) if class_names else 150

# 3. 모델 로드
@st.cache_resource
def load_model(model_name):
    try:
        # 모델 뼈대 준비
        if "resnet18" in model_name:
            model = models.resnet18(weights=None)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            
        elif "googlenet" in model_name:
            model = models.googlenet(weights=None, aux_logits=False)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            
        elif "mobilenet_v2" in model_name:
            model = models.mobilenet_v2(weights=None)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        
        # 파일명 조립
        pth_path = f"It's_Pikachu_{model_name}.pth"
        
        model.load_state_dict(torch.load(pth_path, map_location=device), strict=False)
        model = model.to(device)
        model.eval()
        return model
        
    except Exception as e:
        st.error(f"모델을 불러오는 중 에러가 발생했습니다: {e}")
        return None

# --- GUI 레이아웃 ---
st.title("🐾 포켓몬 분류기 (4-Model 비교)")
st.write("학습된 4가지 모델 중 하나를 선택해 성능을 비교해보세요!")

model_options = ["resnet18_finetune", "resnet18_extract", "googlenet", "mobilenet_v2"]
selected_model_name = st.selectbox("🤖 사용할 모델을 선택하세요:", model_options, index=0)

model = load_model(selected_model_name)

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

uploaded_file = st.file_uploader("포켓몬 이미지 파일 선택", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='업로드된 이미지', use_container_width=True)
    
    st.info(f"**{selected_model_name}** 모델로 분석 중입니다...")
    
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(input_batch)
        probabilities = torch.softmax(output, dim=1)
        top_probs, top_indices = torch.topk(probabilities, 5, dim=1)
        
    st.subheader("Top-5 Predictions:")
    for rank, (prob, index) in enumerate(zip(top_probs[0].cpu().tolist(), top_indices[0].cpu().tolist())):
        pokemon_name = class_names[index] if class_names else f"Class {index}"
        st.write(f"{rank + 1}. **{pokemon_name}** ({prob * 100:.2f}%)")