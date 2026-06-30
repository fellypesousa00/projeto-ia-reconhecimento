import streamlit as tf
import cv2
import mediapipe as mp
import numpy as np

# Configuração da página do Streamlit (Interface)
st.set_page_config(
    page_title="IA Rastreamento Humano",
    page_icon="🤖",
    layout="wide"
)

# Estilização CSS para deixar a interface moderna
st.markdown("""
    <style>
    .main { background-color: #0f172a; color: #f8fafc; }
    h1 { color: #38bdf8; font-family: 'Helvetica Neue', sans-serif; }
    .stButton>button { background-color: #38bdf8; color: white; border-radius: 8px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🤖 IA de Rastreamento de Movimentos e Expressões")
st.write("Esta inteligência artificial analisa expressões faciais, mãos e postura corporal em tempo real.")

# Inicializando o MediaPipe Holistic (engloba rosto, mãos e corpo)
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Sidebar para configurações
st.sidebar.header("⚙️ Configurações da IA")
detection_confidence = st.sidebar.slider("Confiança Mínima de Detecção", 0.0, 1.0, 0.5)
tracking_confidence = st.sidebar.slider("Confiança Mínima de Rastreamento", 0.0, 1.0, 0.5)

# Seleção de entrada
opcoes_entrada = ["Webcam Local", "Vídeo de Exemplo"]
escolha = st.sidebar.selectbox("Escolha a fonte de entrada:", opcoes_entrada)

# Área principal de exibição do vídeo
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📹 Feed de Vídeo com IA Ativa")
    FRAME_WINDOW = st.image([]) # Janela onde o vídeo atualizado vai aparecer

with col2:
    st.subheader("📊 Status das Reações")
    status_placeholder = st.empty()
    status_placeholder.info("Aguardando inicialização da câmera...")

# Botão para iniciar/parar
run = st.checkbox('Ativar Câmera/Processamento')

# Inicializa o modelo Holistic do MediaPipe
with mp_holistic.Holistic(
    min_detection_confidence=detection_confidence,
    min_tracking_confidence=tracking_confidence
) as holistic:

    if run:
        # Se for local, usa a webcam (0). Nota sobre o Render abaixo.
        cap = cv2.VideoCapture(0)
        status_placeholder.success("IA conectada com sucesso!")

        while cap.isOpened() and run:
            ret, frame = cap.read()
            if not ret:
                st.write("Não foi possível acessar a câmera.")
                break

            # Inverte horizontalmente para efeito de espelho e muda BGR para RGB
            frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
            frame.flags.writeable = False
            
            # Processa o frame com a IA
            results = holistic.process(frame)
            
            frame.flags.writeable = True
            
            # Desenha os pontos de rastreamento no frame
            # 1. Rosto (Expressões)
            if results.face_landmarks:
                mp_drawing.draw_landmarks(
                    frame, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                    mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                    mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                )
            
            # 2. Postura (Movimentação do corpo)
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                )

            # 3. Mãos (Esquerda e Direita)
            if results.left_hand_landmarks:
                mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            if results.right_hand_landmarks:
                mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

            # Atualiza o frame na interface do Streamlit
            FRAME_WINDOW.image(frame)
            
            # Lógica simples de exibição de dados na barra lateral
            if results.face_landmarks:
                status_placeholder.metric(label="Rosto Detectado", value="Ativo", delta="Analisando Expressões")
            else:
                status_placeholder.metric(label="Rosto Detectado", value="Inativo", delta=None)

        cap.release()
    else:
        status_placeholder.warning("Sistema pausado.")
