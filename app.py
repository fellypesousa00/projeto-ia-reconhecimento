import streamlit as st
import cv2
import numpy as np

# Nova forma de importar o MediaPipe para evitar o erro de 'solutions'
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# SE O ERRO PERSISTIR, use esta alternativa clássica que o Render aceita:
try:
    import mediapipe.python.solutions.holistic as mp_holistic
    import mediapipe.python.solutions.drawing_utils as mp_drawing
except ImportError:
    # Fallback para compatibilidade de versões
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils

# Configuração da página do Streamlit (Interface)
st.set_page_config(
    page_title="IA Rastreamento Humano",
    page_icon="🤖",
    layout="wide"
)

# Estilização CSS para garantir o visual Dark Neon moderno
st.markdown("""
    <style>
    .main { background-color: #0f172a; color: #f8fafc; }
    h1 { color: #38bdf8; font-family: 'Helvetica Neue', sans-serif; }
    .stButton>button { background-color: #38bdf8; color: white; border-radius: 8px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🤖 IA de Rastreamento de Movimentos e Expressões")
st.write("Esta inteligência artificial analisa expressões faciais, mãos e postura corporal em tempo real.")

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
    FRAME_WINDOW = st.image([]) 

with col2:
    st.subheader("📊 Status das Reações")
    status_placeholder = st.empty()
    status_placeholder.info("Aguardando inicialização da câmera...")

# Botão para iniciar/parar
run = st.checkbox('Ativar Câmera/Processamento')

# Inicializa o modelo Holistic usando o bloco try/except que configuramos no topo
with mp_holistic.Holistic(
    min_detection_confidence=detection_confidence,
    min_tracking_confidence=tracking_confidence
) as holistic:

    if run:
        cap = cv2.VideoCapture(0)
        status_placeholder.success("IA conectada com sucesso!")

        while cap.isOpened() and run:
            ret, frame = cap.read()
            if not ret:
                st.write("Não foi possível acessar a câmera.")
                break

            frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
            frame.flags.writeable = False
            
            results = holistic.process(frame)
            frame.flags.writeable = True
            
            # 1. Rosto (Expressões)
            if results.face_landmarks:
                mp_drawing.draw_landmarks(
                    frame, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                    mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                    mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                )
            
            # 2. Postura (Corpo)
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                )

            # 3. Mãos
            if results.left_hand_landmarks:
                mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            if results.right_hand_landmarks:
                mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

            FRAME_WINDOW.image(frame)
            
            if results.face_landmarks:
                status_placeholder.metric(label="Rosto Detectado", value="Ativo", delta="Analisando Expressões")
            else:
                status_placeholder.metric(label="Rosto Detectado", value="Inativo", delta=None)

        cap.release()
    else:
        status_placeholder.warning("Sistema pausado.")
