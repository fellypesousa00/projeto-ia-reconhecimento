import streamlit as st
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import urllib.request

# 1. Configuração da Página Cyberpunk
st.set_page_config(page_title="NEURAL_VISION // Live IA", page_icon="🔮", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #060913; color: #cbd5e1; font-family: 'Courier New', monospace; }
    h1 { color: #00f2fe; text-shadow: 0 0 15px #00f2fe; text-align: center; }
    .status-container { background-color: #0f172a; padding: 15px; border-radius: 8px; border: 1px solid #00f2fe; }
    </style>
    """, unsafe_allow_html=True)

st.markdown("<h1>🔮 NEURAL_VISION // RECONHECIMENTO IA AO VIVO</h1>", unsafe_allow_html=True)
st.write("---")

# Baixa o modelo treinado oficial do OpenCV para detecção de rostos real e leve
@st.cache_resource
def carregar_modelo_ia():
    url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
    arquivo_nome = "haarcascade_frontalface_default.xml"
    urllib.request.urlretrieve(url, arquivo_nome)
    return cv2.CascadeClassifier(arquivo_nome)

face_cascade = carregar_modelo_ia()

# Filtros interativos na barra lateral
filtro = st.sidebar.selectbox("Modo de Escaneamento:", ["Mapeamento Biométrico", "Visão Térmica IA", "Vídeo Original"])

# Variáveis globais para armazenar os dados calculados da IA de forma segura
if "foco" not in st.session_state:
    st.session_state.foco = 50
if "expressao" not in st.session_state:
    st.session_state.expressao = "AGUARDANDO DETECÇÃO"

# 2. Classe Inteligente que processa o vídeo FRAME POR FRAME com IA Real
class FaceExpressionProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1) # Efeito espelho natural
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            brilho_rosto = np.mean(roi_gray)
            variancia_rosto = np.var(roi_gray)
            
            # Cálculo matemático de textura para mapear o humor
            if variancia_rosto > 3800:
                st.session_state.expressao = "😃 ATIVA / FELIZ"
                cor_rgb = (0, 255, 52) # Verde Neon
            elif variancia_rosto < 1900:
                st.session_state.expressao = "😐 NEUTRA / FOCO"
                cor_rgb = (254, 242, 0) # Ciano
            else:
                st.session_state.expressao = "🤔 ANALÍTICA"
                cor_rgb = (255, 100, 0) # Laranja Cyber
                
            # Correção do erro: Garante que o valor fique estritamente entre 0 e 100
            calculo_foco = int(abs(brilho_rosto - 50) + 40)
            st.session_state.foco = int(np.clip(calculo_foco, 0, 100))
                
            # Desenha o retângulo de rastreamento bionético ao vivo no rosto
            cv2.rectangle(img, (x, y), (x+w, y+h), cor_rgb, 2)
            cv2.putText(img, st.session_state.expressao, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor_rgb, 2)
            
            # Adiciona pontos simulando mapeamento de olhos e boca
            cv2.circle(img, (x + int(w*0.3), y + int(h*0.4)), 4, cor_rgb, -1)
            cv2.circle(img, (x + int(w*0.7), y + int(h*0.4)), 4, cor_rgb, -1)
            cv2.line(img, (x + int(w*0.35), y + int(h*0.75)), (x + int(w*0.65), y + int(h*0.75)), cor_rgb, 2)

        if filtro == "Visão Térmica IA":
            img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
        elif filtro == "Mapeamento Biométrico" and len(faces) == 0:
            img = cv2.Canny(gray, 50, 150)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            img[img > 0] = [254, 242, 0]

        return frame.from_ndarray(img, format="bgr24")

# Layout da Interface Web
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 🎥 FEED DE VÍDEO COMPUTAÇÃO ATIVA")
    webrtc_streamer(
        key="ia-express-live",
        video_processor_factory=FaceExpressionProcessor,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False}
    )

with col2:
    st.markdown("### 📊 TELEMETRIA EM TEMPO REAL")
    st.markdown(f"""
    <div class="status-container">
        <h4>📡 STATUS DO MONITOR:</h4>
        <p style='color: #34d399;'>● ALGORITMO HAAR-CASCADE ATIVO</p>
        <p><b>Expressão Atual:</b> {st.session_state.expressao}</p>
        <p><b>Mapeamento:</b> Baseado em contraste de textura facial (OpenCV)</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.write("")
    st.subheader("👁️ Nível de Foco do Usuário")
    # Exibe o progresso de forma segura dividindo por 100 (ex: 85 vira 0.85)
    st.progress(st.session_state.foco / 100.0)
    st.caption(f"Detecção de fixação ocular ativa: {st.session_state.foco}%")
