import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.io.wavfile as wav
from scipy.signal import butter, filtfilt
from scipy.io import wavfile

st.set_page_config(
    page_title="Laboratorio de Señales",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------
# Menú lateral
# -------------------
st.sidebar.title("📘 Laboratorio Nº 1")
menu = st.sidebar.selectbox(
    "Selecciona el punto",
    ["Inicio", "Punto 1", "Punto 2", "Punto 3", "Punto 4"]
)

# ================================================================
# INICIO
# ================================================================
if menu == "Inicio":
    st.markdown("<h1 style='text-align: center;'>Transformación de señales, Señales y Sistemas</h1>", unsafe_allow_html=True)
    st.write("Bienvenida/o 🙌. Usa el menú de la izquierda para navegar entre los puntos del laboratorio.")

    st.subheader("Descripción general")
    st.write(
        "El laboratorio tiene como fin poner en práctica los conceptos teóricos adquiridos en el curso de Señales "
        "y Sistemas sobre las operaciones básicas de transformación de señales y trasladarlas a un escenario "
        "gráfico a nivel computacional. Las operaciones que se llevan a cabo en este trabajo son el desplazamiento "
        "y el escalamiento en el tiempo, junto con su equivalente en lo denominado diezmado e interpolación."
    )

elif menu == "Punto 3":
    st.header("🎧 Punto 3 – Modulación y Demodulación Estéreo")

    st.subheader("Carga de archivos WAV")
    x1_file = st.file_uploader("Sube archivo x1.wav", type=["wav"])
    x2_file = st.file_uploader("Sube archivo x2.wav", type=["wav"])

    if x1_file is not None and x2_file is not None:
        import io
        fs1, x1_t = wav.read(io.BytesIO(x1_file.read()))
        fs2, x2_t = wav.read(io.BytesIO(x2_file.read()))

        N = min(len(x1_t), len(x2_t))
        x1_t = x1_t[:N]
        x2_t = x2_t[:N]

        if x1_t.ndim == 2:
            x1_t = np.mean(x1_t, axis=1)
        if x2_t.ndim == 2:
            x2_t = np.mean(x2_t, axis=1)

        delta1 = 1/fs1
        fp = 30000
        t = np.arange(N)*delta1

        # Modulación
        x1_mod_t = x1_t*np.cos(2*np.pi*fp*t)
        x2_mod_t = x2_t*np.sin(2*np.pi*fp*t)
        y1_t = x1_mod_t + x2_mod_t
        y1_t = y1_t / np.max(np.abs(y1_t))

        def plot_signal(x, t, title):
            fig, ax = plt.subplots()
            ax.plot(t, x)
            ax.set_xlabel("Tiempo (s)")
            ax.set_ylabel("Amplitud")
            ax.set_title(title)
            st.pyplot(fig)

        Ts = 1/fs1
        X_f = lambda x: np.fft.fftshift(np.fft.fft(x))
        Delta_f = 1/(N*Ts)
        f = np.arange(-N/2, N/2)*Delta_f

        def plot_fft(X, title):
            fig, ax = plt.subplots()
            ax.plot(f, np.abs(X)/N)
            ax.set_xlabel("Frecuencia (Hz)")
            ax.set_ylabel("Amplitud")
            ax.set_title(title)
            st.pyplot(fig)

        # ---------- Secciones con expander ----------
        with st.expander("📊 Señales en tiempo"):
            plot_signal(x1_t, t, "x1_t")
            plot_signal(x1_mod_t, t, "x1_t * cos(w0 t)")
            plot_signal(x2_t, t, "x2_t")
            plot_signal(x2_mod_t, t, "x2_t * sin(w0 t)")
            plot_signal(y1_t, t, "y1_t (señal modulada)")

        with st.expander("📊 Transformadas FFT"):
            plot_fft(X_f(x1_t), "X1(f)")
            plot_fft(X_f(x1_mod_t), "X1_mod(f)")
            plot_fft(X_f(x2_t), "X2(f)")
            plot_fft(X_f(x2_mod_t), "X2_mod(f)")
            plot_fft(X_f(y1_t), "Y1(f)")

        # ---------- Demodulación ----------
        z1_t = y1_t*np.cos(2*np.pi*fp*t)
        z2_t = y1_t*np.sin(2*np.pi*fp*t)
        Z1_f = X_f(z1_t)
        Z2_f = X_f(z2_t)

        with st.expander("🎛️ Demodulación (Tiempo y FFT)"):
            plot_signal(z1_t, t, "z1_t")
            plot_signal(z2_t, t, "z2_t")
            plot_fft(Z1_f, "Z1(f)")
            plot_fft(Z2_f, "Z2(f)")

        # ---------- Filtro Pasabajos ----------
        fpb = np.abs(f) <= 6000
        Z1_f_fil = Z1_f * fpb
        Z2_f_fil = Z2_f * fpb
        Z1_fil_t = np.fft.ifft(np.fft.ifftshift(Z1_f_fil))
        Z2_fil_t = np.fft.ifft(np.fft.ifftshift(Z2_f_fil))

        with st.expander("🔻 Filtro Pasabajos y Señales Filtradas"):
            plot_fft(Z1_f_fil, "Z1 filtrada (f)")
            plot_fft(Z2_f_fil, "Z2 filtrada (f)")
            plot_signal(Z1_fil_t.real, t, "z1_t filtrada")
            plot_signal(Z2_fil_t.real, t, "z2_t filtrada")

        # ---------- Audio ----------
        with st.expander("🔊 Reproducción de audio"):
            st.write("👉 Señales originales")
            st.audio(x1_t.astype(np.float32), sample_rate=fs1)
            st.audio(x2_t.astype(np.float32), sample_rate=fs1)
            st.write("👉 Señales demoduladas y filtradas")
            st.audio(Z1_fil_t.real.astype(np.float32), sample_rate=fs1)
            st.audio(Z2_fil_t.real.astype(np.float32), sample_rate=fs1)
elif menu == "Punto 1":
    st.header("📈 Punto 1 – Series de Fourier")

    # Selección de función
    op = st.selectbox(
        "Indique función de interés:",
        ("Función 1", "Función 2", "Función 3", "Función 4")
    )

    # Convertimos la opción en número
    func_num = {"Función 1":1, "Función 2":2, "Función 3":3, "Función 4":4}.get(op,0)

    # Número de armónicos
    N = st.number_input("Número de armónicos", min_value=1, max_value=100, value=10, step=1)

    fs = 0.01
    a = np.zeros(N+1)
    b = np.zeros(N+1)
    c = np.zeros(N+1)
    k = np.arange(0, N+1)

    # Funciones auxiliares
    def plot_espectro(n, xn, title="Espectro en línea"):
        fig, ax = plt.subplots(figsize=(8,4))
        ax.stem(n, xn)  # Se quita use_line_collection para compatibilidad
        ax.set_title(title)
        ax.set_xlabel("Armónico")
        ax.set_ylabel("Amplitud")
        ax.grid(True)
        st.pyplot(fig)

    def graficar(t, xt, title="Reconstrucción de la señal"):
        fig, ax = plt.subplots(figsize=(8,4))
        ax.plot(t, xt)
        ax.set_title(title)
        ax.set_xlabel("Tiempo")
        ax.set_ylabel("x(t)")
        ax.grid(True)
        st.pyplot(fig)

    # ---------------------------------------------------------
    # Función 1
    # ---------------------------------------------------------
    if func_num == 1:
        T = 4
        for n in range(N+1):
            if n==0:
                a[n] = 0
            elif n%2==1:
                a[n] = 8*(n*np.pi)**-2
            else:
                a[n] = 0

        with st.expander("📊 Espectro"):
            plot_espectro(k, a)

        t = np.arange(-1.5*T, 1.5*T + fs, fs)
        xt = np.zeros_like(t)
        for n in k:
            if n==0:
                xt += a[n]
            else:
                xt += a[0] + a[n] * np.cos(n*(2*np.pi/T)*t)

        with st.expander("📊 Señal reconstruida"):
            graficar(t, xt)

    # ---------------------------------------------------------
    # Función 2
    # ---------------------------------------------------------
    elif func_num == 2:
        T = 2*np.pi
        for n in range(N+1):
            if n==0:
                b[n]=0
            else:
                b[n] = -2*np.cos(n*np.pi)/n

        with st.expander("📊 Espectro"):
            plot_espectro(k, b)

        t = np.arange(-1.5*T, 1.5*T + fs, fs)
        xt = np.zeros_like(t)
        for n in k:
            if n==0:
                xt += a[n]
            else:
                xt += a[0] + b[n] * np.sin(n*t)

        with st.expander("📊 Señal reconstruida"):
            graficar(t, xt)

    # ---------------------------------------------------------
    # Función 3
    # ---------------------------------------------------------
    elif func_num == 3:
        T = 2*np.pi
        for n in range(N+1):
            if n==0:
                a[n] = (np.pi**2)/3
            else:
                a[n] = 4*np.cos(n*np.pi)/n**2

        with st.expander("📊 Espectro"):
            plot_espectro(k, a)

        t = np.arange(-1.5*T, 1.5*T + fs, fs)
        xt = np.zeros_like(t)
        for n in k:
            if n==0:
                xt += a[n]
            else:
                xt += a[n]*np.cos(n*t)
        xtt = (np.pi**2)/3 + xt

        with st.expander("📊 Señal reconstruida"):
            graficar(t, xtt)

    # ---------------------------------------------------------
    # Función 4
    # ---------------------------------------------------------
    elif func_num == 4:
        T = 3.5
        for n in range(N+1):
            if n==0:
                c[n]=0.25
            elif n%2==1:
                c[n] = (1/(n*np.pi))*((4/(n*np.pi)**2)+9)**0.5
            else:
                c[n] = 1/(n*np.pi)

        with st.expander("📊 Espectro"):
            plot_espectro(k, c)

        t = np.arange(-T, T + fs, fs)
        xt = np.zeros_like(t)
        for n in k:
            if n==0:
                xt += a[n]
            else:
                xt += (1/(n**2 * np.pi**2))*(1-(-1)**n)*np.cos(n*np.pi*t) + (1/(n*np.pi))*(1-2*(-1)**n)*np.sin(n*np.pi*t)
        xtt = 0.25 + xt

        with st.expander("📊 Señal reconstruida"):
            graficar(t, xtt)

    else:
        st.warning("Opción no válida")
elif menu == "Punto 4":

    st.header("📡 Punto 4 – Modulación AM (según notebook)")

    # Parámetros
    fs = 6000
    Ts = 1/fs
    t = np.arange(0, 3, Ts)

    f0 = 800
    w0 = 2*np.pi*f0

    f1, f2, f3 = 100, 200, 400
    w1, w2, w3 = 2*np.pi*f1, 2*np.pi*f2, 2*np.pi*f3

    h = 300  # muestras a graficar

    # Señales mensaje individuales
    y1_t = 3*np.cos(w1*t)
    y2_t = 4*np.cos(w2*t)
    y3_t = 5*np.cos(w3*t)
    y_t = y1_t + y2_t + y3_t

    # Portadora
    p_t = np.cos(w0*t)

    # Modulación AM
    y_mod_t = (1 + y_t) * p_t

    # FFT
    N = len(t)
    f = np.arange(-N/2, N/2) * (fs/N)

    Y1_f = np.fft.fftshift(np.fft.fft(y1_t)) / N
    Y2_f = np.fft.fftshift(np.fft.fft(y2_t)) / N
    Y3_f = np.fft.fftshift(np.fft.fft(y3_t)) / N
    Y_f  = np.fft.fftshift(np.fft.fft(y_t)) / N
    Y_mod_f = np.fft.fftshift(np.fft.fft(y_mod_t)) / N

    # -----------------------------
    # Señales individuales
    # -----------------------------
    with st.expander("📊 Señales individuales (y1, y2, y3)"):
        signals = [y1_t, y2_t, y3_t]
        labels = ["y1(t)", "y2(t)", "y3(t)"]

        for sig, label in zip(signals, labels):
            fig, ax = plt.subplots(figsize=(10,4))
            ax.plot(t[:h], sig[:h])
            ax.set_title(label)
            ax.set_xlabel("Tiempo (s)")
            ax.set_ylabel("Amplitud")
            ax.grid()
            st.pyplot(fig)

    # -----------------------------
    # Señal mensaje y señal modulada
    # -----------------------------
    with st.expander("📡 Señal mensaje y modulada"):
        to_plot = [y_t, y_mod_t]
        labels = ["y(t) — Mensaje", "y_mod(t) — Señal Modulada AM"]

        for sig, label in zip(to_plot, labels):
            fig, ax = plt.subplots(figsize=(10,4))
            ax.plot(t[:h], sig[:h])
            ax.set_title(label)
            ax.set_xlabel("Tiempo (s)")
            ax.set_ylabel("Amplitud")
            ax.grid()
            st.pyplot(fig)

    # -----------------------------
    # FFT señales individuales
    # -----------------------------
    with st.expander("📈 FFT – Señales individuales"):
        ffts = [Y1_f, Y2_f, Y3_f]
        labels = ["Y1(f)", "Y2(f)", "Y3(f)"]

        for Y, label in zip(ffts, labels):
            fig, ax = plt.subplots(figsize=(10,4))
            ax.plot(f, np.abs(Y))
            ax.set_title(label)
            ax.set_xlabel("Frecuencia (Hz)")
            ax.set_ylabel("Magnitud")
            ax.grid()
            st.pyplot(fig)

    # -----------------------------
    # FFT mensaje y modulada
    # -----------------------------
    with st.expander("📡 FFT – Mensaje y Señal AM"):
        ffts = [Y_f, Y_mod_f]
        labels = ["Y(f)", "Y_mod(f)"]

        for Y, label in zip(ffts, labels):
            fig, ax = plt.subplots(figsize=(10,4))
            ax.plot(f, np.abs(Y))
            ax.set_title(label)
            ax.set_xlabel("Frecuencia (Hz)")
            ax.set_ylabel("Magnitud")
            ax.grid()
            st.pyplot(fig)

    # -----------------------------
    # Señales moduladas individuales
    # -----------------------------
    y1_mod = y1_t * p_t
    y2_mod = y2_t * p_t
    y3_mod = y3_t * p_t

    with st.expander("📈 Señales moduladas individuales"):
        signals = [y1_mod, y2_mod, y3_mod]
        labels = ["y1_mod(t)", "y2_mod(t)", "y3_mod(t)"]

        for sig, label in zip(signals, labels):
            fig, ax = plt.subplots(figsize=(10,4))
            ax.plot(t[:h], sig[:h])
            ax.set_title(label)
            ax.set_xlabel("Tiempo (s)")
            ax.set_ylabel("Amplitud")
            ax.grid()
            st.pyplot(fig)

    # -----------------------------
    # Modulación AM con diferentes μ
    # -----------------------------
    mu_values = [1.2, 1.0, 0.7]

    for mu in mu_values:
        y_mod_mu = (1 + mu * y_t) * p_t
        Y_mod_mu = np.fft.fftshift(np.fft.fft(y_mod_mu)) / N

        with st.expander(f"📡 Modulación AM para μ = {mu}"):

            fig, ax = plt.subplots(figsize=(10,4))
            ax.plot(t[:600], y_mod_mu[:600])
            ax.set_title(f"Señal AM (μ = {mu}) — Tiempo")
            ax.set_xlabel("Tiempo (s)")
            ax.set_ylabel("Amplitud")
            ax.grid()
            st.pyplot(fig)

            fig, ax = plt.subplots(figsize=(10,4))
            ax.plot(f, np.abs(Y_mod_mu))
            ax.set_title(f"FFT AM (μ = {mu})")
            ax.set_xlabel("Frecuencia (Hz)")
            ax.set_ylabel("Magnitud")
            ax.grid()
            st.pyplot(fig)

    # -----------------------------
    # Rectificación media onda
    # -----------------------------
    for mu in mu_values:
        y_mod_mu = (1 + mu*y_t) * p_t
        y_rect = np.maximum(y_mod_mu, 0)

        with st.expander(f"📈 Rectificación media onda — μ = {mu}"):
            fig, ax = plt.subplots(figsize=(10,4))
            ax.plot(t[:600], y_rect[:600])
            ax.set_title(f"Rectificación media onda (μ = {mu})")
            ax.set_xlabel("Tiempo (s)")
            ax.set_ylabel("Amplitud")
            ax.grid()
            st.pyplot(fig)

elif menu == "Punto 2":

    st.header("🎧 Punto 2 – Modulación AM de una señal de audio")

    st.subheader("1️⃣ Cargar archivo de audio WAV")

    audio_file = st.file_uploader("Sube un archivo .wav", type=["wav"])

    if audio_file is not None:
        fs_audio, señal_audio = wavfile.read(audio_file)

        # ✔️ Convertir a MONO si el audio es estéreo
        if len(señal_audio.shape) > 1:
            señal_audio = señal_audio.mean(axis=1)

        # ✔️ Normalizar y asegurar formato float32
        señal_audio = señal_audio.astype(np.float32)
        señal_audio = señal_audio / np.max(np.abs(señal_audio))

        # Crear eje de tiempo
        dur = len(señal_audio) / fs_audio
        t_audio = np.linspace(0, dur, len(señal_audio))

        # --- Gráfica Audio Original ---
        with st.expander("📊 Señal de audio original"):
            fig, ax = plt.subplots(figsize=(10,4))
            ax.plot(t_audio, señal_audio)
            ax.set_title("Audio Original en el Tiempo")
            ax.set_xlabel("Tiempo (s)")
            ax.set_ylabel("Amplitud")
            ax.grid()
            st.pyplot(fig)

        # ------------------------------------------------------------
        # 2️⃣ Parámetros de modulación
        # ------------------------------------------------------------
        fp_mod = 40000              # Frecuencia de la portadora
        fs_mod = 1e6                # Tasa de muestreo para modulación
        amp_port = 1                # Amplitud portadora
        filtro_corte = 4000         # Filtro pasa bajos para recuperar audio

        # Resampleo del audio
        t_mod = np.arange(0, dur, 1/fs_mod)
        señal_base = np.interp(t_mod, t_audio, señal_audio)

        with st.expander("📊 Señal de audio interpolada para modulación"):
            fig, ax = plt.subplots(figsize=(10,4))
            ax.plot(t_mod[:5000], señal_base[:5000])
            ax.set_title("Señal Base Interpolada (Zoom)")
            ax.set_xlabel("Tiempo (s)")
            ax.set_ylabel("Amplitud")
            ax.grid()
            st.pyplot(fig)

        # ------------------------------------------------------------
        # 3️⃣ Generar portadora
        # ------------------------------------------------------------
        portadora = amp_port * np.cos(2*np.pi*fp_mod*t_mod)

        with st.expander("📡 Portadora"):
            fig, ax = plt.subplots(figsize=(10,4))
            ax.plot(t_mod[:2000], portadora[:2000])
            ax.set_title("Portadora AM")
            ax.set_xlabel("Tiempo (s)")
            ax.set_ylabel("Amplitud")
            ax.grid()
            st.pyplot(fig)

        # ------------------------------------------------------------
        # 4️⃣ FFT de señal base
        # ------------------------------------------------------------
        N = len(señal_base)
        f = np.fft.fftshift(np.fft.fftfreq(N, 1/fs_mod))
        X_base = np.abs(np.fft.fftshift(np.fft.fft(señal_base))) / N

        with st.expander("📈 FFT de la señal base"):
            fig, ax = plt.subplots(figsize=(10,4))
            ax.plot(f, X_base)
            ax.set_title("FFT de la Señal Base")
            ax.set_xlabel("Frecuencia (Hz)")
            ax.set_ylabel("Magnitud")
            ax.set_xlim(-60000, 60000)
            ax.grid()
            st.pyplot(fig)

        # ------------------------------------------------------------
        # 5️⃣ Modulación AM (DSB-SC)
        # ------------------------------------------------------------
        señal_mod = señal_base * portadora
        X_mod = np.abs(np.fft.fftshift(np.fft.fft(señal_mod))) / N

        with st.expander("📡 Señal Modulada AM"):
            fig, ax = plt.subplots(figsize=(10,4))
            ax.plot(t_mod[:2000], señal_mod[:2000])
            ax.set_title("Señal Modulada (Zoom)")
            ax.set_xlabel("Tiempo (s)")
            ax.set_ylabel("Amplitud")
            ax.grid()
            st.pyplot(fig)

        with st.expander("📈 FFT de señal modulada"):
            fig, ax = plt.subplots(figsize=(10,4))
            ax.plot(f, X_mod)
            ax.set_title("FFT de Señal Modulada")
            ax.set_xlabel("Frecuencia (Hz)")
            ax.set_ylabel("Magnitud")
            ax.set_xlim(-60000,60000)
            ax.grid()
            st.pyplot(fig)

        # ------------------------------------------------------------
        # 6️⃣ Demodulación (multiplicación por portadora)
        # ------------------------------------------------------------
        señal_demod = señal_mod * portadora
        X_demod = np.abs(np.fft.fftshift(np.fft.fft(señal_demod))) / N

        with st.expander("🎚 Señal Demodulada (sin filtrar)"):
            fig, ax = plt.subplots(figsize=(10,4))
            ax.plot(t_mod[:2000], señal_demod[:2000])
            ax.set_title("Demodulación por Multiplicación")
            ax.set_xlabel("Tiempo (s)")
            ax.set_ylabel("Amplitud")
            ax.grid()
            st.pyplot(fig)

        with st.expander("📈 FFT de señal demodulada"):
            fig, ax = plt.subplots(figsize=(10,4))
            ax.plot(f, X_demod)
            ax.set_title("FFT de la Señal Demodulada")
            ax.set_xlabel("Frecuencia (Hz)")
            ax.set_ylabel("Magnitud")
            ax.grid()
            st.pyplot(fig)

        # ------------------------------------------------------------
        # 7️⃣ Filtrado pasa bajos para recuperar audio
        # ------------------------------------------------------------
        b, a = butter(6, filtro_corte/(fs_mod/2), btype='low')
        señal_recuperada = filtfilt(b, a, señal_demod)

        with st.expander("🔊 Señal recuperada (DOMINIO DEL TIEMPO)"):
            fig, ax = plt.subplots(figsize=(10,4))
            ax.plot(t_mod[:5000], señal_recuperada[:5000])
            ax.set_title("Señal AM Recuperada")
            ax.set_xlabel("Tiempo (s)")
            ax.set_ylabel("Amplitud")
            ax.grid()
            st.pyplot(fig)

        st.success("✅ Señal recuperada correctamente.")
