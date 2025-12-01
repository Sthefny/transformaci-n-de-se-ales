import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.io.wavfile as wav
from scipy.signal import butter, filtfilt
from scipy.io import wavfile
from scipy import signal

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
    st.markdown(
        """
        Este proyecto corresponde al **Laboratorio 3** de la asignatura *Señales y Sistemas* de la Universidad del Norte. Su propósito es aplicar los conceptos fundamentales de las **Series de Fourier**, la **Transformada de Fourier** y los métodos de **modulación y demodulación** mediante aplicaciones interactivas en **Python** con **GUI**.

        A lo largo del laboratorio se implementan herramientas para:
        - Analizar señales periódicas mediante coeficientes de Fourier y su reconstrucción armónica.
        - Estudiar la modulación en amplitud (AM), desde la filtración de una señal de audio real hasta su demodulación y recuperación.
        - Explorar la multiplexación en cuadratura (señales en fase y en cuadratura).
        - Realizar modulación **DSB-LC** con distintos índices de modulación.

        Cada módulo incluye visualizaciones en el **dominio del tiempo** y en el **dominio de la frecuencia (DEP)**, facilitando una comprensión gráfica e interactiva de la teoría aplicada.
        """
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


    # ----- Señal moduladora -----
    fs = 2000
    Ts = 1/fs
    t = np.arange(0, 2+Ts, Ts)

    f1, A = 5, 0.5
    f2, B = 15, 0.3
    f3, C = 20, 0.2

    y1 = A * np.sin(2 * np.pi * f1 * t)
    y2 = B * np.sin(2 * np.pi * f2 * t)
    y3 = C * np.sin(2 * np.pi * f3 * t)
    y = y1 + y2 + y3

    # ----- Portadora -----
    fc = 500 # Frecuencia de la portadora
    fs_c = 20 * fc
    Ts_c = 1 / fs_c
    t_c = np.arange(0, 2, Ts_c)
    portadora = np.cos(2 * np.pi * fc * t_c)

    # Interpolación de la señal moduladora
    y_interp = np.interp(t_c, t, y)

    # ----- Índices de modulación -----
    mu1 = 0.7
    mu2 = 1
    mu3 = 2

    dsb_lc_mod1 = (1 + mu1 * y_interp) * portadora
    dsb_lc_mod2 = (1 + mu2 * y_interp) * portadora
    dsb_lc_mod3 = (1 + mu3 * y_interp) * portadora

    # ----- Funciones para graficar -----
    def graficar_tiempo(signal, t_vector, titulo, t_max=None):
        fig, ax = plt.subplots(figsize=(10,3))
        if t_max is not None:
            idx = t_vector <= t_max
            ax.plot(t_vector[idx], signal[idx])
        else:
            ax.plot(t_vector, signal)
        ax.set_title(titulo)
        ax.set_xlabel("Tiempo (s)")
        ax.set_ylabel("Amplitud")
        ax.grid(True)
        st.pyplot(fig)

    def graficar_fft(signal, fs, titulo):
        N = len(signal)
        fft_cent = np.fft.fftshift(np.fft.fft(signal))
        Delta_f = fs / N
        f = np.arange(-N//2, N//2) * Delta_f
        magnitud = np.abs(fft_cent) / N
        fig, ax = plt.subplots(figsize=(10,3))
        ax.plot(f, magnitud)
        ax.set_title(titulo)
        ax.set_xlabel("Frecuencia (Hz)")
        ax.set_ylabel("Magnitud")
        ax.set_xlim(-700, 700)
        ax.grid(True)
        st.pyplot(fig)

    def graficar_rectificada(signal, t_vector, titulo, t_max=None):
        fig, ax = plt.subplots(figsize=(10,3))
        if t_max is not None:
            idx = t_vector <= t_max
            ax.plot(t_vector[idx], np.abs(signal[idx]))
            ax.set_xlim(0, t_max)
        else:
            ax.plot(t_vector, np.abs(signal))
        ax.set_title(titulo)
        ax.set_xlabel("Tiempo (s)")
        ax.set_ylabel("Amplitud")
        ax.grid(True)
        st.pyplot(fig)

    # ----- Streamlit menú -----
    st.title("Punto 4: Señales y AM-DSB-LC")

    seccion = st.selectbox("Selecciona la sección a visualizar:", 
                        ["Señales moduladoras", "Señal AM-DSB-LC"])

    # ----- Visualización de señales moduladoras -----
    if seccion == "Señales moduladoras":
        senal_mod = st.selectbox("Selecciona la señal moduladora:", ["y1", "y2", "y3", "y = y1+y2+y3"])
        
        if senal_mod == "y1":
            sig = y1
            titulo = "Señal y1 (5 Hz, A=0.5)"
        elif senal_mod == "y2":
            sig = y2
            titulo = "Señal y2 (15 Hz, A=0.3)"
        elif senal_mod == "y3":
            sig = y3
            titulo = "Señal y3 (20 Hz, A=0.2)"
        else:
            sig = y
            titulo = "Señal y = y1 + y2 + y3"
        
        graficar_tiempo(sig, t, titulo, t_max=0.25)
        graficar_fft(sig, fs, f"Espectro de {titulo}")

    # ----- Visualización de señales AM-DSB-LC -----
    else:
        senal_am = st.selectbox("Selecciona la señal AM:", ["μ = 0.7", "μ = 1", "μ = 2"])
        
        if senal_am == "μ = 0.7":
            sig = dsb_lc_mod1
            mu_val = mu1
        elif senal_am == "μ = 1":
            sig = dsb_lc_mod2
            mu_val = mu2
        else:
            sig = dsb_lc_mod3
            mu_val = mu3
        
        t_max = st.slider("Tiempo máximo a graficar (s)", 0.01, 0.5, 0.25, 0.01)
        
        graficar_tiempo(sig, t_c, f"AM-DSB-LC en el tiempo, μ = {mu_val}", t_max)
        graficar_fft(sig, fs_c, f"Espectro AM-DSB-LC, μ = {mu_val}")
        graficar_rectificada(sig, t_c, f"Señal rectificada AM-DSB-LC, μ = {mu_val}", t_max)
elif menu == "Punto 2":

    # ----- Parámetros -----
    fp_mod = 40000      # Frecuencia de la portadora
    fs_mod = 1e5        # Frecuencia de muestreo para procesamiento (100 kHz)
    amp_port = 1
    filtro_corte = 4000 # Hz
    duracion_max = 8  # Analizar solo los primeros 0.5 s

    # ----- Subir archivo de audio -----
    uploaded_file = st.file_uploader("Sube tu archivo .wav", type="wav")

    if uploaded_file is not None:
        fs_audio, señal_audio = wavfile.read(uploaded_file)

        # Si es estéreo, pasar a mono
        if señal_audio.ndim == 2:
            señal_audio = np.mean(señal_audio, axis=1)

        # Tomar solo los primeros duracion_max segundos
        num_muestras = int(duracion_max * fs_audio)
        señal_audio = señal_audio[:num_muestras]

        # Interpolación a nueva frecuencia de muestreo
        nuevo_N = int(len(señal_audio) * fs_mod / fs_audio)
        senal_interp = signal.resample(señal_audio, nuevo_N)

        N = len(senal_interp)
        t = np.arange(N) / fs_mod

        # ----- Filtrado en frecuencia -----
        f = np.linspace(-fs_mod/2, fs_mod/2, N)
        X_f = np.fft.fftshift(np.fft.fft(senal_interp))
        H_lp = np.abs(f) <= filtro_corte
        X_filtrada = X_f * H_lp
        senal_base = np.real(np.fft.ifft(np.fft.ifftshift(X_filtrada)))

        # ----- Graficar señal base -----
        fig, ax = plt.subplots(figsize=(10,3))
        ax.plot(t, senal_base, color='orange')
        ax.set_title("Señal Base - Dominio del Tiempo")
        ax.set_xlabel("Tiempo (s)")
        ax.set_ylabel("Amplitud")
        ax.grid(True)
        st.pyplot(fig)

        # ----- Portadora -----
        portadora = amp_port * np.cos(2 * np.pi * fp_mod * t)

        # Graficar portadora (primeras 50 muestras)
        fig, ax = plt.subplots(figsize=(10,3))
        ax.plot(t[:50], portadora[:50], color='crimson')
        ax.set_title("Portadora Coseno 40kHz")
        ax.set_xlabel("Tiempo (s)")
        ax.set_ylabel("Amplitud")
        ax.grid(True)
        st.pyplot(fig)

        # ----- Espectros -----
        X_base = np.abs(np.fft.fftshift(np.fft.fft(senal_base))) / N
        X_port = np.abs(np.fft.fftshift(np.fft.fft(portadora))) / N
        frecs = np.linspace(-fs_mod/2, fs_mod/2, N)

        # Espectro señal base
        fig, ax = plt.subplots(figsize=(10,3))
        ax.plot(frecs, X_base, color='orange')
        ax.set_title("Espectro Señal Base")
        ax.set_xlabel("Frecuencia (Hz)")
        ax.set_ylabel("Magnitud")
        ax.set_xlim(-10e3, 10e3)
        ax.grid(True)
        st.pyplot(fig)

        # Espectro portadora
        fig, ax = plt.subplots(figsize=(10,3))
        ax.plot(frecs, X_port, color='crimson')
        ax.set_title("Espectro Portadora")
        ax.set_xlabel("Frecuencia (Hz)")
        ax.set_ylabel("Magnitud")
        ax.set_xlim(-60e3, 60e3)
        ax.grid(True)
        st.pyplot(fig)

        # ----- Modulación -----
        senal_mod = senal_base * portadora
        X_mod = np.abs(np.fft.fftshift(np.fft.fft(senal_mod))) / N

        # Tiempo
        fig, ax = plt.subplots(figsize=(10,3))
        ax.plot(t, senal_mod, color='purple')
        ax.set_title("Señal Modulada - Tiempo")
        ax.set_xlabel("Tiempo (s)")
        ax.set_ylabel("Amplitud")
        ax.grid(True)
        st.pyplot(fig)

        # Frecuencia
        fig, ax = plt.subplots(figsize=(10,3))
        ax.plot(frecs, X_mod, color='purple')
        ax.set_title("Señal Modulada - Frecuencia")
        ax.set_xlabel("Frecuencia (Hz)")
        ax.set_ylabel("Magnitud")
        ax.set_xlim(-60e3,60e3)
        ax.grid(True)
        st.pyplot(fig)

        # ----- Demodulación -----
        senal_demod = senal_mod * portadora
        X_demod = np.abs(np.fft.fftshift(np.fft.fft(senal_demod))) / N

        # Tiempo
        fig, ax = plt.subplots(figsize=(10,3))
        ax.plot(t, senal_demod, color='teal')
        ax.set_title("Señal Demodulada - Tiempo")
        ax.set_xlabel("Tiempo (s)")
        ax.set_ylabel("Amplitud")
        ax.grid(True)
        st.pyplot(fig)

        # Frecuencia
        fig, ax = plt.subplots(figsize=(10,3))
        ax.plot(frecs, X_demod, color='teal')
        ax.set_title("Señal Demodulada - Frecuencia")
        ax.set_xlabel("Frecuencia (Hz)")
        ax.set_ylabel("Magnitud")
        ax.set_xlim(-10e3,10e3)
        ax.grid(True)
        st.pyplot(fig)

        # ----- Filtrado de la señal demodulada -----
        X_final = np.fft.fftshift(np.fft.fft(senal_demod)) * (np.abs(frecs) <= filtro_corte)
        senal_rec = np.real(np.fft.ifft(np.fft.ifftshift(X_final)))
        X_rec = np.abs(np.fft.fftshift(np.fft.fft(senal_rec))) / N

        # Señal recuperada - Tiempo
        fig, ax = plt.subplots(figsize=(10,3))
        ax.plot(t, senal_rec, color='blue')
        ax.set_title("Señal Recuperada - Tiempo")
        ax.set_xlabel("Tiempo (s)")
        ax.set_ylabel("Amplitud")
        ax.grid(True)
        st.pyplot(fig)

        # Señal recuperada - Frecuencia
        fig, ax = plt.subplots(figsize=(10,3))
        ax.plot(frecs, X_rec, color='blue')
        ax.set_title("Señal Recuperada - Frecuencia")
        ax.set_xlabel("Frecuencia (Hz)")
        ax.set_ylabel("Magnitud")
        ax.set_xlim(-10e3,10e3)
        ax.grid(True)
        st.pyplot(fig)

        st.success("Señal recuperada correctamente ✅")

        # ----- Reproducción de audio -----
        st.subheader("🎧 Reproduciendo Señales")
        st.audio(senal_rec.astype(np.float32), sample_rate=int(fs_mod), format='audio/wav')
        st.audio(senal_base.astype(np.float32), sample_rate=int(fs_mod), format='audio/wav')
        st.success("✅ Fin del procesamiento.")

