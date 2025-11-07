import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time

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
    ["Inicio", "Punto 1", "Punto 2"]
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


# ================================================================
# PUNTO 1: SEÑALES ORIGINALES Y CONVOLUCIÓN
# ================================================================
elif menu == "Punto 1":
    st.markdown("<h2 style='text-align: center;'>Punto 1</h2>", unsafe_allow_html=True)

    tipo_senal = st.sidebar.selectbox(
        "Selecciona el tipo de señal",
        ("Señales continuas", "Señales discretas"),
    )

    # ============================================================
    # SEÑALES CONTINUAS
    if tipo_senal == "Señales continuas":


        st.title("🔁 Convolución Continua — Visualización Interactiva")

        # ---------- Parámetros ----------
        delta = 0.05
        FRAME_DELAY = 1e-25

        # ---------- Definición de señales ----------
        def crear_senales():
            t_a = np.arange(-1, 5 + delta, delta)
            x_a = np.piecewise(t_a, [t_a < 0, (t_a >= 0) & (t_a < 3), (t_a >= 3) & (t_a < 5), t_a >= 5],
                            [0, 2, -2, 0])

            t_b = np.arange(-2, 2 + delta, delta)
            x_b = np.piecewise(t_b, [t_b < -1, (t_b >= -1) & (t_b <= 1), t_b > 1],
                            [0, lambda t: -t, 0])

            t_c = np.arange(-2, 5 + delta, delta)
            x_c = np.piecewise(
                t_c,
                [t_c < -1, (t_c >= -1) & (t_c < 1), (t_c >= 1) & (t_c < 3), (t_c >= 3) & (t_c < 5), t_c >= 5],
                [0, 2, lambda t: -2*(t-1)+2, -2, 0]
            )

            t_d = np.arange(-3, 3 + delta, delta)
            x_d = np.piecewise(t_d, [t_d < -3, (t_d >= -3) & (t_d <= 3), t_d > 3],
                            [0, lambda t: np.exp(-np.abs(t)), 0])

            return {"a": (t_a, x_a), "b": (t_b, x_b), "c": (t_c, x_c), "d": (t_d, x_d)}

        senales = crear_senales()

        # ---------- Estado ----------
        if "cont_anim_running" not in st.session_state:
            st.session_state.cont_anim_running = False
        if "cont_anim_stop" not in st.session_state:
            st.session_state.cont_anim_stop = False

        # ---------- Interfaz de selección ----------
        with st.expander("🧩 Selección de señales"):
            col1, col2 = st.columns(2)
            with col1:
                s1 = st.selectbox("Selecciona la primera señal x(t):", list(senales.keys()), index=0)
            with col2:
                s2 = st.selectbox("Selecciona la segunda señal h(t):", list(senales.keys()), index=1)

        start, stop = st.columns([1, 1])
        start_btn = start.button("▶ Iniciar animación continua")
        stop_btn = stop.button("⏹ Detener animación")

        visor = st.empty()

        if stop_btn:
            st.session_state.cont_anim_stop = True

        # ---------- Obtener señales ----------
        t_x, x_t = senales[s1]
        t_h, h_t = senales[s2]

        # ---------- Cálculo de convolución ----------
        y_conv = np.convolve(x_t, h_t, mode='full') * delta
        t_conv = np.linspace(t_x[0] + t_h[0], t_x[-1] + t_h[-1], len(y_conv))

        # ---------- Vista previa ----------
        with st.container():
            fig_prev, axs_prev = plt.subplots(2, 1, figsize=(10, 5))
            fig_prev.suptitle("Vista previa de las señales seleccionadas", fontsize=14, fontweight="bold")
            axs_prev[0].plot(t_x, x_t, linewidth=2, label="x(t)")
            axs_prev[0].legend(); axs_prev[0].grid(alpha=0.4)
            axs_prev[1].plot(t_h, h_t, linewidth=2, color="C1", label="h(t)")
            axs_prev[1].legend(); axs_prev[1].grid(alpha=0.4)
            fig_prev.tight_layout(rect=[0, 0.03, 1, 0.95])
            visor.pyplot(fig_prev, clear_figure=True)
            plt.close(fig_prev)

        # ---------- Función de animación ----------
        def animar_convolucion():
            st.session_state.cont_anim_stop = False
            st.session_state.cont_anim_running = True

            for k, t_k in enumerate(t_conv):
                if st.session_state.cont_anim_stop:
                    st.session_state.cont_anim_running = False
                    return

                h_shifted = np.interp(t_x, t_h + t_k, h_t, left=0, right=0)
                product = x_t * h_shifted

                fig, axs = plt.subplots(3, 1, figsize=(10, 7))
                fig.suptitle(f"t = {t_k:.2f} — Paso {k+1}/{len(t_conv)}", fontsize=14, fontweight="bold")

                axs[0].plot(t_x, x_t, label="x(t)", linewidth=2)
                axs[0].plot(t_x, h_shifted, label="h(t) desplazada", color="C1", linewidth=2)
                axs[0].legend(); axs[0].grid(alpha=0.4)

                axs[1].plot(t_x, product, color="C2")
                axs[1].set_title("Producto x(t)·h(t desplazada)")
                axs[1].grid(alpha=0.4)

                axs[2].plot(t_conv[:k+1], y_conv[:k+1], color="purple")
                axs[2].set_title("y(t) — construcción progresiva")
                axs[2].grid(alpha=0.4)

                fig.tight_layout(rect=[0, 0.03, 1, 0.95])
                visor.pyplot(fig, clear_figure=True)
                plt.close(fig)
                time.sleep(FRAME_DELAY)

            # Resultado final
            fig_final, ax_final = plt.subplots(1, 1, figsize=(10, 3))
            ax_final.plot(t_conv, y_conv, color="purple", linewidth=2)
            ax_final.set_title("y(t) — Convolución completa")
            ax_final.grid(alpha=0.4)
            fig_final.tight_layout()
            visor.pyplot(fig_final, clear_figure=True)
            plt.close(fig_final)

            st.session_state.cont_anim_running = False

        if start_btn and not st.session_state.cont_anim_running:
            animar_convolucion()



    # ============================================================
    # SEÑALES DISCRETAS (caso B corregido)
    # ============================================================
    elif tipo_senal == "Señales discretas":
        modo_disc = st.sidebar.selectbox(
            "Selecciona lo que deseas visualizar",
            ("Funciones individuales", "Convoluciones (animadas)")
        )

        # ===============================
        # INCISO A
        # ===============================
        na = np.arange(-6, 6, 1)
        nha = np.arange(-5, 5, 1)
        xna = 6 - np.abs(na)
        hna = np.ones(len(nha))

        # ===============================
        # INCISO B
        # ===============================
        nxb = np.arange(-3, 8, 1)
        nhb = np.arange(0, 10, 1)

        xnb = np.ones(len(nxb))
        hnb = (6/7)**nhb
        # ===============================
        # FUNCIONES INDIVIDUALES
        # ===============================
        if modo_disc == "Funciones individuales":
            inciso = st.selectbox("Selecciona el inciso", ("A", "B"))
            opcion = st.selectbox("Selecciona la señal", ("x[n]", "h[n]"))
            fig, ax = plt.subplots(figsize=(8, 4))

            if inciso == "A":
                if opcion == "x[n]":
                    ax.stem(na, xna, basefmt=" ")
                    ax.set_title("Inciso A — x[n] = 6 - |n|")
                else:
                    ax.stem(nha, hna, basefmt=" ")
                    ax.set_title("Inciso A — h[n] = 1")
            else:
                if opcion == "x[n]":
                    ax.stem(nxb, xnb, basefmt=" ")
                    ax.set_title("Inciso B — x[n] = u[n+3] - u[n-7]")
                else:
                    ax.stem(nhb, hnb, basefmt=" ")
                    ax.set_title("Inciso B — h[n] = (6/7)^n · (u[n] - u[n−9])")

            ax.set_xlabel("n")
            ax.set_ylabel("Amplitud")
            ax.grid(True)
            st.pyplot(fig)

        # ===============================
        # ANIMACIONES DE CONVOLUCIÓN
        # ===============================
        elif modo_disc == "Convoluciones (animadas)":
            inciso = st.selectbox("Selecciona el inciso", ("A", "B"))

            if inciso == "A":
                st.subheader("🔄 Animación de convolución discreta — Inciso A")

                ya_py = np.convolve(hna, xna)
                na_con = np.arange(na[0] + nha[0], na[-1] + nha[-1] + 1)

                fig, axs = plt.subplots(4, 1, figsize=(8, 10))
                placeholder = st.empty()
                y = []

                h_flip = hna[::-1]
                n_flip = -nha[::-1]

                for n0 in na_con:
                    h_shift = np.zeros_like(na, dtype=float)
                    for i, ni in enumerate(n_flip):
                        n_pos = n0 - ni
                        if n_pos in na:
                            idx = np.where(na == n_pos)[0][0]
                            h_shift[idx] = h_flip[i]

                    producto = xna * h_shift
                    y_val = np.sum(producto)
                    y.append(y_val)

                    axs[0].cla(); axs[1].cla(); axs[2].cla(); axs[3].cla()
                    axs[0].stem(na, xna, basefmt=" ")
                    axs[1].stem(na, h_shift, basefmt=" ", linefmt="r-", markerfmt="ro")
                    axs[2].stem(na, producto, basefmt=" ", linefmt="orange", markerfmt="o")
                    axs[3].stem(na_con[:len(y)], y, basefmt=" ", linefmt="purple", markerfmt="o")

                    axs[0].set_title("x[n]")
                    axs[1].set_title(f"h[n−{n0}] desplazada e invertida")
                    axs[2].set_title("Producto x[n]·h[n−k]")
                    axs[3].set_title("Construcción progresiva de y[n]")

                    for ax in axs:
                        ax.set_xlabel("n")
                        ax.set_ylabel("Amplitud")
                        ax.grid(True)

                    plt.tight_layout()
                    placeholder.pyplot(fig)
                    time.sleep(0.1)

                fig_final, axf = plt.subplots(figsize=(8, 4))
                axf.stem(na_con, ya_py, basefmt=" ")
                axf.set_title("Convolución discreta final y[n] — Inciso A")
                axf.set_xlabel("n")
                axf.set_ylabel("y[n]")
                axf.grid(True)
                st.pyplot(fig_final)

                st.success("✅ Animación discreta (Inciso A) completada correctamente.")

            # ===============================
            # INCISO B — ANIMACIÓN
            # ===============================
            else:
                st.subheader("🔄 Animación de convolución discreta — Inciso B")

                # Definición de señales
                nxb = np.arange(-3, 8, 1)
                nhb = np.arange(0, 10, 1)
                xnb = np.ones(len(nxb))
                hnb = (6/7)**nhb

                # Convolución con numpy
                yb_py = np.convolve(xnb, hnb)
                nb_con = np.arange(nxb[0] + nhb[0], nxb[-1] + nhb[-1] + 1)  # (-3 a 16)

                # Crear figuras
                fig, axs = plt.subplots(4, 1, figsize=(8, 10))
                placeholder = st.empty()
                y = []

                # Inversión e índices
                h_flip = hnb[::-1]
                n_flip = -nhb[::-1]

                for n0 in nb_con:
                    # Extender rango para evitar recortes
                    n_ext = np.arange(nxb[0] - len(h_flip), nxb[-1] + len(h_flip))
                    h_shift = np.zeros_like(n_ext, dtype=float)
                    x_ext = np.zeros_like(n_ext, dtype=float)

                    # Ubicar x[n] dentro del rango extendido
                    x_ext[(n_ext >= nxb[0]) & (n_ext <= nxb[-1])] = xnb

                    # Desplazar h[n] invertida
                    for i, ni in enumerate(n_flip):
                        n_pos = n0 - ni
                        if n_pos in n_ext:
                            idx = np.where(n_ext == n_pos)[0][0]
                            h_shift[idx] = h_flip[i]

                    producto = x_ext * h_shift
                    y_val = np.sum(producto)
                    y.append(y_val)

                    # Limpiar y actualizar gráficas
                    for ax in axs:
                        ax.cla()

                    axs[0].stem(n_ext, x_ext, basefmt=" ")
                    axs[1].stem(n_ext, h_shift, basefmt=" ", linefmt="r-", markerfmt="ro")
                    axs[2].stem(n_ext, producto, basefmt=" ", linefmt="orange", markerfmt="o")
                    axs[3].stem(nb_con[:len(y)], y, basefmt=" ", linefmt="purple", markerfmt="o")

                    axs[0].set_title("x[n]")
                    axs[1].set_title(f"h[n−{n0}] desplazada e invertida")
                    axs[2].set_title("Producto x[n]·h[n−k]")
                    axs[3].set_title("Construcción progresiva de y[n]")

                    for ax in axs:
                        ax.set_xlabel("n")
                        ax.set_ylabel("Amplitud")
                        ax.grid(True)

                    plt.tight_layout()
                    placeholder.pyplot(fig)
                    time.sleep(0.08)

                # Resultado final
                fig_final, axf = plt.subplots(figsize=(8, 4))
                axf.stem(nb_con, yb_py, basefmt=" ")
                axf.set_title("Convolución discreta final y[n] — Inciso B (corregido)")
                axf.set_xlabel("n")
                axf.set_ylabel("y[n]")
                axf.grid(True)
                st.pyplot(fig_final)


# ================================================================
# PUNTO 2: COMPARACIÓN CONVOLUCIONES CONTINUAS
# ================================================================
elif menu == "Punto 2":

    st.markdown("<h2 style='text-align:center;'>Punto 2 — Comparación de Convoluciones Continuas</h2>", unsafe_allow_html=True)

    delta = 0.05

    def u(t):
        return np.where(t >= 0, 1, 0)

    # --- Selección de caso ---
    caso = st.selectbox("Selecciona el caso", ["a", "b", "c"], index=0)
    visor = st.empty()

    # =============== CASO A ===============
    if caso == "a":
        t_x = np.arange(-1, 5 + delta, delta)
        t_h = np.arange(0, 6 + delta, delta)

        x_t = np.exp(-4 * t_x / 5) * (u(t_x + 1) - u(t_x - 5))
        h_t = np.exp(-t_h / 4) * u(t_h)

        # Convolución con numpy
        y_tpy = np.convolve(x_t, h_t) * delta
        len_y = len(t_x) + len(t_h) - 1
        t_y = np.arange(t_x[0] + t_h[0], t_x[0] + t_h[0] + len_y * delta, delta)

        # Convolución manual
        t_y1 = np.arange(-5, -1, delta)
        t_y2 = np.arange(-1, 5, delta)
        t_y3 = np.arange(5, 20, delta)
        y_t1 = np.zeros_like(t_y1)
        y_t2 = (20/11) * np.exp(-t_y2/4) * (np.exp(11/20) - np.exp(-11*t_y2/20))
        y_t3 = (20/11) * np.exp(-t_y3/4) * (np.exp(11/20) - np.exp(-11/4))
        y_tm = np.concatenate((y_t1, y_t2, y_t3))
        t_m = np.concatenate((t_y1, t_y2, t_y3))

    # =============== CASO B ===============
    elif caso == "b":
        t_b = np.arange(-1, 6 + delta, delta)
        t_hb = np.arange(-4, 4 + delta, delta)

        h_t = np.exp(-0.5 * t_b) * u(t_b + 1)
        x_t = np.exp(0.5 * t_hb) * (u(t_hb + 4) - u(t_hb)) + np.exp(-0.5 * t_hb) * (u(t_hb) - u(t_hb - 4))

        y_tpy = np.convolve(x_t, h_t) * delta
        len_yb = len(x_t) + len(h_t) - 1
        t_y = np.arange(t_hb[0] + t_b[0], t_hb[0] + t_b[0] + len_yb * delta, delta)

        # Manual
        tm1_b = np.arange(-6, -5, delta)
        tm2_b = np.arange(-5, -1, delta)
        tm3_b = np.arange(-1, 3, delta)
        tm4_b = np.arange(3, 10, delta)

        ym1_b = np.zeros_like(tm1_b)
        ym2_b = np.exp(1 + tm2_b/2) - np.exp(-4 - tm2_b/2)
        ym3_b = np.exp(-tm3_b/2) * (tm3_b + 2 - np.exp(-4))
        ym4_b = (5 - np.exp(-4)) * np.exp(-tm4_b/2)

        y_tm = np.concatenate((ym1_b, ym2_b, ym3_b, ym4_b))
        t_m = np.concatenate((tm1_b, tm2_b, tm3_b, tm4_b))

    # =============== CASO C ===============
    elif caso == "c":
        t_c = np.arange(-6, 1 + delta, delta)
        t_ch = np.arange(-1, 4 + delta, delta)

        h_t = np.exp(t_c) * u(1 - t_c)
        x_t = u(t_ch + 1) - u(t_ch - 4)

        y_tpy = np.convolve(x_t, h_t) * delta
        len_yc = len(x_t) + len(h_t) - 1
        t_y = np.arange(t_ch[0] + t_c[0], t_ch[0] + t_c[0] + len_yc * delta, delta)

        t1_c = np.arange(-6, 0, delta)
        t2_c = np.arange(0, 5, delta)
        t3_c = np.arange(5, 10, delta)

        y1_c = np.exp(t1_c + 1) - np.exp(t1_c - 4)
        y2_c = np.exp(1) - np.exp(t2_c - 4)
        y3_c = np.zeros_like(t3_c)

        y_tm = np.concatenate((y1_c, y2_c, y3_c))
        t_m = np.concatenate((t1_c, t2_c, t3_c))

    # --- GRAFICAR (una debajo de otra) ---
    fig, axs = plt.subplots(4, 1, figsize=(9, 12))
    fig.suptitle(f"Comparación de Convoluciones — Caso {caso.upper()}", fontsize=16, fontweight="bold")

    # Señales originales
    axs[0].plot(t_x if caso=="a" else t_hb if caso=="b" else t_ch, x_t, label='x(t)')
    axs[0].plot(t_h if caso=="a" else t_b if caso=="b" else t_c, h_t, label='h(t)')
    axs[0].set_title("Señales originales")
    axs[0].legend(); axs[0].grid(True)

    # Convolución con Python
    axs[1].plot(t_y, y_tpy, 'r', label='np.convolve')
    axs[1].set_title("Convolución con np.convolve")
    axs[1].legend(); axs[1].grid(True)

    # Convolución manual
    axs[2].plot(t_m, y_tm, 'g', label='Manual')
    axs[2].set_title("Convolución manual")
    axs[2].legend(); axs[2].grid(True)

    # Comparación
    axs[3].plot(t_y, y_tpy, 'r', label='Python')
    axs[3].plot(t_m, y_tm, 'g--', label='Manual')
    axs[3].set_title("Comparación manual vs np.convolve")
    axs[3].legend(); axs[3].grid(True)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    visor.pyplot(fig, clear_figure=True)