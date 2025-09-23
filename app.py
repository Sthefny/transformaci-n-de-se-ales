import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Laboratorio de Señales",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------
# Menú lateral con título fijo
# -------------------
st.sidebar.title("📘 Laboratorio Nº 1")

menu = st.sidebar.selectbox(
    "Selecciona el punto",
    ["Inicio", "Punto 1", "Punto 2", "Punto 3", "Punto 4"]
)

# ================================================================
# PUNTO 1: SEÑALES ORIGINALES
# ================================================================
if menu == "Inicio":
    st.markdown(
        "<h1 style='text-align: center;'>Transformación de señales, Señales y Sistemas</h1>",
        unsafe_allow_html=True
    )
    st.write("Bienvenida/o 🙌. Usa el menú de la izquierda para navegar entre los puntos del laboratorio.")
    
    st.subheader("Descripción general")
    st.write(
        "El laboratorio tiene como fin poner en práctica los conceptos teóricos adquiridos en el curso de Señales "
        "y Sistemas sobre las operaciones básicas de transformación de señales y trasladarlas a un escenario "
        "gráfico a nivel computacional. Las operaciones que se llevan a cabo en este trabajo son el desplazamiento "
        "y el escalamiento en el tiempo, junto con su equivalente en lo denominado diezmado e interpolación."
    )

elif menu == "Punto 1":
    st.header("Punto 1: Señales originales")
    
    Señal = st.sidebar.selectbox(
        "Seleccione qué señal quiere visualizar",
        ["Señal continua 1", "Señal continua 2", "Secuencia discreta 1", "Secuencia discreta 2"]
    ) 

    fs = 100
    delt = 1/fs

    # --- Señal continua 1 ---
    p1, p2, p3, p4, p5 = -2, -1, 1, 3, 4
    t1a = np.arange(p1,p2,delt)
    t1b = np.arange(p2,p3,delt)
    t1c = np.arange(p3,p4,delt)
    t1d = np.arange(p4,p5+delt,delt)
    x1a = 2*t1a+4
    x1b = 2*np.ones(len(t1b))
    x1c = 3*np.ones(len(t1c))
    x1d = -3*t1d+12
    t1 = np.concatenate((t1a,t1b,t1c,t1d))
    x1 = np.concatenate((x1a,x1b,x1c,x1d))

    # --- Señal continua 2 ---
    p11, p22, p33, p44, p55 = -3, -2, 0, 2, 3
    t2a = np.arange(p11,p22,delt)
    t2b = np.arange(p22,p33,delt)
    t2c = np.arange(p33,p44,delt)
    t2d = np.arange(p44,p55+delt,delt)
    x2a = t2a+3
    x2b = (t2b/2)+3
    x2c = -t2c+3
    x2d = np.ones(len(t2d))
    t2 = np.concatenate((t2a,t2b,t2c,t2d))
    x2 = np.concatenate((x2a,x2b,x2c,x2d))

    # --- Secuencia discreta 1 ---
    n_in1, n_fin1 = -5, 16
    n1 = np.arange(n_in1, n_fin1+1)
    xn1 = [0,0,0,0,0,-4,0,3,5,2,-3,-1,3,6,8,3,-1,0,0,0,0,0]

    # --- Secuencia discreta 2 ---
    n_in2, n_fin2 = -10, 10
    n2 = np.arange(n_in2, n_fin2+1)
    xn2 = np.zeros(len(n2), dtype=float)
    for i in n2:
        k = i - n_in2
        if -10 <= i <= -6:
            xn2[k] = 0
        elif -5 <= i <= 0:
            xn2[k] = (3/4)**i
        elif 1 <= i <= 5:
            xn2[k] = (7/4)**i
        elif 6 <= i <= 10:
            xn2[k] = 0
        else:
            xn2[k] = 0

    # --- Mostrar la señal seleccionada ---
    fig, ax = plt.subplots(figsize=(6,4))
    if Señal == "Señal continua 1":
        ax.plot(t1, x1, color="blue")
        ax.set_title("Señal continua 1")
    elif Señal == "Señal continua 2":
        ax.plot(t2, x2, color="green")
        ax.set_title("Señal continua 2")
    elif Señal == "Secuencia discreta 1":
        ax.stem(n1, xn1, linefmt="b-", markerfmt="bo", basefmt="r-")
        ax.set_title("Secuencia discreta 1")
    elif Señal == "Secuencia discreta 2":
        ax.stem(n2, xn2, linefmt="g-", markerfmt="go", basefmt="r-")
        ax.set_title("Secuencia discreta 2")
    ax.grid(True)
    st.pyplot(fig)
    st.markdown("---")

# ================================================================
# PUNTO 2: TRANSFORMACIONES
# ================================================================
elif menu == "Punto 2":
    st.header("Punto 2: Transformaciones")

    Tipo = st.sidebar.selectbox(
        "Seleccione el tipo de señal",
        ["Dominio continuo", "Dominio discreto"]
    )

    # ----------------------------------------------------------
    # Señales continuas con transformaciones
    # ----------------------------------------------------------
    if Tipo == "Dominio continuo":
        fs = 100
        delt = 1/fs

        # Señal continua 1
        p1, p2, p3, p4, p5 = -2, -1, 1, 3, 4
        t1a = np.arange(p1, p2, delt)
        t1b = np.arange(p2, p3, delt)
        t1c = np.arange(p3, p4, delt)
        t1d = np.arange(p4, p5+delt, delt)
        x1a = 2*t1a+4
        x1b = 2*np.ones(len(t1b))
        x1c = 3*np.ones(len(t1c))
        x1d = -3*t1d+12
        t1 = np.concatenate((t1a, t1b, t1c, t1d))
        x1 = np.concatenate((x1a, x1b, x1c, x1d))

        # Señal continua 2
        p1, p2, p3, p4, p5 = -3, -2, 0, 2, 3
        t2a = np.arange(p1, p2, delt)
        t2b = np.arange(p2, p3, delt)
        t2c = np.arange(p3, p4, delt)
        t2d = np.arange(p4, p5+delt, delt)
        x2a = t2a+3
        x2b = (t2b/2)+3
        x2c = -t2c+3
        x2d = np.ones(len(t2d))
        t2 = np.concatenate((t2a, t2b, t2c, t2d))
        x2 = np.concatenate((x2a, x2b, x2c, x2d))

        # Definición de métodos
        def metodo1(t, retraso, escalamiento):
            t_desplazado = t + retraso
            return t_desplazado / escalamiento

        def metodo2(t, retraso, escalamiento):
            t_escalado = t / escalamiento
            return t_escalado + (retraso / escalamiento)

        st.subheader("Transformación de señales continuas")

        # Selección
        op1 = st.sidebar.selectbox("Seleccione la función continua", ["señal continua 1","señal continua 2"])
        metodo = st.sidebar.selectbox(
            "Seleccione el método",
            [1, 2],
            format_func=lambda x: "Metodo #1" if x == 1 else "Metodo #2"
        )
        t0 = st.sidebar.number_input("Valor de retraso (t0)", value=0.0, step=0.5)
        esc = st.sidebar.number_input("Valor de escalamiento (a)", value=1.0, step=0.5)

        # Señal base
        if op1 == 1:
            t_base, x_base = t1, x1
        else:
            t_base, x_base = t2, x2

        # Aplicar
        if metodo == 1:
            t_trans = metodo1(t_base, t0, esc)
        else:
            t_trans = metodo2(t_base, t0, esc)

        # Gráfica
                # Señal base
        if op1 == 1:
            t_base, x_base = t1, x1
        else:
            t_base, x_base = t2, x2

        # Gráfica 1: Señal original
        fig1, ax1 = plt.subplots(figsize=(6,4))
        ax1.plot(t_base, x_base, color="blue")
        ax1.set_title("1. Señal original")
        ax1.grid(True)
        st.pyplot(fig1)
        st.markdown("---")

        # Método 1: desplazar → escalar
        if metodo == 1:
            # Paso 1: Desplazamiento
            t_desplazada = t_base + t0
            fig2, ax2 = plt.subplots(figsize=(6,4))
            ax2.plot(t_desplazada, x_base, color="green")
            ax2.set_title(f"2. Señal desplazada (t0={t0})")
            ax2.grid(True)
            st.pyplot(fig2)
            st.markdown("---")

            # Paso 2: Escalamiento
            t_escalada = t_desplazada / esc
            fig3, ax3 = plt.subplots(figsize=(6,4))
            ax3.plot(t_escalada, x_base, color="red")
            ax3.set_title(f"3. Señal escalada despues del desplazamiento (a={esc})")
            ax3.grid(True)
            st.pyplot(fig3)
            st.markdown("---")

        # Método 2: escalar → desplazar
        elif metodo == 2:
            # Paso 1: Escalamiento
            t_escalada = t_base / esc
            fig2, ax2 = plt.subplots(figsize=(6,4))
            ax2.plot(t_escalada, x_base, color="orange")
            ax2.set_title(f"2. Señal escalada (a={esc})")
            ax2.grid(True)
            st.pyplot(fig2)
            st.markdown("---")

            # Paso 2: Desplazamiento
            t_final = t_escalada + (t0 / esc)
            fig3, ax3 = plt.subplots(figsize=(6,4))
            ax3.plot(t_final, x_base, color="red")
            ax3.set_title(f"3. Señal desplazada después del escalamiento (t0={t0})")
            ax3.grid(True)
            st.pyplot(fig3)
            st.markdown("---")


        # ----------------------------------------------------------
        # Señales discretas con transformaciones
        # ----------------------------------------------------------
    elif Tipo == "Dominio discreto":

        st.subheader("Transformación de señales discretas")

        # --- Secuencia discreta 1 ---
        n_in1, n_fin1 = -5, 16
        n1 = np.arange(n_in1, n_fin1+1)
        xn1 = [0,0,0,0,0,-4,0,3,5,2,-3,-1,3,6,8,3,-1,0,0,0,0,0]

        # --- Secuencia discreta 2 ---
        n_in2, n_fin2 = -10, 10
        n2 = np.arange(n_in2, n_fin2+1)
        xn2 = np.zeros(len(n2), dtype=float)
        for i in n2:
            k = i - n_in2
            if -10 <= i <= -6:
                xn2[k] = 0
            elif -5 <= i <= 0:
                xn2[k] = (3/4)**i
            elif 1 <= i <= 5:
                xn2[k] = (7/4)**i
            elif 6 <= i <= 10:
                xn2[k] = 0
            else:
                xn2[k] = 0

        # Función de transformación discreta
        def transformar_discreta(n, x, t0, M, metodo):
            if metodo == 1:
                # Método 1: primero desplazamiento, luego escalamiento
                n_des = n + t0
                if M == 1:
                    return (n_des, x)
                elif M < -1 or M > 1:
                    D = int(abs(M))
                    return (n_des[::D], x[::D])
                else:  # -1 <= M < 1 → Interpolación
                    L = int(round(1.0 / abs(M)))
                    L_n = len(x)
                    N = L * (L_n - 1) + 1
                    nI = n_des[0] + np.arange(N) / L

                    xn_0 = np.zeros(N, dtype=float)
                    xn_0[::L] = x

                    xn_esc = xn_0.copy()
                    for i in range(1, N):
                        if xn_esc[i] == 0:
                            xn_esc[i] = xn_esc[i-1]

                    xn_lin = np.zeros(N, dtype=float)
                    k = 0
                    for i in range(L_n - 1):
                        xi = x[i]
                        dx = x[i+1] - xi
                        xn_lin[k] = xi
                        for j in range(1, L):
                            xn_lin[k + j] = xi + (j / L) * dx
                        k += L
                    xn_lin[-1] = x[-1]

                    if M < 0:
                        nI = -nI[::-1]
                        xn_0 = xn_0[::-1]
                        xn_esc = xn_esc[::-1]
                        xn_lin = xn_lin[::-1]

                    return (nI, xn_0, xn_esc, xn_lin)

            elif metodo == 2:
                # Método 2: primero escalamiento, luego desplazamiento
                des_escalado = t0 / M
                if M == 1:
                    return (n + int(des_escalado), x)
                elif M > 1 or M < -1:
                    D = int(abs(M))
                    n_diez = n[::D]
                    x_diez = x[::D]
                    return (n_diez + int(des_escalado), x_diez)
                else:  # -1 <= M < 1 → Interpolación
                    L = int(round(1.0 / abs(M)))
                    L_n = len(x)
                    N = L * (L_n - 1) + 1
                    nI_base = n[0] + np.arange(N) / L
                    nI = nI_base + des_escalado

                    xn_0 = np.zeros(N, dtype=float)
                    xn_0[::L] = x

                    xn_esc = xn_0.copy()
                    for i in range(1, N):
                        if xn_esc[i] == 0:
                            xn_esc[i] = xn_esc[i-1]

                    xn_lin = np.zeros(N, dtype=float)
                    k = 0
                    for i in range(L_n - 1):
                        xi = x[i]
                        dx = x[i+1] - xi
                        xn_lin[k] = xi
                        for j in range(1, L):
                            xn_lin[k + j] = xi + (j / L) * dx
                        k += L
                    xn_lin[-1] = x[-1]

                    if M < 0:
                        nI = -nI[::-1]
                        xn_0 = xn_0[::-1]
                        xn_esc = xn_esc[::-1]
                        xn_lin = xn_lin[::-1]

                    return (nI, xn_0, xn_esc, xn_lin)
            else:
                raise ValueError("Método no válido (use 1 o 2).")

        # ----------------------------
        # Interfaz gráfica con Streamlit
        # ----------------------------
        op = st.sidebar.selectbox("Seleccione la secuencia discreta", ["secuencia discreta 1", "secuencia discreta 2"])
        metodo = st.sidebar.selectbox("Seleccione el método", [1, 2], format_func=lambda x: f"Método #{x}")
        t0 = st.sidebar.number_input("Valor de retraso (t0)", value=0, step=1)
        M = st.sidebar.number_input("Valor de escalamiento (M)", value=1.0, step=0.5)

        if op == "secuencia discreta 1":
            n_base, x_base = n1, xn1
        else:
            n_base, x_base = n2, xn2

        # Procesar según el rango de M
        if M < -1 or M >= 1:
            n_out, x_out = transformar_discreta(n_base, x_base, t0, M, metodo)
            fig, ax = plt.subplots(figsize=(7,4))
            ax.stem(n_out, x_out)
            ax.set_title(f'Secuencia {op[-1]} (método {metodo}) — t0={t0}, M={M}')
            ax.grid(True)
            st.pyplot(fig)

        elif -1 <= M < 1:
            nI, x0, xesc, xlin = transformar_discreta(n_base, x_base, t0, M, metodo)
            fig, axs = plt.subplots(3, 1, figsize=(7, 10))
            axs[0].stem(nI, x0);   axs[0].set_title('Interpolación por ceros');   axs[0].grid(True)
            axs[1].stem(nI, xesc); axs[1].set_title('Interpolación por escalón'); axs[1].grid(True)
            axs[2].stem(nI, xlin); axs[2].set_title('Interpolación lineal');      axs[2].grid(True)
            plt.tight_layout()
            st.pyplot(fig)



# ================================================================
# PUNTO 4: CARGA DE ARCHIVOS
# ================================================================
elif menu == "Punto 4":
    st.header("Punto 4: Señales muestreadas y sobre muestreo")

    # Subir archivos .txt
    st.sidebar.subheader("Carga de señales")
    file1 = st.sidebar.file_uploader("Subir señal muestreada a 2 kHz", type=["txt"])
    file2 = st.sidebar.file_uploader("Subir señal muestreada a 2.2 kHz", type=["txt"])

    if file1 is not None and file2 is not None:
        # Cargar los datos de los archivos
        data1 = np.loadtxt(file1)
        data2 = np.loadtxt(file2)

        # Ejes de tiempo (asumiendo que los .txt tienen solo los valores de amplitud)
        fs1 = 2000   # Hz
        fs2 = 2200   # Hz
        t1 = np.arange(len(data1)) / fs1
        t2 = np.arange(len(data2)) / fs2

        # Mostrar gráficas originales
        fig, ax = plt.subplots(2, 1, figsize=(7, 5))
        ax[0].plot(t1, data1, label="Señal 1 (2 kHz)", color="blue")
        ax[0].set_title("Señal muestreada a 2 kHz")
        ax[0].grid(True)

        ax[1].plot(t2, data2, label="Señal 2 (2.2 kHz)", color="green")
        ax[1].set_title("Señal muestreada a 2.2 kHz")
        ax[1].grid(True)

        plt.tight_layout()
        st.pyplot(fig)
        st.markdown("---")

    else:
        st.info("Por favor sube los dos archivos .txt para continuar.")


# ================================================================
# PUNTO 3: Retardo, escalamiento y suma de señales
# ================================================================
elif menu == "Punto 3":
    st.header("Punto 3: Retardo, escalamiento y suma de señales")

    # Selector para elegir cuál señal usar
    caso = st.sidebar.selectbox(
        "Seleccione el caso de suma",
        ["Señal continua con x1(t)", "Señal continua con x2(t)"]
    )

    fs = 100
    delt = 1/fs

    # =====================================================
    # CASO A: Transformaciones y suma con señal x1(t)
    # =====================================================
    if caso == "Señal continua con x1(t)":
        # Señal continua original (x1)
        p1, p2, p3, p4, p5 = -2, -1, 1, 3, 4
        t1a = np.arange(p1, p2, delt)
        t1b = np.arange(p2, p3, delt)
        t1c = np.arange(p3, p4, delt)
        t1d = np.arange(p4, p5+delt, delt)
        x1a = 2*t1a+4
        x1b = 2*np.ones(len(t1b))
        x1c = 3*np.ones(len(t1c))
        x1d = -3*t1d+12
        t1 = np.concatenate((t1a, t1b, t1c, t1d))
        x1 = np.concatenate((x1a, x1b, x1c, x1d))

        # Gráfica original
        fig0, ax0 = plt.subplots(figsize=(6,4))
        ax0.plot(t1, x1, color="black")
        ax0.set_title("Señal original x1(t)")
        ax0.grid(True)
        st.pyplot(fig0)

        # Retardo
        t0, t0a = -2, -1
        # Crear ejes retardados
        p11, p21, p31, p41, p51 = -2+t0, -1+t0, 1+t0, 3+t0, 4+t0
        t11 = np.arange(p11, p21, delt)
        t12 = np.arange(p21, p31, delt)
        t13 = np.arange(p31, p41, delt)
        t14 = np.arange(p41, p51+delt, delt)
        tr1 = np.concatenate((t11, t12, t13, t14))

        p12, p22, p32, p42, p52 = -2+t0a, -1+t0a, 1+t0a, 3+t0a, 4+t0a
        t21 = np.arange(p12, p22, delt)
        t22 = np.arange(p22, p32, delt)
        t23 = np.arange(p32, p42, delt)
        t24 = np.arange(p42, p52+delt, delt)
        tr2 = np.concatenate((t21, t22, t23, t24))

        # Escalamiento
        esc1, esc2 = 3, -4
        tesc1 = tr1 * esc1
        tesc2 = tr2 * esc2

        # Gráfica escalada 1
        fig1, ax1 = plt.subplots(figsize=(6,4))
        ax1.plot(tesc1, x1, color="blue")
        ax1.set_title("x1(t/3 + 2)")
        ax1.grid(True)
        st.pyplot(fig1)

        # Gráfica escalada 2
        fig2, ax2 = plt.subplots(figsize=(6,4))
        ax2.plot(tesc2, x1, color="green")
        ax2.set_title("x1(1 - t/4)")
        ax2.grid(True)
        st.pyplot(fig2)

        # Suma
        t_min, t_max = min(tesc1.min(), tesc2.min()), max(tesc1.max(), tesc2.max())
        t_sum = np.arange(t_min, t_max+delt, delt)
        s1 = t_sum/esc1 - t0
        s2 = t_sum/esc2 - t0a
        x11_interp = np.interp(s1, t1, x1)
        x21_interp = np.interp(s2, t1, x1)
        x_sum = x11_interp + x21_interp

        fig3, ax3 = plt.subplots(figsize=(12,8))
        ax3.plot(t_sum, x_sum, color="red")
        ax3.set_title("Suma: x1(t/3 + 2) + x1(1 - t/4)")
        ax3.grid(True)
        st.pyplot(fig3)

    # =====================================================
    # CASO B: Transformaciones y suma con señal x2(t)
    # =====================================================
    elif caso == "Señal continua con x2(t)":
        delt = 0.01  # paso base

        # --- Señal continua 2 (original) ---
        p11, p22, p33, p44, p55 = -3, -2, 0, 2, 3
        t2a = np.arange(p11, p22, delt)
        t2b = np.arange(p22, p33, delt)
        t2c = np.arange(p33, p44, delt)
        t2d = np.arange(p44, p55 + delt, delt)

        x2a = t2a + 3
        x2b = (t2b / 2) + 3
        x2c = -t2c + 3
        x2d = np.ones(len(t2d))

        t2 = np.concatenate((t2a, t2b, t2c, t2d))
        x2 = np.concatenate((x2a, x2b, x2c, x2d))

        # Definimos x(t) como función
        def x_func(t):
            return np.interp(t, t2, x2, left=0, right=0)

        # Rango amplio de tiempo
        t = np.arange(-20, 20, delt/5)

        # Transformaciones
        x1 = x_func(t/3 + 2)       # x(t/3 + 2)
        x2_tr = x_func(1 - t/4)    # x(1 - t/4)
        x_sum = x1 + x2_tr

        # Gráficas
        fig1, ax1 = plt.subplots(figsize=(8,4))
        ax1.plot(t2, x2, 'b', linewidth=2, label='$x(t)$')
        ax1.set_title("Señal original $x(t)$")
        ax1.grid(True); ax1.legend()
        st.pyplot(fig1)

        fig2, ax2 = plt.subplots(figsize=(8,4))
        ax2.plot(t, x1, 'r', linewidth=2, label=r'$x\left(\frac{t}{3} + 2\right)$')
        ax2.set_title("Transformación 1")
        ax2.grid(True); ax2.legend()
        st.pyplot(fig2)

        fig3, ax3 = plt.subplots(figsize=(8,4))
        ax3.plot(t, x2_tr, 'g', linewidth=2, label=r'$x\left(1 - \frac{t}{4}\right)$')
        ax3.set_title("Transformación 2")
        ax3.grid(True); ax3.legend()
        st.pyplot(fig3)

        fig4, ax4 = plt.subplots(figsize=(10,5))
        ax4.plot(t, x_sum, 'm', linewidth=2, label="Suma final")
        ax4.set_title("Suma de las transformaciones")
        ax4.grid(True); ax4.legend()
        st.pyplot(fig4)



# ================================================================
# INTEGRANTES
# ================================================================
st.sidebar.markdown("---")  
st.sidebar.subheader("👩‍🎓 Integrantes")
st.sidebar.write("- Sthefany Morales\n- Alejandro Rovira\n- Sebastian Pupo")
