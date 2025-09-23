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
# PUNTO 4: CARGA DE ARCHIVOS Y SUMA DE SEÑALES
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

        # Frecuencias
        f1 = 2000
        f2 = 2200
        delta1 = 1 / f1
        delta2 = 1 / f2
        t1 = np.arange(len(data1)) * delta1
        t2 = np.arange(len(data2)) * delta2

        # Gráficas iniciales
        fig, ax = plt.subplots()
        ax.plot(t1, data1, label="Señal 1 (2 kHz)", color="blue")
        ax.set_title("Señal 1")
        ax.grid(True)
        st.pyplot(fig)

        fig, ax = plt.subplots()
        ax.plot(t2, data2, label="Señal 2 (2.2 kHz)", color="orange")
        ax.set_title("Señal 2")
        ax.grid(True)
        st.pyplot(fig)

        # Muestreo común
        fs = 22000  # mcm de 2000 y 2200
        deltas = 1 / fs
        min_time = min(t1[-1], t2[-1])
        t_com = np.arange(0, min_time, deltas)

        x1_n = np.interp(t_com, t1, data1)
        x2_n = np.interp(t_com, t2, data2)

        # Graficar señales re-muestreadas
        fig, ax = plt.subplots()
        ax.plot(t_com, x1_n, label="Señal 1 re-muestreada")
        ax.legend()
        st.pyplot(fig)

        fig, ax = plt.subplots()
        ax.plot(t_com, x2_n, label="Señal 2 re-muestreada", color="orange")
        ax.legend()
        st.pyplot(fig)

        # Suma de señales
        x_total = x1_n + x2_n
        fig, ax = plt.subplots()
        ax.plot(t_com, x_total, label="Señal total", color="green")
        ax.set_title("Suma de Señales")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

    else:
        st.info("Por favor sube los dos archivos .txt para continuar.")

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

        # ----------------------------------------------------------
        # Definición de funciones de transformación
        # ----------------------------------------------------------
        def metodo1(t, retraso, escalamiento):
            # Primero desplazamiento y luego escalamiento
            t_desplazado = t - retraso
            t_transformado = t_desplazado / escalamiento
            return t_desplazado, t_transformado

        def metodo2(t, retraso, escalamiento):
            # Primero escalamiento y luego desplazamiento
            t_escalado = t / escalamiento
            t_transformado = t_escalado - (retraso / escalamiento)
            return t_escalado, t_transformado

        # ----------------------------------------------------------
        # Configuración en la barra lateral
        # ----------------------------------------------------------
        st.sidebar.header("Configuración - Señales continuas")
        tipo = st.sidebar.selectbox("Seleccione tipo de señal", ["Señal continua 1", "Señal continua 2"])
        metodo = st.sidebar.selectbox("Método de transformación", ["Metodo 1", "Metodo 2"])
        retraso = st.sidebar.number_input("Valor de t0 (desplazamiento)", value=0.0, step=0.5)
        escala = st.sidebar.number_input("Valor de escalamiento (a)", value=1.0, step=0.5)

        # Selección de señal base
        if tipo == "Señal continua 1":
            t_base, x_base = t1, x1
        else:
            t_base, x_base = t2, x2

        # Aplicar método de transformación
        if metodo == ("Metodo 1"):
            # Desplazar primero
            t_desplazada, t_final = metodo1(t_base, retraso, escala)
            t_intermedia = t_desplazada  # desplazada
            t_siguiente = t_final        # después escalada
            titulos = [
                "1. Señal original",
                f"2. Señal desplazada (t0={retraso})",
                f"3. Señal escalada (a={escala})",
                "4. Señal final"
            ]
        else:
            # Escalar primero
            t_escalada, t_final = metodo2(t_base, retraso, escala)
            t_intermedia = t_escalada    # escalada
            t_siguiente = t_final        # después desplazada
            titulos = [
                "1. Señal original",
                f"2. Señal escalada (a={escala})",
                f"3. Señal desplazada (t0={retraso})",
                "4. Señal final"
            ]

        # ----------------------------------------------------------
        # Mostrar gráficas en orden
        # ----------------------------------------------------------
        fig, axs = plt.subplots(4, 1, figsize=(7, 12))

        axs[0].plot(t_base, x_base, color="blue")
        axs[0].set_title(titulos[0])

        axs[1].plot(t_intermedia, x_base, color="green")
        axs[1].set_title(titulos[1])

        axs[2].plot(t_siguiente, x_base, color="orange")
        axs[2].set_title(titulos[2])

        axs[3].plot(t_final, x_base, color="red")
        axs[3].set_title(titulos[3])

        for ax in axs: ax.grid(True)
        plt.tight_layout()
        st.pyplot(fig)



    # ----------------------------------------------------------
# Señales discretas con transformaciones
# ----------------------------------------------------------
    elif Tipo == "Dominio discreto":
    
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
# ====== FUNCIÓN ======
        def trans_discreta(n, x, t0, M, metodo):
            n_in = n[0]
            n_fin = n[-1]

            if metodo == 1:
                # ---- Método 1: desplazamiento → escalamiento ----
                n_des = n - t0

                if M == 1:
                    return (n_des, x)

                elif M == -1:
                    nI = -n_des[::-1]
                    x_mod = x[::-1]
                    return (nI, x_mod)

                elif abs(M) > 1:
                    # Escalamiento (submuestreo)
                    D = int(abs(M))
                    x_mod = x[::-D] if M < 0 else x[::D]

                    n_des_escalado = n_des / M
                    nI = n_des_escalado[::-D] if M < 0 else n_des_escalado[::D]

                    return (nI, x_mod)

                else:  # Interpolación: -1 < M < 1
                    L = int(round(1.0 / abs(M)))
                    L_n = len(x)
                    N = L * (L_n - 1) + 1

                    xn_0 = np.zeros(N, dtype=float)
                    xn_0[::L] = x
                    if M < 0 :
                        xn_0[::-L] = x

                    xn_esc = xn_0.copy()
                    for i in range(1, N):
                        if xn_esc[i] == 0:
                            xn_esc[i] = xn_esc[i - 1]

                    xn_lin = np.zeros(N, dtype=float)
                    k = 0
                    for i in range(L_n - 1):
                        xi = x[i]
                        dx = x[i + 1] - xi
                        xn_lin[k] = xi
                        for j in range(1, L):
                            xn_lin[k + j] = xi + (j / L) * dx
                        k += L
                    xn_lin[-1] = x[-1]

                    if M < 0 :
                        xn_lin[::-L] = x
                    nI = np.linspace(n_des[0] / M, n_des[-1] / M, N)


                    if M < 0:
                        nI = -nI[::-1]
                        xn_0 = xn_0[::-1]
                        xn_esc = xn_esc[::-1]
                        xn_lin = xn_lin[::-1]

                    return (nI, xn_0, xn_esc, xn_lin)

            elif metodo == 2:
                # ---- Método 2: escalamiento → desplazamiento ----
                des_escalado = t0 / M

                if M == 1:
                    nI = n - int(des_escalado)
                    return (nI, x)

                elif M == -1:
                    x_mod = x[::-1]
                    n_escalado = -n[::-1]
                    nI = n_escalado + int(des_escalado)
                    return (nI, x_mod)

                elif abs(M) > 1:
                    D = int(abs(M))
                    x_mod = x[::-D] if M < 0 else x[::D]

                    n_escalado = n / M
                    nI = n_escalado[::-D] if M < 0 else n_escalado[::D]
                    nI = nI - des_escalado  # Aplicar desplazamiento

                    return (nI, x_mod)

                else:  # Interpolación: -1 < M < 1
                    L = int(round(1.0 / abs(M)))
                    L_n = len(x)
                    N = L * (L_n - 1) + 1

                    xn_0 = np.zeros(N, dtype=float)
                    xn_0[::L] = x
                    if M < 0 :
                        xn_0[::-L] = x

                    xn_esc = xn_0.copy()
                    for i in range(1, N):
                        if xn_esc[i] == 0:
                            xn_esc[i] = xn_esc[i - 1]

                    xn_lin = np.zeros(N, dtype=float)
                    k = 0
                    for i in range(L_n - 1):
                        xi = x[i]
                        dx = x[i + 1] - xi
                        xn_lin[k] = xi
                        for j in range(1, L):
                            xn_lin[k + j] = xi + (j / L) * dx
                        k += L
                    xn_lin[-1] = x[-1]
                    if M < 0 : 
                        xn_lin [::-L] = x

                    # Tiempo escalado y desplazado
                    n_escalado = n / M
                    nI = np.linspace(n_escalado[0] - des_escalado, n_escalado[-1] - des_escalado, N)

                    if M < 0:
                        nI = -nI[::-1]
                        xn_0 = xn_0[::-1]
                        xn_esc = xn_esc[::-1]
                        xn_lin = xn_lin[::-1]

                    return (nI, xn_0, xn_esc, xn_lin)

            else:
                raise ValueError("Método no válido (use 1 o 2).")

        # =========== AQUÍ SÓLO CAMBIAMOS LAS ENTRADAS / SALIDAS ==========
        st.sidebar.header("Configuración - Señales discretas")
        op = st.sidebar.selectbox("Seleccione la secuencia discreta a transformar", [1, 2])
        metodo = st.sidebar.selectbox("Seleccione el método", [1, 2])
        t0 = st.sidebar.number_input("Ingrese el valor del desplazamiento (entero):", value=0, step=1)
        M = st.sidebar.number_input("Ingrese escalamiento:", value=1.0, step=0.5)

        if op == 1:
            if abs(M)>=1:
                n_out, x_out = trans_discreta(n1, xn1, t0, M, metodo)
                fig, ax = plt.subplots()
                ax.stem(n_out, x_out)
                ax.set_title(f'Secuencia 1 (método {metodo}) — t0={t0}, M={M}')
                ax.grid()
                st.pyplot(fig)
            elif  -1 < M < 1 : # --> INTERPOLACIONES
                nI, x0, xesc, xlin = trans_discreta(n1, xn1, t0, M, metodo)
                fig, axs = plt.subplots(3,1,figsize=(6,8))
                axs[0].stem(nI, x0);   axs[0].set_title('Interp. por ceros');  axs[0].grid()
                axs[1].stem(nI, xesc); axs[1].set_title('Interp. por escalón');axs[1].grid()
                axs[2].stem(nI, xlin); axs[2].set_title('Interp. lineal');     axs[2].grid()
                fig.suptitle(f'Secuencia 1 (método {metodo}) — t0={t0}, M={M}')
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.error("Error.")

        elif op == 2:
            if abs(M)>=1:
                n_out, x_out = trans_discreta(n2, xn2, t0, M, metodo)
                fig, ax = plt.subplots()
                ax.stem(n_out, x_out)
                ax.set_title(f'Secuencia 2 (método {metodo}) — t0={t0}, M={M}')
                ax.grid()
                st.pyplot(fig)
            elif -1 < M < 1:
                nI, x0, xesc, xlin = trans_discreta(n2, xn2, t0, M, metodo)
                fig, axs = plt.subplots(3,1,figsize=(6,8))
                axs[0].stem(nI, x0);   axs[0].set_title('Interp. por ceros');  axs[0].grid()
                axs[1].stem(nI, xesc); axs[1].set_title('Interp. por escalón');axs[1].grid()
                axs[2].stem(nI, xlin); axs[2].set_title('Interp. lineal');     axs[2].grid()
                fig.suptitle(f'Secuencia 2 (método {metodo}) — t0={t0}, M={M}')
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.error("Error.")

        else:
            st.error("Opción no válida")


# ================================================================
# INTEGRANTES
# ================================================================
st.sidebar.markdown("---")  
st.sidebar.subheader("👩‍🎓 Integrantes")
st.sidebar.write("- Sthefany Morales\n- Alejandro Rovira\n- Sebastian Pupo")
