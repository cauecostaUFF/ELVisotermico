import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
# from matplotlib.widgets import Button
# from matplotlib.widgets import TextBox

st.session_state.visibility = "collapsed"
st.session_state.disabled = True
st.set_page_config(
    page_title="Equilíbrio líquido-vapor isotérmico",
    # page_icon="🧊",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Definindo modelos
modelos = {0: r"Margules 2 sufixos",1: r"Margules 3 sufixos",2: r"NRTL"}
def format_func(option):
    return modelos[option]

# Informações
extra, main_col1,main_col2,main_col3 = st.columns([0.5,10,12,8])
with main_col2:
    st.write("")
    st.write("")
    st.write("")
    st.write(r"Pressões de saturação:")

with main_col1:
    # Selecionando modelo
    st.write(r"Selecione o modelo:")
    # st.write(r"Modelo de Margules: G$^{E}$ = n$\beta$RTx$_A$x$_B$")
    option = st.selectbox(
        "Modelo de Margules",
        options=list(modelos.keys()),format_func=format_func,
        label_visibility=st.session_state.visibility,
        disabled=False,
    )


# Entrada de dados
# valores = [1.3,2.8,2.2,2.2,2.2,1]
valores = [0.2,1,2.7,1.3,2,1.,3]
extra,col4, col5,col1, col2, col3, col6 = st.columns(valores)
with col1:
    st.latex(r"P^{sat}_A = ")
    st.latex(r"P^{sat}_B = ")

with col2:
    PsatA = st.number_input("",0,1000, value=100,label_visibility=st.session_state.visibility)
    PsatB = st.number_input("",0,1000,value=50,label_visibility=st.session_state.visibility)


with col3:
    st.latex(r" kPa ")
    st.latex(r" kPa ")
if option == 0:
    with col4:
        st.latex(r"\beta=")
    with col5:
        beta = st.number_input(r"\beta=", step=0.05, label_visibility=st.session_state.visibility)
        param = np.array([beta])
elif option == 1:
    with col4:
        st.latex(r"\alpha=")
        st.latex(r"\beta=")
    with col5:
        alpha = st.number_input(r"\alpha=", step=0.05, label_visibility=st.session_state.visibility)
        beta = st.number_input(r"\beta=", step=0.05, label_visibility=st.session_state.visibility)
        param = np.array([alpha, beta])
elif option == 2:
    with col4:
        st.latex(r"\alpha=")
        st.latex(r"\tau_{A,B}=")
        st.latex(r"\tau_{B,A}=")

    with col5:
        alpha = st.number_input(r"\alpha=", min_value=0.0,max_value=1.0,value = 0.30,step=0.01, label_visibility=st.session_state.visibility)
        dg12 = st.number_input(r"dg12", step=0.05, label_visibility=st.session_state.visibility)
        dg21 = st.number_input(r"dg21=", step=0.05, label_visibility=st.session_state.visibility)
        param = np.array([alpha,dg12,dg21])

with col6:
    Graph = st.radio(
        "Escolha o gráfico",
        ('Pxy', 'x,y', 'Delta G'))


# Cálculos termodinâmicos:
def nrtl(alpha, tau, x):
    G = np.exp(-alpha * tau)
    ncomp = x.shape[0]
    gamma = np.zeros_like(x)

    for i in range(ncomp):
        summ = 0
        for j in range(ncomp):
            summ += x[j] * G[i, j] / np.sum(G[:, j] * x) * (tau[i, j] -
                                                            (np.sum(x * tau[:, j] * G[:, j]) / np.sum(G[:, j] * x)))
        gamma[i] = np.sum(tau[:, i] * G[:, i] * x) / np.sum(G[:, i] * x) + summ

    return np.exp(gamma)

def Calcular(param,Psat = np.array([100,50])):
    x = np.linspace(1e-8,1-1e-8,101)
    if option == 0:
        beta = param[0]
        gamma = np.array([np.exp(beta*(1-x)**2), np.exp(beta*x**2)])
    elif option == 1:
        alpha = param[0]
        beta = param[1]
        gamma = np.array([np.exp((alpha + 2 * (beta - alpha) * x) * (1 - x) ** 2),
                          np.exp((beta + 2 * (alpha - beta) * (1 - x)) * x ** 2)])
    elif option == 2:
        alpha = param[0]
        tau = np.array([[0, param[1]],[param[2],0]])
        gamma = np.array([np.zeros_like(x), np.zeros_like(x)])
        for i in range(0, len(x)):
            xp = np.array([x[i], 1 - x[i]])
            gamma[:, i] = nrtl(alpha, tau, xp)
    P = x*gamma[0]*Psat[0] + (1-x)*gamma[1]*Psat[1]
    y = x*gamma[0]*Psat[0]/P

    G = x * np.log(x) + (1 - x) * np.log(1 - x) + x*np.log(gamma[0]) + (1-x)*np.log(gamma[1])
    # G = x * np.log(x) + (1 - x) * np.log(1 - x) + beta*x*(1 - x)
    return x,y,P,G

x,y,P,G = Calcular(param,np.array([PsatA,PsatB]))

# Definindo labels
xlabel = "x$_A$,y$_A$"
ylabel = "P (kPa)"

fig, ax = plt.subplots()
# plt.gca()
# plt.figtext(0.68,0.90, "Modelo de Margules: ", va="top", ha="left")

plt.subplots_adjust(top=0.75,bottom=0.25,left=0.15)
if Graph == 'Pxy':
    lz, = ax.plot([0, 1], [P[0], P[-1]], '--k', lw=1)
    lx, = ax.plot(x, P, lw=2)
    ly, = ax.plot(y, P, lw=2)

    xmin,xmax,ymin,ymax = ax.axis()
    ax.axis([0,1, ymin,ymax])
    # Ajustando a legenda
    ax.set_xlabel(xlabel,fontsize=12)
    ax.set_ylabel(ylabel,fontsize=12)
elif Graph == 'x,y':
    lz, = ax.plot([0, 1], [0, 1], '--k', lw=1)
    lx, = ax.plot(x, y, lw=2)
    xmin, xmax, ymin, ymax = ax.axis()
    ax.axis([0, 1, ymin, ymax])
    ax.set_xlabel("x$_A$", fontsize=12)
    ax.set_ylabel("y$_A$", fontsize=12)
elif Graph == 'Delta G':
    lz, = ax.plot([0, 1], [0, 0], '--k', lw=1)
    lx, = ax.plot(x, G, lw=2)
    xmin, xmax, ymin, ymax = ax.axis()
    ax.axis([0, 1, ymin, ymax])
    ax.set_xlabel("x$_A$", fontsize=12)
    ax.set_ylabel(r"$\frac{\Delta_{mist} G}{RT}$", fontsize=16)

plt.show()
st.pyplot(plt)
col_end1,col_end2,col_end3 = st.columns(3)
with col_end3:
    st.write("Desenvolvido por: Cauê Costa\nE-mail: cauecosta@id.uff.br")
with col_end2:
    st.image("UFF.png", width=150)
with col_end1:
    st.image("molmod.png", width=150)

