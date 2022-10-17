import streamlit as st
import numpy as np
import matplotlib.pyplot as plt


class Dataset:
    def __init__(self,info):
        with open('Sistemas estimados/' +str(info) ,"r") as arq:
            cont = arq.readlines()
            for i in range(0 ,len(cont)):
                exec('self. ' +cont[i])


# Definindo os sistemas
sistemas ={-1:r"Manipulação de parâmetros",
0:r"Sistema etilbenzeno e heptano a 327.76K",
1:r"Sistema 1-hepteno e tolueno a 328.15K",
2:r"Sistema heptano e p-xileno a 313.11K",
3:r"Sistema benzeno e ciclohexano a 313.15K",
4:r"Sistema metilciclopentano e benzeno a 313.14K",
5:r"Sistema ciclohexano e tolueno a 298.15K",
6:r"Sistema hexafluorobenzeno e tolueno a 303.15K",
7:r"Sistema dissulfito de carbono e ciclohexano a 298.15K",
8:r"Sistema dissulfito de carbono e ciclopentano a 288.15K",
9:r"Sistema hexafluorobenzeno e p-xileno a 313.15K",
10:r"Sistema hexafluorobenzeno e ciclohexano a 303.15K",
11:r"Sistema tolueno e 4-metil-2-pentanona a 323.15K",
12:r"Sistema tolueno e 2-pentanona a 323.15K",
13:r"Sistema benzeno e tiofeno a 328.15K",
14:r"Sistema hexafluorobenzeno e diisopropil éter a 298.13K",
15:r"Sistema heptano e tiofeno a 328.15K",
16:r"Sistema heptano e 3-pentanona a 353.15K",
17:r"Sistema decano e acetona a 333.15K",
18:r"Sistema benzeno e dietilamina a 328.15K",
19:r"Sistema benzeno e trietilamina a 353.15K",
20:r"Sistema tolueno e nitrobenzeno a 373.15K",
21:r"Sistema heptano e cloreto de butila a 323.15K",
22:r"Sistema ciclopentano e clorofórmio a 298.15K",
23:r"Sistema heptano e trietilamina a 333.15K",
24:r"Sistema etilbenzeno e nitrobenzeno a 373.15K",
25:r"Sistema benzeno e nitrometano a 318.15K",
26:r"Sistema benzeno e tert-butanol a 318.15K",
27:r"Sistema heptano e iodeto de etila a 323.15K",
28:r"Sistema octano e piridina a 353.15K",
29:r"Sistema ciclohexano e etanol a 308.15K",
30:r"Sistema pentano e 1-butanol a 303.15K",
31:r"Sistema hexafluorobenzeno e 1-propanol a 288.15K",
32:r"Sistema hexafluorobenzeno e metanol a 288.15K",
33:r"Sistema aldeído propiônico e 2-butanona a 318.15K",
34:r"Sistema dietil éter e acetona a 303.15K",
35:r"Sistema dietil éter e iodeto de metila a 308.15K",
36:r"Sistema acetona e clorofórmio a 308.15K",
37:r"Sistema acetona e metanol a 328.15K",
38:r"Sistema 1,4 dioxano e metanol a 308.15K",
39:r"Sistema etanol e trietilamina a 308K",
40:r"Sistema 1-propanol e 2-metil-1-propanol a 343.15K",
41:r"Sistema metanol e 2-metil-1-propanol a 323.15K",
42:r"Sistema etanol e 2-metil-1-propanol a 333.15K",
43:r"Sistema butilamina e 1-butanol a 313.15K",
44:r"Sistema dietilamina e etanol a 313.15K",
45:r"Sistema etanol e acetonitrila a 313.15K",
46:r"Sistema butilamina e 1-propanol a 318.15K",
47:r"Sistema bromobenzeno e ciclohexanol a 383.15K",
48:r"Sistema 1,2-dicloroetano e 2-metil-1-propanol a 323.15K",
49:r"Sistema metanol e 1,2-dicloroetano a 313.15K",
50:r"Sistema água e dietilamina a 311.5K",
51:r"Sistema água e piridina a 362.98K",
52:r"Sistema hexano e nitroetano a 298.15K",
53:r"Sistema 2-metilpentano e nitroetano a 298.15K",
54:r"Sistema nitroetano e octano a 308.15K",
55:r"Sistema água e fenol a 317.55K",
56:r"Sistema tetraclorometano e ácido acético a 293.15K",
57:r"Sistema clorobenzeno e ácido propiônico a 313.15K"}

def format_sistemas(option):
    return sistemas[option]

# Definindo modelos
modelos = {0: r"Margules 2 sufixos",1: r"Margules 3 sufixos",2: r"NRTL"}
def format_func(option):
    return modelos[option]

st.session_state.visibility = "collapsed"
st.session_state.disabled = True
st.set_page_config(
    page_title="Equilíbrio líquido-vapor isotérmico",
    # page_icon="🧊",
    layout="centered",
    initial_sidebar_state="expanded"
)

main_col1,main_col2,main_col3 = st.columns([8,12,4])

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

with main_col2:
    st.write(r"Selecione o sistema:")
    sist = st.selectbox(
        "Ajuste de parâmetros",
        options=list(sistemas.keys()),format_func=format_sistemas,
        label_visibility=st.session_state.visibility,
        disabled=False,
    )
with main_col3:
    Graph = st.radio(
        "Escolha o gráfico:",
        ('Pxy', 'x,y', 'Delta G'))

text_col1, text_col2, text_col3= st.columns([9,9, 6])
with text_col3:
    st.write("Pressões de saturação:")
if sist != -1:
    with text_col2:
        st.write("Sistema:")
with text_col1:
    st.write("Parâmetros do modelo:")

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
    return x,y,P,G

if sist != -1:
    dataset = Dataset(sist)



    par_col1,par_col2,psat_col1,psat_col3,extra= st.columns([5,6,11,4,3])
    with par_col1:
        if option == 0:
            st.latex(r"\beta = "+str(f'{dataset.param2suf[0]:.3f}'))
        elif option == 1:
            st.latex(r"\alpha = " + str(f'{dataset.param3suf[0]:.3f}'))
            st.latex(r"\beta = " + str(f'{dataset.param3suf[1]:.3f}'))
        elif option == 2:
            st.latex(r"\alpha = " + str(f'{dataset.paramNRTL[0]:.2f}'))
    with psat_col1:
        st.write("Componente A: "+dataset.compA)
        st.write("Componente B: " + dataset.compB)
        st.write("Temperatura (em K): " + str(dataset.T))
    if option == 2:
        with par_col2:
            st.latex(r"\tau_{A,B}= " + str(f'{dataset.paramNRTL[1]:.3f}'))
            st.latex(r"\tau_{B,A}= " + str(f'{dataset.paramNRTL[2]:.3f}'))
    with psat_col3:
        st.latex(r"P^{sat}_A = "+str(f'{dataset.Psat[0]:.2f}')+"\ kPa")
        st.latex(r"P^{sat}_B = " + str(f'{dataset.Psat[1]:.2f}') + "\ kPa")
    if option==0:
        param = dataset.param2suf
    elif option == 1:
        param = dataset.param3suf
    elif option == 2:
        param = dataset.paramNRTL
    x,y,P,G = Calcular(param,dataset.Psat)
else:
    valores = [0.2, 1, 2.7,0.8,2.45, 1.15, 2, 1.]
    extra, col4, col5,  col6,col7 ,col1, col2, col3= st.columns(valores)
    with col1:
        st.latex(r"P^{sat}_A = ")
        st.latex(r"P^{sat}_B = ")

    with col2:
        PsatA = st.number_input("", 0, 1000, value=100, label_visibility=st.session_state.visibility)
        PsatB = st.number_input("", 0, 1000, value=50, label_visibility=st.session_state.visibility)

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
        with col6:
            st.latex(r"\tau_{A,B}=")
            st.latex(r"\tau_{B,A}=")

        with col5:
            alpha = st.number_input(r"\alpha=", min_value=0.0, max_value=1.0, value=0.30, step=0.01,
                                    label_visibility=st.session_state.visibility)
        with col7:
            dg12 = st.number_input(r"dg12", step=0.05, label_visibility=st.session_state.visibility)
            dg21 = st.number_input(r"dg21=", step=0.05, label_visibility=st.session_state.visibility)
            param = np.array([alpha, dg12, dg21])
    x, y, P, G = Calcular(param, np.array([PsatA, PsatB]))

# Definindo labels
xlabel = "x$_A$,y$_A$"
ylabel = "P (kPa)"


fig, ax = plt.subplots()

plt.subplots_adjust(top=0.75,bottom=0.25,left=0.15)
if Graph == 'Pxy':
    lz, = ax.plot([0, 1], [P[0], P[-1]], '--k', lw=1)
    lx, = ax.plot(x, P, lw=2)
    ly, = ax.plot(y, P, lw=2)
    if sist!=-1:
        lexpx = ax.plot(dataset.xexp,dataset.Pexp,'ok',ms=4)
        lexpy = ax.plot(dataset.yexp, dataset.Pexp, 'ok', ms=4)
        plt.legend(lexpx,["Dados experimentais"])

    xmin,xmax,ymin,ymax = ax.axis()
    ax.axis([0,1, ymin,ymax])
    # Ajustando a legenda
    ax.set_xlabel(xlabel,fontsize=12)
    ax.set_ylabel(ylabel,fontsize=12)
elif Graph == 'x,y':
    lz, = ax.plot([0, 1], [0, 1], '--k', lw=1)
    lx, = ax.plot(x, y, lw=2)
    if sist != -1:
        lexpx = ax.plot(dataset.xexp, dataset.yexp, 'ok', ms=4)
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
