# Importação das bibliotecas
import streamlit as st 
import pandas as pd
import joblib
# Importando as classes customizadas necessárias para desserializar o pipeline
from utils import BinaryEncoder, OneHotEncodingTransformer, MinMaxTransformer

# Carregando o pipeline e o modelo treinados (uma única vez ao iniciar a aplicação)
@st.cache_resource
def load_model_and_pipeline():
    """Carrega o modelo e pipeline treinados (cached para melhor performance)"""
    model = joblib.load('modelo/final_model.joblib')
    pipeline = joblib.load('modelo/pipeline.joblib')
    return model, pipeline

model, pipeline = load_model_and_pipeline()

# Configuração da página
st.set_page_config(
    page_title="Preditor de Risco de Obesidade",
    page_icon="🏥",
    layout="wide"
)

############################# Streamlit ############################

st.markdown("<h1 style='text-align: center;'> Preditor de Risco de Obesidade 🏥</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #666;'>Preencha o formulário com suas informações de saúde e hábitos</h3>", unsafe_allow_html=True)

st.info('⚕️ Este sistema avalia o risco de obesidade com base em informações pessoais, hábitos alimentares e estilo de vida.')

# Dicionários de mapeamento (interface amigável -> valor do modelo)
GENDER_MAP = {'Feminino': 'Female', 'Masculino': 'Male'}
YES_NO_MAP = {'Sim': 'yes', 'Não': 'no'}
FCVC_MAP = {'Raramente': 1, 'Às vezes': 2, 'Sempre': 3}
NCP_MAP = {'1 refeição': 1, '2 refeições': 2, '3 refeições': 3, '4 ou mais refeições': 4}
CAEC_MAP = {'Não': 'no', 'Às vezes': 'Sometimes', 'Frequentemente': 'Frequently', 'Sempre': 'Always'}
CH2O_MAP = {'Menos de 1L/dia': 1, '1-2L/dia': 2, 'Mais de 2L/dia': 3}
CALC_MAP = {'Não bebo': 'no', 'Às vezes': 'Sometimes', 'Frequentemente': 'Frequently', 'Sempre': 'Always'}
FAF_MAP = {'Nenhuma': 0, '1-2 vezes/semana': 1, '3-4 vezes/semana': 2, '5 ou mais vezes/semana': 3}
TUE_MAP = {'0-2 horas/dia': 0, '3-5 horas/dia': 1, 'Mais de 5 horas/dia': 2}
MTRANS_MAP = {'Automóvel': 'Automobile', 'Motocicleta': 'Motorbike', 'Bicicleta': 'Bike', 
              'Transporte público': 'Public_Transportation', 'A pé': 'Walking'}

# Criando colunas para organizar melhor o layout
col1, col2 = st.columns(2, gap="large")

with col1:
    st.write('### 📋 Informações Pessoais')
    
    # Gênero
    input_gender = st.selectbox('Gênero', list(GENDER_MAP.keys()), help='Sexo biológico')
    
    # Idade
    input_age = st.slider('Idade (anos)', min_value=14, max_value=100, value=25, help='Idade em anos')
    
    # Altura
    input_height = st.number_input('Altura (metros)', min_value=1.40, max_value=2.00, value=1.70, step=0.01, help='Sua altura em metros')
    
    # Peso
    input_weight = st.number_input('Peso (kg)', min_value=30.0, max_value=200.0, value=70.0, step=0.5, help='Seu peso em quilogramas')
    
    # Histórico familiar
    input_family_history = st.radio('Histórico familiar de obesidade?', list(YES_NO_MAP.keys()), help='Alguém na sua família tem ou teve obesidade?')

with col2:
    st.write('### 🍽️ Hábitos Alimentares')
    
    # Consumo de alimentos calóricos
    input_favc = st.radio('Consumo frequente de alimentos muito calóricos?', list(YES_NO_MAP.keys()), 
                           help='Você come frequentemente fast-food, frituras ou alimentos muito calóricos?')
    
    # Frequência de consumo de vegetais
    input_fcvc = st.selectbox('Frequência de consumo de vegetais', 
                              list(FCVC_MAP.keys()),
                              index=1,
                              help='Com que frequência você consome vegetais nas refeições?')
    
    # Número de refeições principais
    input_ncp = st.selectbox('Número de refeições principais por dia', 
                             list(NCP_MAP.keys()),
                             index=2,
                             help='Quantas refeições principais você faz por dia?')
    
    # Consumo entre refeições
    input_caec = st.selectbox('Consumo de lanches entre refeições', 
                              list(CAEC_MAP.keys()),
                              help='Com que frequência você come entre as refeições?')
    
    # Consumo de água
    input_ch2o = st.selectbox('Consumo diário de água', 
                              list(CH2O_MAP.keys()),
                              index=1,
                              help='Quanto de água você consome por dia?')

# Segunda linha de colunas
col3, col4 = st.columns(2, gap="large")

with col3:
    st.write('### 🚬 Hábitos de Vida')
    
    # Fumar
    input_smoke = st.radio('Você fuma?', list(YES_NO_MAP.keys()), help='Hábito de fumar')
    
    # Monitoramento de calorias
    input_scc = st.radio('Monitora a ingestão de calorias?', list(YES_NO_MAP.keys()), 
                         help='Você acompanha quantas calorias consome por dia?')
    
    # Consumo de álcool
    input_calc = st.selectbox('Consumo de bebida alcoólica', 
                              list(CALC_MAP.keys()),
                              help='Com que frequência você consome álcool?')

with col4:
    st.write('### 🏃 Atividade Física e Tecnologia')
    
    # Frequência de atividade física
    input_faf = st.selectbox('Frequência de atividade física', 
                             list(FAF_MAP.keys()),
                             index=1,
                             help='Quantas vezes por semana você pratica atividade física?')
    
    # Tempo usando dispositivos eletrônicos
    input_tue = st.selectbox('Tempo usando dispositivos eletrônicos', 
                             list(TUE_MAP.keys()),
                             index=1,
                             help='Quanto tempo por dia você usa celular, computador, TV, etc?')
    
    # Meio de transporte
    input_mtrans = st.selectbox('Meio de transporte principal', 
                                list(MTRANS_MAP.keys()),
                                help='Como você geralmente se desloca?')

st.markdown('---')

# Botão de predição
if st.button('🔍 AVALIAR RISCO DE OBESIDADE', type='primary', use_container_width=True):
    with st.spinner('Analisando suas informações...'):
        
        # Convertendo os valores selecionados para os valores do modelo
        gender_modelo = GENDER_MAP[input_gender]
        family_history_modelo = YES_NO_MAP[input_family_history]
        favc_modelo = YES_NO_MAP[input_favc]
        fcvc_modelo = FCVC_MAP[input_fcvc]
        ncp_modelo = NCP_MAP[input_ncp]
        caec_modelo = CAEC_MAP[input_caec]
        smoke_modelo = YES_NO_MAP[input_smoke]
        ch2o_modelo = CH2O_MAP[input_ch2o]
        scc_modelo = YES_NO_MAP[input_scc]
        calc_modelo = CALC_MAP[input_calc]
        faf_modelo = FAF_MAP[input_faf]
        tue_modelo = TUE_MAP[input_tue]
        mtrans_modelo = MTRANS_MAP[input_mtrans]
        
        # Criando DataFrame com os dados do usuário
        dados_usuario = pd.DataFrame({
            'Gender': [gender_modelo],
            'Age': [float(input_age)],
            'Height': [float(input_height)],
            'Weight': [float(input_weight)],
            'family_history': [family_history_modelo],
            'FAVC': [favc_modelo],
            'FCVC': [float(fcvc_modelo)],
            'NCP': [float(ncp_modelo)],
            'CAEC': [caec_modelo],
            'SMOKE': [smoke_modelo],
            'CH2O': [float(ch2o_modelo)],
            'SCC': [scc_modelo],
            'FAF': [float(faf_modelo)],
            'TUE': [float(tue_modelo)],
            'CALC': [calc_modelo],
            'MTRANS': [mtrans_modelo]
        })
        
        # Calculando IMC
        dados_usuario['BMI'] = dados_usuario['Weight'] / (dados_usuario['Height'] ** 2)
        bmi = dados_usuario['BMI'].iloc[0]
        
        # Transformando os dados do usuário usando o pipeline já treinado durante o desenvolvimento do modelo
        dados_processados = pipeline.transform(dados_usuario)
        
        # Fazendo predição
        predicao = model.predict(dados_processados)
        probabilidade = model.predict_proba(dados_processados)
        
        # Exibindo resultados
        st.markdown('---')
        st.markdown('## 📊 Resultado da Análise')
        
        # Exibindo IMC
        col_result1, col_result2 = st.columns(2)
        
        with col_result1:
            st.metric(label="Seu IMC (Índice de Massa Corporal)", value=f"{bmi:.2f} kg/m²")
        
        with col_result2:
            prob_obesidade = probabilidade[0][1] * 100
            st.metric(label="Probabilidade de Obesidade", value=f"{prob_obesidade:.1f}%")
        
        st.markdown('---')
        
        # Resultado da predição baseado na probabilidade
        prob_obesidade = probabilidade[0][1] * 100
        
        if prob_obesidade >= 70:
            # Alto Risco
            st.error('### ⚠️ ALTO RISCO DE OBESIDADE')
            st.warning(f'''
            **Atenção!** Com base nas informações fornecidas, o modelo identificou um **alto risco de obesidade** 
            (probabilidade de {prob_obesidade:.1f}%).
            
            **Recomendações Urgentes:**
            - 🏥 **Consulte imediatamente** um médico ou nutricionista para avaliação completa
            - 🥗 Revise urgentemente seus hábitos alimentares
            - 🏃 Inicie um programa de atividades físicas (com orientação profissional)
            - 💧 Mantenha-se bem hidratado
            - 📊 Monitore regularmente seu peso e IMC
            - 🩺 Realize exames de saúde preventivos
            
            *Este resultado é apenas uma indicação baseada em dados estatísticos e não substitui avaliação médica profissional.*
            ''')
            
        elif prob_obesidade >= 30:
            # Médio Risco
            st.warning('### ⚡ MÉDIO RISCO DE OBESIDADE')
            st.info(f'''
            **Atenção!** Com base nas informações fornecidas, você apresenta um **risco moderado de obesidade** 
            (probabilidade de {prob_obesidade:.1f}%).
            
            **Recomendações Importantes:**
            - 🏥 Considere consultar um nutricionista para orientação personalizada
            - 🥗 Revise seus hábitos alimentares e reduza alimentos ultraprocessados
            - 🏃 Aumente gradualmente a frequência de atividades físicas
            - 💧 Aumente o consumo de água diário
            - 📊 Monitore seu peso e IMC regularmente
            - 🎯 Estabeleça metas de saúde realistas
            - 😴 Melhore a qualidade do sono
            
            **Importante:** Este é um momento ideal para mudanças preventivas! Pequenas alterações nos hábitos 
            podem fazer grande diferença.
            
            *Este resultado é apenas uma indicação baseada em dados estatísticos e não substitui avaliação médica profissional.*
            ''')
            
        else:
            # Baixo Risco
            st.success('### ✅ BAIXO RISCO DE OBESIDADE')
            st.info(f'''
            **Parabéns!** Com base nas informações fornecidas, você apresenta um **baixo risco de obesidade** 
            (probabilidade de {prob_obesidade:.1f}%).
            
            **Continue mantendo hábitos saudáveis:**
            - 🥗 Mantenha uma alimentação balanceada e variada
            - 🏃 Continue praticando atividades físicas regularmente
            - 💧 Mantenha-se bem hidratado
            - 😴 Durma bem e controle o estresse
            - 📊 Faça check-ups médicos regularmente
            - 🎯 Mantenha um estilo de vida ativo
            
            *Lembre-se: manter um estilo de vida saudável é um processo contínuo!*
            ''')

# Rodapé
st.markdown('---')
st.markdown('''
<div style='text-align: center; color: #666; font-size: 0.9em;'>
    <p>⚕️ Este sistema utiliza Machine Learning para avaliar risco de obesidade</p>
    <p>Os resultados são baseados em análise estatística e não substituem avaliação médica profissional</p>
    <p>Criado por Nicole Tometich e Giovanni Gerodo como entrega do Tech Challenge final da Fase 4 - Data viz and production models</p>
    <p>Pós graduação em Data Analytics FIAP</p>
</div>
''', unsafe_allow_html=True)
