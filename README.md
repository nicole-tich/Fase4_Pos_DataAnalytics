# 🏥 Preditor de Risco de Obesidade

Aplicação web desenvolvida com Streamlit que utiliza Machine Learning (Regressão Logística) para avaliar o risco de obesidade com base em informações pessoais, hábitos alimentares e estilo de vida.

🌐 **[Acesse a aplicação online](https://fase4-data-analytics-obesity.streamlit.app/)**


## 📋 Sobre o Projeto

Este projeto utiliza um modelo de **Regressão Logística** treinado para classificar indivíduos quanto ao risco de obesidade. O modelo foi treinado com dados do dataset `Obesity.csv` e alcançou excelentes métricas de desempenho.

### Características do Modelo

- **Algoritmo**: Regressão Logística
- **Variável Target**: HasObesity (0 = Baixo Risco, 1 = Alto Risco)
- **Features**: 16 variáveis incluindo dados demográficos, hábitos alimentares e estilo de vida

## 🚀 Como Executar

### Pré-requisitos

- Python 3.8 ou superior
- pip (gerenciador de pacotes Python)

### Instalação

1. Clone ou baixe este repositório

2. Navegue até a pasta do projeto:
```bash
cd Deploy
```

3. Instale as dependências:
```bash
pip install -r requirements.txt
```

### Executando a Aplicação

Execute o comando:
```bash
streamlit run app.py
```

A aplicação abrirá automaticamente no seu navegador em `http://localhost:8501`

## 📊 Como Usar

1. **Informações Pessoais**:
   - Selecione seu gênero
   - Informe idade, altura e peso
   - Indique se há histórico familiar de obesidade

2. **Hábitos Alimentares**:
   - Frequência de consumo de alimentos calóricos
   - Frequência de consumo de vegetais
   - Número de refeições principais
   - Consumo entre refeições
   - Consumo diário de água

3. **Hábitos de Vida**:
   - Hábito de fumar
   - Monitoramento de calorias
   - Consumo de álcool

4. **Atividade Física e Tecnologia**:
   - Frequência de atividade física
   - Tempo usando dispositivos eletrônicos
   - Meio de transporte principal

5. Clique em **"AVALIAR RISCO DE OBESIDADE"** para obter o resultado

## 📁 Estrutura do Projeto

```
Deploy/
│
├── app.py                      # Aplicação Streamlit principal
├── utils.py                    # Classes de transformação para pipeline
├── requirements.txt            # Dependências do projeto
├── README.md                   # Este arquivo
│
├── dados/
│   └── Obesity.csv            # Dataset com dados de treinamento
│
├── modelo/
│   └── final_model.joblib     # Modelo treinado
│
└── Notebooks/
    └── Obesity_ML_Model.ipynb # Notebook com análise e treinamento
```

## 🔬 Variáveis do Dataset

| Variável | Descrição | Valores |
|----------|-----------|---------|
| Gender | Gênero | Female, Male |
| Age | Idade em anos | 14-61 |
| Height | Altura em metros | 1.45-1.98 |
| Weight | Peso em kg | 39-173 |
| family_history | Histórico familiar de obesidade | yes, no |
| FAVC | Consumo de alimentos calóricos | yes, no |
| FCVC | Frequência de consumo de vegetais | 1-3 |
| NCP | Número de refeições principais | 1-4 |
| CAEC | Consumo entre refeições | no, Sometimes, Frequently, Always |
| SMOKE | Hábito de fumar | yes, no |
| CH2O | Consumo diário de água | 1-3 |
| SCC | Monitora calorias | yes, no |
| FAF | Frequência de atividade física | 0-3 |
| TUE | Tempo usando eletrônicos | 0-2 |
| CALC | Consumo de álcool | no, Sometimes, Frequently, Always |
| MTRANS | Meio de transporte | Automobile, Motorbike, Bike, Public_Transportation, Walking |

## 🎯 Interpretação dos Resultados

A aplicação retorna:

- **IMC (Índice de Massa Corporal)**: Calculado automaticamente
- **Probabilidade de Obesidade**: Percentual de risco
- **Classificação**: 
  - ✅ **Baixo Risco**: Indica baixa probabilidade de obesidade
  - ⚠️ **Alto Risco**: Indica alta probabilidade de obesidade com recomendações

## ⚠️ Aviso Importante

Este sistema utiliza Machine Learning para fins educacionais e de demonstração. Os resultados são baseados em análise estatística e **não substituem avaliação médica profissional**. Sempre consulte um profissional de saúde para diagnósticos e orientações médicas.

## 🛠️ Tecnologias Utilizadas

- **Python 3.x**
- **Streamlit**: Interface web interativa
- **Scikit-learn**: Machine Learning e pipeline de processamento
- **Pandas**: Manipulação de dados
- **NumPy**: Operações numéricas
- **Joblib**: Serialização do modelo

## 📝 Desenvolvimento

O modelo foi desenvolvido seguindo as etapas:

1. **Análise Exploratória**: Compreensão dos dados e relações entre variáveis
2. **Pré-processamento**: 
   - Arredondamento de variáveis com ruído
   - Criação da variável BMI
   - Criação da variável target binária (HasObesity)
3. **Transformações**:
   - Label Encoding para variáveis binárias
   - One-Hot Encoding para variáveis categóricas
   - Min-Max Scaling para variáveis numéricas
4. **Treinamento**: Teste de múltiplos modelos lineares
5. **Avaliação**: Seleção do melhor modelo baseado em métricas de desempenho

## 👨‍💻 Autor

Grupo 206
Nicole Tometich
Giovanni Gerodo

Desenvolvido como parte do Pós-Graduação em Data Analytics - FIAP
Fase 4 - Data Visualization e Deploy

---

Para mais informações, consulte o notebook `Obesity_ML_Model.ipynb` na pasta Notebooks.

