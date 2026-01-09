# Documentação do Projeto: Sistema Preditivo de Obesidade

## Contexto do Projeto

Este projeto foi desenvolvido como entrega de trabalho da pós-graduação da Fase 4 em Data Analytics na FIAP. O desafio proposto simula uma situação real de contratação como cientista de dados de um hospital, com o objetivo de criar uma solução de Machine Learning para auxiliar a equipe médica no diagnóstico de obesidade.

### Objetivo

Desenvolver um **modelo preditivo de Machine Learning** utilizando a base de dados `obesity.csv` para auxiliar médicos e médicas a prever se uma pessoa pode ter obesidade, apoiando assim a tomada de decisão clínica.

---

## Estrutura do Projeto

O desenvolvimento foi organizado em **6 partes principais**, seguindo as melhores práticas de projetos de Data Science:

### **Parte 1: Carregamento e Tratamento dos Dados**
- Importação da base de dados
- Verificação de valores nulos e duplicatas
- Análise das dimensões do dataset (2.113 registros, 17 features)
- Criação de categorias agrupadas de obesidade
- Arredondamento de variáveis categóricas com ruído
- Criação da variável target binária `HasObesity`

### **Parte 2: Análise Exploratória dos Dados (EDA)**
- Análise de correlação entre variáveis numéricas
- Distribuição de variáveis demográficas (idade, altura, peso)
- Cálculo e análise do IMC (Índice de Massa Corporal)
- Análise de variáveis categóricas comportamentais
- Correlação Point-Biserial (variáveis numéricas vs target)
- Cramér's V (variáveis categóricas vs target)
- Verificação de outliers

### **Parte 3: Preparação dos Dados para Machine Learning**
- Divisão treino/teste (80%/20%) com estratificação
- Criação de transformadores customizados:
  - `MinMaxTransformer`: Normalização de variáveis numéricas
  - `BinaryEncoder`: Codificação de variáveis binárias
  - `OneHotEncodingTransformer`: Codificação de variáveis categóricas
- Construção de pipeline de preprocessamento

### **Parte 4: Treinamento dos Modelos**
- Aplicação do pipeline de transformação
- Treinamento de 3 modelos lineares de classificação
- Avaliação detalhada de cada modelo

### **Parte 5: Comparação dos Modelos**
- Análise comparativa de todas as métricas
- Seleção do melhor modelo considerando o contexto médico
- Justificativa técnica e clínica da escolha

### **Parte 6: Salvamento do Melhor Modelo**
- Serialização do modelo final em formato `.joblib`
- Preparação para uso em produção

---

## Modelos de Machine Learning Testados

### 1. Regressão Logística
**Descrição**: Modelo clássico de classificação binária que estima probabilidades usando a função logística (sigmoid).

**Características**:
- Simples e interpretável
- Fornece probabilidades calibradas
- Boa performance com features linearmente separáveis

**Configuração**:
```python
LogisticRegression(random_state=42, max_iter=1000)
```

### 2. Ridge Classifier
**Descrição**: Classificador baseado em regressão Ridge com regularização L2, calibrado para fornecer probabilidades.

**Características**:
- Regularização L2 penaliza coeficientes grandes
- Robustez contra multicolinearidade
- Requer calibração para produzir probabilidades (via `CalibratedClassifierCV`)

**Configuração**:
```python
RidgeClassifier(random_state=42) + CalibratedClassifierCV(cv=5)
```

### 3. SGD Classifier (Stochastic Gradient Descent)
**Descrição**: Classificador linear otimizado por descida de gradiente estocástico, eficiente para grandes datasets.

**Características**:
- Treinamento incremental
- Eficiente computacionalmente
- Configurado com `loss='log_loss'` para classificação logística

**Configuração**:
```python
SGDClassifier(loss='log_loss', random_state=42, max_iter=1000, tol=1e-3)
```

### Justificativa da Escolha de Modelos Lineares

Para este problema de classificação binária com features bem definidas, optou-se por **modelos lineares** pelos seguintes motivos:

1. **Interpretabilidade**: Em contexto médico, é crucial entender quais fatores influenciam a predição
2. **Simplicidade**: Menor risco de overfitting com dataset de tamanho moderado
3. **Eficiência Computacional**: Treinamento e inferência rápidos
4. **Adequação**: Relação entre features e target pode ser capturada adequadamente por modelos lineares

Modelos mais complexos (Random Forest, XGBoost) não foram necessários pois não apresentariam ganhos significativos de performance neste caso específico.

---

## Métricas de Avaliação

### 1. AUC Score (Area Under the ROC Curve)

**O que é**: Área sob a curva ROC, que varia de 0 a 1.

**Por que usar**: Mede a capacidade geral do modelo de discriminar entre as duas classes (obesidade vs não-obesidade), independente do threshold escolhido.

**Interpretação**:
- AUC = 1.0: Discriminação perfeita
- AUC = 0.9-1.0: Excelente
- AUC = 0.8-0.9: Muito bom
- AUC = 0.7-0.8: Bom
- AUC = 0.5: Aleatório (sem poder discriminatório)

### 2. Matriz de Confusão

**O que é**: Tabela que mostra as previsões corretas e incorretas separadas por classe.

**Por que usar**: Permite visualizar **onde** o modelo está errando, diferenciando entre os dois tipos de erros possíveis.

**No contexto médico**:
- **Falso Positivo (FP)**: Pessoa saudável diagnosticada com obesidade → Exames adicionais desnecessários (inconveniente, mas não perigoso)
- **Falso Negativo (FN)**: Pessoa com obesidade não diagnosticada → **NÃO recebe tratamento necessário** (risco à saúde - **CRÍTICO**)

### 3. Sensibilidade (Recall / Taxa de Verdadeiros Positivos)

**Fórmula**: `Sensibilidade = VP / (VP + FN)`

**O que mede**: Percentual de casos de obesidade que foram corretamente detectados pelo modelo.

**Interpretação**: Sensibilidade de 95% significa que o modelo detecta 95% dos casos de obesidade, mas deixa passar 5% (falsos negativos).

### 4. Especificidade (Taxa de Verdadeiros Negativos)

**Fórmula**: `Especificidade = VN / (VN + FP)`

**O que mede**: Percentual de casos sem obesidade que foram corretamente identificados como negativos.

**Por que usar**: Avalia a capacidade do modelo de não gerar "alarmes falsos", evitando diagnósticos incorretos em pessoas saudáveis.

### 5. Taxa de Falsos Negativos

**Fórmula**: `Taxa FN = FN / (VP + FN)`

**Por que é a métrica mais crítica neste projeto**: 
- Em contexto médico de obesidade, **falsos negativos são mais graves que falsos positivos**
- Um falso negativo significa que uma pessoa com obesidade não será tratada, colocando sua saúde em risco
- Um falso positivo resulta em exames adicionais, mas garante que casos reais não sejam perdidos

**Decisão de modelo**: Priorizamos o modelo com **menor taxa de falsos negativos**, mesmo que isso resulte em um leve aumento de falsos positivos.

### 6. Taxa de Falsos Positivos

**Fórmula**: `Taxa FP = FP / (VN + FP)`

**O que mede**: Percentual de pessoas sem obesidade que foram incorretamente classificadas como tendo obesidade.

### 7. Acurácia

**Fórmula**: `Acurácia = (VP + VN) / Total`

**O que mede**: Percentual total de predições corretas.

### 8. Teste KS (Kolmogorov-Smirnov)

**O que é**: Teste estatístico que mede a máxima separação entre as distribuições de probabilidades previstas para as duas classes.

**Por que usar**: Complementa o AUC, medindo a diferença entre as distribuições de scores das classes positiva e negativa.

### 9. Curva ROC (Receiver Operating Characteristic)

**O que é**: Gráfico que mostra a relação entre Taxa de Verdadeiros Positivos (Sensibilidade) e Taxa de Falsos Positivos para diferentes thresholds.

**Por que usar**: 
- Visualiza o trade-off entre sensibilidade e especificidade
- Permite escolher o threshold ideal baseado no contexto clínico
- A área sob a curva é o AUC Score

**Interpretação**: Quanto mais a curva se aproxima do canto superior esquerdo, melhor o modelo.

----

## Pipeline de Transformação

O pipeline desenvolvido garante que todas as transformações sejam aplicadas de forma consistente em treino e teste:

```python
Pipeline([
    ('binary_encoder', BinaryEncoder()),        # Codifica variáveis binárias (Gender, FAVC, etc.)
    ('onehot_encoder', OneHotEncodingTransformer()),  # Codifica variáveis categóricas (CAEC, CALC, MTRANS)
    ('min_max_scaler', MinMaxTransformer())     # Normaliza variáveis numéricas (Age, Weight, IMC, etc.)
])
```

**Vantagens do Pipeline**:
- Previne data leakage (fit apenas no treino, transform no teste)
- Garante reprodutibilidade
- Facilita deployment em produção
- Encapsula toda a lógica de preprocessamento

---

## Variável Target

### Definição: `HasObesity`

**Criação**:
```python
HasObesity = 1  se  Obesity in ['Obesity_Type_I', 'Obesity_Type_II', 'Obesity_Type_III']
HasObesity = 0  caso contrário (Normal Weight, Overweight, Insufficient Weight)
```

**Justificativa**:
- Transforma problema multiclasse em **classificação binária**
- Foco específico na detecção de obesidade (objetivo clínico)
- Simplifica o problema sem perder a essência diagnóstica
- Facilita interpretação médica: "tem obesidade" vs "não tem obesidade"

---

## Features Utilizadas

- **Age**: Idade do indivíduo
- **Gender**: Gênero (Male/Female)
- **Height**: Altura em metros
- **Weight**: Peso em quilogramas
- **IMC**: Índice de Massa Corporal (calculado)
- **FCVC**: Frequência de consumo de vegetais
- **NCP**: Número de refeições principais por dia
- **CAEC**: Consumo de alimentos entre refeições
- **SMOKE**: Hábito de fumar
- **CH2O**: Consumo diário de água
- **SCC**: Monitoramento de calorias consumidas
- **FAF**: Frequência de atividade física
- **TUE**: Tempo de uso de dispositivos eletrônicos
- **CALC**: Frequência de consumo de álcool
- **MTRANS**: Meio de transporte utilizado
- **family_history**: Histórico familiar de obesidade
- **FAVC**: Consumo frequente de alimentos altamente calóricos

**Total**: 17 features preditivas

---

## Principais Insights da Análise Exploratória

1. **IMC é o preditor mais forte**: Correlação Point-Biserial mais alta com obesidade
2. **Histórico familiar importa**: Cramér's V alto indica forte associação
3. **Fatores comportamentais**: FAF (atividade física) e CH2O (água) correlacionados negativamente com obesidade
4. **Dataset balanceado**: Não foi necessário aplicar técnicas de oversampling/undersampling
5. **Ausência de multicolinearidade**: Nenhuma correlação entre features excedeu 50%

### Métodos de Correlação Utilizados

#### Correlação Point-Biserial

**O que é**: Medida de correlação entre uma **variável contínua** e uma **variável binária**.

**Por que foi utilizada**: 
- Precisávamos medir a relação entre variáveis numéricas contínuas (Age, Weight, Height, IMC, etc.) e o target `HasObesity` (0 = sem obesidade, 1 = com obesidade)

#### Cramér's V

**O que é**: Medida de associação entre duas **variáveis categóricas**, baseada no teste qui-quadrado (χ²).

**Por que foi utilizada**:
- Precisávamos medir a associação entre variáveis categóricas (Gender, CAEC, CALC, MTRANS, etc.) e o target binário `HasObesity`

### Decisão sobre Outliers

**Por que NÃO removemos outliers**:

1. **Relevância Clínica**: 
   - Valores extremos de peso, altura e IMC são clinicamente relevantes para predição de obesidade
   - Os outliers encontrados não são erros de medição, mas representam pessoas reais em extremos da população
   - Remover esses casos diminuiria a capacidade do modelo de generalizar para toda a população

2. **Características do Problema**:
   - Obesidade extrema (Tipo II e III) naturalmente gera valores de peso e IMC que seriam identificados como outliers estatisticamente
   - Remover esses casos seria **remover exatamente os pacientes que mais precisam do diagnóstico**

3. **Natureza dos Dados**:
   - Os "outliers" identificados (peso muito alto, IMC elevado) são **informativos**, não ruidosos
   - São exatamente esses casos extremos que o modelo precisa aprender a identificar

4. **Análise Realizada**:
   - Verificamos outliers usando método IQR (Interquartile Range)
   - Identificamos que a maioria dos outliers estava na variável `Weight` e `IMC`
   - Confirmamos que esses valores eram consistentes com casos de obesidade severa

**Decisão Técnica**: Mantivemos todos os dados originais para garantir que o modelo possa identificar corretamente **todo o espectro de obesidade**, incluindo casos mais severos que são justamente os de maior risco à saúde.

**Exceção**: Se tivéssemos identificado outliers claramente resultantes de **erros de digitação** (ex: altura de 0.5m, peso de 500kg), estes seriam corrigidos ou removidos. Mas não foi o caso neste dataset.

---

## Critério de Seleção do Melhor Modelo

### Abordagem Inicial vs. Abordagem Final

**- Abordagem Inicial (Inadequada)**:
- Seleção baseada apenas no **AUC Score mais alto**
- Não considerava o contexto médico específico
- Ignorava o impacto diferenciado dos tipos de erros

**- Abordagem Final (Adequada)**:
- **Priorização da menor taxa de falsos negativos**
- Consideração do contexto médico onde falsos negativos são críticos
- Análise comparativa de 7 métricas diferentes
- Justificativa clínica e técnica da escolha

### Hierarquia de Critérios

1. **Menor Taxa de Falsos Negativos** (Prioridade máxima)
   - Minimiza casos de obesidade não detectados
   - Garante que pacientes recebam tratamento necessário

2. **Alta Sensibilidade**
   - Detecta o máximo possível de casos reais
   - Correlacionada inversamente com falsos negativos

3. **AUC Score Alto**
   - Confirma boa capacidade discriminatória geral
   - Valida a qualidade do modelo em múltiplos thresholds

4. **Especificidade Razoável**
   - Evita excesso de alarmes falsos
   - Balanceamento com sensibilidade

5. **Taxa Aceitável de Falsos Positivos**
   - Considerada secundária no contexto médico
   - Preferível ter mais FP que FN

---

## Resultado Final

### Modelo Recomendado: **Regressão Logística**

**Justificativa**:
- ✅ **Menor taxa de falsos negativos** entre os 3 modelos testados
- ✅ **Alta sensibilidade**: Detecta a grande maioria dos casos de obesidade
- ✅ **AUC Score excelente**: Boa capacidade discriminatória geral
- ✅ **Interpretabilidade**: Coeficientes podem ser analisados pela equipe médica
- ✅ **Simplicidade**: Fácil implementação e manutenção em ambiente hospitalar

### Trade-off Aceito

Embora o modelo possa apresentar uma taxa de falsos positivos ligeiramente maior que outros modelos com AUC superior, essa escolha é justificada pelo contexto médico onde:
- **Prioridade**: Não deixar casos de obesidade sem diagnóstico
- **Aceitável**: Alguns pacientes passarem por exames adicionais desnecessários
- **Inaceitável**: Pacientes com obesidade não receberem tratamento adequado

---

### Limitações e Considerações

 **Importante**: O modelo é uma **ferramenta de apoio**, não substitui o julgamento clínico profissional.

**Limitações**:
- Baseado em dados históricos específicos
- Conjunto de dados relativamente pequeno para ser considerado crível
- Requer recalibração periódica
- Não considera fatores clínicos não presentes no dataset
- Performance pode variar em diferentes populações


