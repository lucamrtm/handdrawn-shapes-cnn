# Classificação de Formas Desenhadas à Mão com CNN

Rede neural convolucional para classificar **círculo, quadrado e triângulo** desenhados à mão, usando o dataset QuickDraw em imagens 28×28 em tons de cinza.  
---

## 1. Objetivo

Treinar uma CNN capaz de reconhecer três formas simples desenhadas à mão (**círculo, quadrado, triângulo**) a partir de imagens 28×28 (1 canal), avaliando:

- Acurácia em treino, validação e teste  
- Métricas por classe (precision, recall, F1-score)  
- Efeito de técnicas como **EarlyStopping** e **ReduceLROnPlateau** 

---

## 2. Dataset

- **Fonte:** QuickDraw (arquivos `full_numpy_bitmap_*` de círculo, quadrado e triângulo)  
- **Tamanho usado:** 6000 imagens  
  - 2000 de cada classe (círculo, quadrado, triângulo)  
- **Formato:**
  - 28×28 pixels  
  - 1 canal (tons de cinza)  
  - valores de intensidade entre 0 e 255  
- **Divisão (principal):**
  - Treino: 80% (4800 imagens)  
  - Validação: 10% (600 imagens)  
  - Teste: 10% (600 imagens)  
  - Split estratificado para manter o balanceamento das classes

---

## 3. Pré-processamento

- Carregamos 2000 exemplos de cada classe e **concatenamos** em um único array `X`
- Criamos os rótulos inteiros:
  - círculo = 0  
  - quadrado = 1  
  - triângulo = 2
- **Normalização:** `X = X.astype("float32") / 255.0`
- Adicionamos o canal de cinza: `X = X.reshape(-1, 28, 28, 1)`
- Dividimos em treino/val/test com `train_test_split`
- Criamos datasets do TensorFlow com:
  - `shuffle` apenas no treino  
  - `batch_size = 32`  
  - `prefetch(tf.data.AUTOTUNE)` para otimizar leitura

---

## 4. Arquitetura do Modelo

Modelo `Sequential` com 2 camadas convolucionais:

1. **Conv2D(32 filtros, 3×3, ReLU)**  
   - Entrada: `(28, 28, 1)`  
   - Extrai bordas e pequenos padrões  
2. **MaxPooling2D(2×2)**  
   - Reduz para ~14×14, foca nas features mais fortes  
3. **Conv2D(64 filtros, 3×3, ReLU)**  
   - Combina bordas em formas mais complexas (cantos, pedaços de círculo/quadrado)  
4. **MaxPooling2D(2×2)**  
   - Reduz para ~7×7  
5. **Flatten**  
   - transforma o mapa de features em vetor 1D (1600 neurônios)  
6. **Dense(128, ReLU)**  
   - “Camada de decisão” que combina as features  
7. **Dropout(0.5)**  
   - Zera aleatoriamente 50% dos neurônios da Dense durante o treino (reduz overfitting)  
8. **Dense(3, Softmax)**  
   - Saída de probabilidade para cada classe (círculo, quadrado, triângulo)

Total de parâmetros treináveis ≈ 224k.

---

## 5. Treinamento (hiperparâmetros)

- **Loss:** `sparse_categorical_crossentropy`  
  - rótulos são inteiros (0, 1, 2), não one-hot
- **Otimizador:** `Adam` com `learning_rate = 1e-3`  
  - Ajusta os pesos tentando minimizar a loss;
- **Batch size:** `32`  
  - compromisso entre estabilidade do gradiente, tempo de treino e memória
- **Épocas solicitadas:** 20  
- **Callbacks:**
  - `ReduceLROnPlateau(monitor="val_loss")`  
    - Se a `val_loss` para de melhorar por algumas épocas, reduz o LR para refinar melhor os pesos
  - `EarlyStopping(monitor="val_loss", patience=..., restore_best_weights=True)`  
    - Para o treino quando a `val_loss` não melhora mais  
    - Restaura automaticamente os **melhores pesos** (não os da última época)  
    - No experimento, o treino parou por volta da **época 9**, evitando overfitting desnecessário

---

## 6. Resultados

### 6.1 Métricas globais (conjunto de teste)

- **Test loss:** ≈ 0.065  
- **Test accuracy:** ≈ **98.8%**  
  - O modelo acerta ~99 em cada 100 imagens de teste

### 6.2 Métricas por classe

- **Círculo:** precision 0.98 • recall 1.00 • F1 0.99  
- **Quadrado:** precision 0.99 • recall 0.97 • F1 0.98  
- **Triângulo:** precision 1.00 • recall 0.99 • F1 0.99  

Interpretação rápida:

- Precision alta → quando o modelo diz “é círculo/quadrado/triângulo”, ele quase sempre acerta  
- Recall alto → quase não deixa passar exemplos sem reconhecer  
- F1 perto de 1.0 → bom equilíbrio entre não errar positivos e não gerar falsos positivos

### 6.3 Matriz de confusão

Principais padrões:

- **592 / 600 imagens corretas**  
- Alguns **quadrados** confundidos com círculos  
- Poucos triângulos confundidos com quadrados  
- Nenhum círculo previsto como outra forma

Erros aparecem em desenhos mais ambíguos (quadrados mais arredondados, triângulos estranhos etc.), o que é coerente com o comportamento esperado.

---

## 7. Como rodar o projeto

### 7.1. Requisitos

- **Python 3.10**  
- Bibliotecas principais:
  - `numpy`
  - `matplotlib`
  - `scikit-learn`
  - `tensorflow`

Você pode instalar manualmente ou usar um ambiente virtual.

### 7.2. Criar ambiente virtual (opcional, mas recomendado)

```bash
# dentro da pasta do projeto
python3 -m venv .venv
source .venv/bin/activate   # Linux / macOS
# .venv\Scripts\activate    # Windows (PowerShell/CMD)

pip install --upgrade pip
pip install numpy matplotlib scikit-learn tensorflow
                     ...