# 🚢 Duelo de Modelos — Previsão de Sobrevivência no Titanic

Projeto de Machine Learning ponta a ponta aplicado à competição [Titanic — Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic) do Kaggle, desenvolvido como parte do curso **EBAC — Profissão: Cientista de Dados**.

Mais do que um exercício de classificação, este repositório documenta um **ciclo real de iteração orientado por feedback do Kaggle**: a primeira versão do modelo pontuou abaixo do esperado no leaderboard, o gap entre validação interna e o mundo real foi diagnosticado, e uma segunda versão corrigiu o problema — com o ganho comprovado por uma nova submissão real, não só por métricas de validação.

## 📊 Resultados

| Versão | Validação cruzada (tunada) | Holdout interno | **Kaggle real** |
|---|---|---|---|
| v1 — XGBoost + SMOTE | 85,3% | 79,9% | **0,755** |
| v2 — Ensemble (SVM + LogReg), sem SMOTE | 86,7% | 79,3% | **0,780** |

O ponto central não é a métrica de validação cruzada (que na verdade foi parecida entre as duas versões) — é o **gap entre a validação interna e o score real**, que caiu de 4,4 para 1,3 pontos percentuais. Isso confirma que o ganho de 2,5 pontos no Kaggle veio de um modelo genuinamente mais generalizável, não de sorte ou de uma métrica de validação inflada.

## 🔍 O diagnóstico que motivou a v2

A v1 usava **SMOTE** para balancear as classes, aplicado *depois* da codificação one-hot das variáveis categóricas. Isso gera valores fracionários sem sentido físico em colunas binárias (ex.: `Title_Mr = 0.4`), ensinando o modelo com ruído artificial que não existe nos dados reais de teste. Combinado com atributos redundantes (`Age` e `AgeBin` ao mesmo tempo, uma variável `Deck` com categorias de 1-2 passageiros), o resultado foi um modelo com boa validação cruzada mas overfitting real.

**O que mudou na v2:**
- SMOTE → pesos de classe (`class_weight='balanced'` / `scale_pos_weight`), sem gerar dados sintéticos inválidos;
- Remoção de atributos redundantes (`AgeBin`, `FareBin`) e simplificação de `Deck` (8 → 4 categorias);
- Novo atributo **`Family_Survival`**: taxa de sobrevivência conhecida de outros integrantes do mesmo grupo familiar/bilhete (sem usar o próprio rótulo do passageiro) — uma técnica clássica e eficaz nesta competição;
- Ensemble por votação soft (SVM + Regressão Logística) em vez de depender de um único modelo "vencedor" do duelo.

## 🧪 Metodologia

1. **Análise Exploratória de Dados** — distribuições, valores ausentes, relações com o alvo.
2. **Diagnóstico e plano de melhoria** — análise honesta do gap holdout → Kaggle.
3. **Tratamento de dados e engenharia de atributos** — imputação sem vazamento (estatísticas calculadas apenas no `train.csv`), extração de título, atributos de família e cabine.
4. **Duelo de modelos** — Regressão Logística, Random Forest, SVM e XGBoost, comparados via validação cruzada estratificada (k=10).
5. **Otimização de hiperparâmetros** — `GridSearchCV` nos finalistas do duelo.
6. **Avaliação final** — holdout nunca visto durante treino/tuning, matriz de confusão, curva ROC, importância de atributos.
7. **Comparação com o notebook de referência do curso** — inclui uma análise crítica de uma limitação metodológica encontrada lá (métricas calculadas *in-sample*, sem holdout).
8. **Geração da submissão oficial** para o `test.csv` do Kaggle.
9. **Análise de negócio** — incluindo uma discussão sobre o teto realista de desempenho nesta competição e por que scores de 90%+ no leaderboard público quase sempre indicam vazamento de dados, não modelagem superior.

## 🛠️ Stack

`Python` · `pandas` · `scikit-learn` · `XGBoost` · `imbalanced-learn` (v1) · `seaborn`/`matplotlib` · Jupyter Notebook

## 📁 Estrutura do repositório

```
├── titanic_v1.ipynb          # Primeira iteração — XGBoost + SMOTE (Kaggle: 0.755)
├── titanic_v2.ipynb          # Segunda iteração — Ensemble sem SMOTE + Family_Survival (Kaggle: 0.780)
├── submission_v1.csv
├── submission_v2.csv
└── README.md
```

## 💡 Principal aprendizado

O maior valor deste projeto não está no número final de acurácia, mas no **processo**: usar o feedback de uma submissão real para diagnosticar overfitting, formular hipóteses específicas sobre a causa, implementar mudanças direcionadas e confirmar o efeito com uma nova submissão — o ciclo de trabalho real de um Cientista de Dados em produção.

---

**Autor:** Talita Luci (https://github.com/TalitaLuci) · Mestrando em Engenharia Mecânica (UDESC) · Curso EBAC — Profissão: Cientista de Dados
