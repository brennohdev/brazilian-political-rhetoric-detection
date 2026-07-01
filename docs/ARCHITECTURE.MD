# Detecção de Manipulação Retórica em Discursos Políticos Brasileiros
## Arquitetura do Projeto — Documento de Referência

**Autores:** Brenno Henrique Alves da Silva Costa, Gustavo Henrique Rocha Oliveira  
**Disciplina:** Processamento de Linguagem Natural — UFPB  
**Data:** Junho 2026

---

## 1. Visão Geral

### 1.1 Pergunta de Pesquisa

Como modelos de linguagem de diferentes arquiteturas (generativos de grande escala e encoders fine-tuned para o português) diferem em sua capacidade de detectar técnicas de manipulação retórica em discursos parlamentares brasileiros? Em que medida esses modelos apresentam viés político sistemático em suas predições?

### 1.2 Hipóteses

- **H1 (Transferibilidade):** Técnicas de manipulação retórica definidas para notícias em inglês (SemEval-2020/2023) são parcialmente transferíveis para discursos parlamentares brasileiros, requerendo adaptações em pelo menos parte das categorias.
- **H2 (Viés Político):** Modelos generativos apresentam viés político mensurável ao classificar técnicas retóricas em textos com diferentes orientações políticas.
- **H3 (Performance Comparativa):** Modelos encoder fine-tuned superam abordagens few-shot com modelos generativos em F1-macro para detecção de manipulação retórica.

### 1.3 Contribuições Científicas Esperadas

1. Primeiro estudo comparando paradigmas (generativo vs. encoder) para detecção de manipulação retórica em discurso parlamentar brasileiro
2. Primeira avaliação de viés político em LLMs aplicados à tarefa de detecção de propaganda
3. Validação da transferibilidade das taxonomias SemEval para o domínio parlamentar em português brasileiro

---

## 2. Pipeline do Projeto

```
┌─────────────────────────────────────────────────────────────────────┐
│                        PIPELINE GERAL                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ETAPA 1: DADOS          ETAPA 2: TAXONOMIA     ETAPA 3: ANOTAÇÃO   │
│  ┌──────────────┐        ┌──────────────┐       ┌──────────────┐    │
│  │ A. Coleta    │        │ Seleção de   │       │ Guidelines   │    │
│  │ B. Filtragem │───────▶│ 6 técnicas   │──────▶│ Calibração   │    │
│  │ C. Amostragem│        │ (justificada)│       │ Anotação     │    │
│  │ D. Piloto    │        └──────────────┘       │ Concordância │    │
│  │ E. Decisão n │                               └──────┬───────┘    │
│  └──────────────┘                                      │             │
│                                                         ▼             │
│  ETAPA 4: CLASSIFICAÇÃO              ETAPA 5: AVALIAÇÃO              │
│  ┌──────────────────────┐            ┌──────────────────────┐        │
│  │ GPT-4o (few-shot)    │            │ H1: Cohen's κ        │        │
│  │ LLaMA 3.x (few-shot) │───────────▶│ H2: McNemar + OR     │        │
│  │ BERTimbau (fine-tuned)│            │ H3: McNemar + Boot CI│        │
│  └──────────────────────┘            └──────────────────────┘        │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. Etapa 1: Dados

### 3.1 Fonte

- API oficial de dados abertos da Câmara dos Deputados: `dadosabertos.camara.leg.br`
- Validada por Soriano et al. (2025) com 453.280 discursos coletados

### 3.2 Fase A — Coleta Bruta

- Coletar discursos com metadados completos: parlamentar, partido, data, tipo de sessão, fase legislativa
- Período: a definir conforme viabilidade (sugestão: incluir ao menos 1 período eleitoral recente)
- Resultado: universo completo de discursos disponíveis

### 3.3 Fase B — Filtragem e Pré-processamento

Baseado em Soriano et al. (2025):
- **Isolamento de monólogos:** regex para detectar troca de falante; preservar apenas primeiro segmento contínuo
- **Remoção de formalidades:** "Vossa Excelência", "Senhor Presidente", "nobres colegas", etc.
- **Normalização:** remoção de marcadores procedurais, padronização de caracteres
- **Deduplicação:** similaridade de cosseno com TF-IDF, threshold ≥ 0.75 (mesmo critério de Soriano et al.)
- **Filtro por tipo de discurso:** priorizar Grande Expediente e Ordem do Dia (mais argumentativos) sobre comunicações de liderança e encaminhamentos procedurais

### 3.4 Fase C — Amostragem Estratificada

- **Classificação política:** usar Zucco & Power (2024) — esquerda (-1), centro (0), direita (+1), mesma classificação usada por Soriano et al. (2025)
- **Estratificação por:** espectro político (esquerda/centro/direita), período temporal
- **Segmentação:** trechos de 3-5 sentenças, seguindo granularidade de Da San Martino et al. (2020)

### 3.5 Fase D — Piloto

- Selecionar ~50-100 segmentos da amostra
- Rodar GPT-4o e BERTimbau nesses segmentos
- Observar:
  - Quais técnicas realmente aparecem em discurso parlamentar (prevalência)
  - Taxa de discordância nos pares mínimos (efeito estimado para H2)
  - Viabilidade de cada técnica candidata

### 3.6 Fase E — Decisão do n

- Com dados do piloto, calcular tamanho amostral definitivo via power analysis
- Para McNemar (α=0.05, poder=0.80):
  - OR ≈ 3.0 → ~30 pares discordantes suficientes
  - OR ≈ 2.0 → ~70-80 pares discordantes
  - OR ≈ 1.5 → ~200+ pares discordantes
- O piloto informa qual cenário é o mais provável

---

## 4. Etapa 2: Taxonomia

### 4.1 Técnicas Candidatas (6)

| Técnica | Justificativa (baseada na literatura) |
|---------|--------------------------------------|
| Loaded Language | Mais frequente em todos os datasets (23.7% no SemEval-2023). GPT-4 γ=0.774 como consolidador (Hasanain 2024). RoBERTa-CRF F1=0.80 (Jose & Greenstadt 2024). |
| Name Calling/Labeling | Segunda mais frequente (18.5% SemEval-2023). GPT-4 γ=0.680 (Hasanain 2024). RoBERTa-CRF F1=0.74. |
| Doubt/Casting Doubt | Terceira mais frequente (12.5% SemEval-2023). GPT-4 γ=0.695-0.815 dependendo do papel (Hasanain 2024). |
| Appeal to Fear | Dificuldade média: RoBERTa-CRF F1=0.48, GPT-4 F1=0.09. Relevante para discurso político. |
| Causal Oversimplification | GPT-4 γ=0.889 como annotator (Hasanain 2024) — melhor performance entre todas as técnicas. |
| Flag-Waving | Muito relevante para política brasileira (nacionalismo, identidade partidária). Dificuldade média: RoBERTa F1=0.82 mas GPT-4 F1=0.14. |

### 4.2 Critérios de Seleção

1. **Frequência:** técnicas que aparecem com volume suficiente para análise estatística
2. **Detectabilidade:** ao menos algum modelo consegue identificá-la com F1 > 0.10 na literatura
3. **Relevância política:** técnicas que fazem sentido no contexto parlamentar brasileiro
4. **Decisão final:** confirmada no piloto (Fase D). Se alguma técnica nunca aparece nos discursos reais, é descartada.

---

## 5. Etapa 3: Anotação e Ground Truth

### 5.1 Protocolo

1. **Guidelines de anotação:** documento em português com definição de cada técnica + 2-3 exemplos positivos e negativos extraídos de discursos parlamentares reais
2. **Calibração:** ambos anotadores rotulam 20-30 segmentos, calculam κ, discutem discordâncias, refinam guidelines
3. **Anotação principal:** independente, sem comunicação durante o processo
4. **Medição de concordância:** Cohen's κ global e por técnica
5. **Resolução de discordâncias:** discussão até consenso (seguindo Banik et al. 2025, κ=0.65-0.68 com 3 anotadores)

### 5.2 Critérios de Qualidade

- κ ≥ 0.60 considerado aceitável (Landis & Koch 1977)
- Se κ < 0.40 para alguma técnica, ela é revisada ou descartada
- Precedente: Banik et al. (2025) usaram 3 pesquisadores com processo idêntico

### 5.3 Volume Alvo

- 100-200 segmentos anotados (corpus principal para comparação de modelos)
- 100-150 pares mínimos (para avaliação de viés)
- Total de trabalho: ~2-3 semanas com protocolo pronto

### 5.4 Construção dos Pares Mínimos (para H2)

- Extrair frases com manipulação retórica real dos discursos
- Criar versões pareadas substituindo apenas o referente político
- Exemplo:
  - "O governo [Lula] está destruindo a economia com políticas irresponsáveis" (Loaded Language)
  - "O governo [Bolsonaro] está destruindo a economia com políticas irresponsáveis" (Loaded Language)
- Controle: substituição não pode alterar sentido gramatical nem plausibilidade
- Volume: 100-150 pares × 3 modelos = 300-450 classificações para análise

---

## 6. Etapa 4: Classificação

### 6.1 Modelos

| Modelo | Paradigma | Configuração | Justificativa |
|--------|-----------|-------------|---------------|
| GPT-4o | Generativo, proprietário | Few-shot com definições da taxonomia + exemplos, saída JSON | Melhor LLM na literatura (Jose & Greenstadt 2024, Hasanain 2024). Baseline generativo forte. |
| LLaMA 3.x (8B) | Generativo, open-source | Mesma configuração de prompt do GPT-4o, via Ollama local | Permite reprodutibilidade. Comparação aberto vs. fechado. |
| BERTimbau-base | Encoder, fine-tuned, PT-BR | Token classification multi-label, sigmoid + BCE loss | Único modelo pré-treinado especificamente em PT-BR (Souza et al. 2020). Testa H3 no idioma alvo. |

### 6.2 Configuração do Few-shot (GPT-4o e LLaMA)

- Prompt estruturado com:
  - Definição de cada técnica (derivada das guidelines de anotação)
  - 2-3 exemplos por técnica (extraídos da calibração)
  - Instrução de saída em JSON: `{"tecnica": "...", "span": "...", "inicio": N, "fim": N}`
- Temperatura: 0 (determinístico)
- Mesmo prompt para ambos os modelos (controle)

### 6.3 Configuração do Fine-tuning (BERTimbau)

- Arquitetura: BERTimbau-base + camada de classificação multi-label (sigmoid)
- Loss: Binary Cross-Entropy
- Hiperparâmetros: lr=2e-5, batch=16, epochs tuned no dev set, threshold tuned (grid search 0.05-0.5)
- Split: 70/15/15 (treino/dev/teste), estratificado por partido e técnica
- Baseado em: Hasanain et al. (2024) que usaram AraBERT com setup idêntico

### 6.4 Infraestrutura

- MacBook M4 16GB RAM
- BERTimbau fine-tune: via PyTorch com backend MPS (Apple Silicon)
- LLaMA 8B: via Ollama com quantização Q4
- GPT-4o: via API OpenAI

---

## 7. Etapa 5: Avaliação

### 7.1 Mapeamento Hipótese → Teste

| Hipótese | Pergunta | Tipo de dado | Teste | O que reportar |
|----------|----------|--------------|-------|----------------|
| H1 | Taxonomia é aplicável? | Anotações de 2 raters, 6 categorias | Cohen's κ | κ global + por técnica |
| H2 | Existe viés político? | Pares mínimos, predição binária pareada | McNemar + Odds Ratio + IC 95% | OR + IC + p, por modelo × técnica |
| H3 | Encoder > LLM? | Predições pareadas no mesmo test set | McNemar + Bootstrap CI (B=10.000) | F1 ± IC 95%, p-valor por par de modelos |

### 7.2 H1: Transferibilidade da Taxonomia

**O que mede:** concordância entre os 2 anotadores ao aplicar taxonomia SemEval em texto parlamentar.

**Teste:** Cohen's κ (2 anotadores, múltiplas categorias).

**Justificativa:** κ corrige para concordância ao acaso. Percentual simples de acordo não serve porque classes frequentes inflam concordância.

**Critério:** κ ≥ 0.60 = aplicável; κ < 0.40 = técnica problemática.

**Referência:** Landis & Koch (1977) para interpretação; Da San Martino et al. (2020) reportaram γ=0.6 com 6 anotadores profissionais.

### 7.3 H2: Viés Político

**O que mede:** se predição muda ao trocar referente político nos pares mínimos.

**Teste:** McNemar (proporções pareadas, resposta binária).

**Quantificação:** Odds Ratio com IC 95%.
- OR > 1 → viés contra esquerda
- OR < 1 → viés contra direita  
- OR = 1 → sem viés
- IC não inclui 1 → efeito significativo

**Correção para múltiplos testes:** Bonferroni. Se testam 3 modelos × 6 técnicas = 18 testes, α ajustado = 0.05/18 ≈ 0.003.

**Justificativa do McNemar (não qui-quadrado):** observações são pareadas (mesmo texto, só muda referente). Qui-quadrado assume independência.

**Justificativa do McNemar (não t-test):** variável é binária (detectou/não detectou), não contínua.

### 7.4 H3: Performance Comparativa

**O que mede:** diferença de performance entre modelos no mesmo test set.

**Métricas base:** Precision, Recall, F1 por técnica; Macro-F1 e Micro-F1 agregados.

**Teste de significância:** McNemar para cada par de sistemas (3 pares: GPT-4o vs. LLaMA, GPT-4o vs. BERTimbau, LLaMA vs. BERTimbau).

**Intervalos de confiança:** Bootstrap (B=10.000 amostras) para F1, reportado como F1 ± IC 95%. Recomendado por Dror et al. (2018).

**O que reportar:** tabela com F1 por modelo × técnica, p-valores McNemar por par, bootstrap CI.

### 7.5 Análise Qualitativa de Erros

Além dos testes quantitativos:
- Exemplos de acertos e erros por modelo
- Padrões de confusão entre técnicas
- Casos onde o viés se manifesta de forma interpretável
- Comparação de raciocínio: o que o modelo "vê" vs. o que o humano anotou

---

## 8. Cronograma Estimado

| Fase | Atividade | Duração |
|------|-----------|---------|
| 1 | Coleta, filtragem, pré-processamento | 1-2 semanas |
| 2 | Amostragem + piloto | 1 semana |
| 3 | Decisão de n + construção dos pares mínimos | 1 semana |
| 4 | Elaboração das guidelines + calibração | 1 semana |
| 5 | Anotação principal | 2-3 semanas |
| 6 | Experimentos com modelos | 2 semanas |
| 7 | Análise estatística + interpretação | 1-2 semanas |
| 8 | Escrita do paper | 2-3 semanas |
| **Total** | | **~12-16 semanas** |

---

## 9. Referências Base

- Da San Martino et al. (2020) — SemEval-2020 Task 11: taxonomia de 14 técnicas, corpus PTC
- Piskorski et al. (2023) — SemEval-2023 Task 3: taxonomia expandida de 23 técnicas, multilíngue
- Soriano et al. (2025) — Câmara dos Deputados: corpus de 453K discursos, API viável
- Hasanain et al. (2024) — GPT-4 como anotador: limitações e capacidades por técnica
- Jose & Greenstadt (2024) — GPT-4 F1=0.16 vs. RoBERTa-CRF F1=0.67
- Banik et al. (2025) — viés humano-modelo em detecção de bias político
- Schneider et al. (2026) — detecção de narrativas manipulativas políticas com LLMs
- Souza et al. (2020) — BERTimbau: modelo pré-treinado para PT-BR
- Dror et al. (2018) — testes estatísticos para NLP (bootstrap CI)
- Zucco & Power (2024) — classificação ideológica dos partidos brasileiros

---

## 10. Decisões Pendentes

| Decisão | Depende de | Quando resolver |
|---------|-----------|----------------|
| Período temporal exato dos discursos | Viabilidade técnica (volume da API) | Fase A |
| Quais das 6 técnicas se mantêm | Resultado do piloto (Fase D) | Após piloto |
| n definitivo (segmentos + pares) | Power analysis com dados do piloto | Após piloto |
| Threshold de predição do BERTimbau | Grid search no dev set | Durante fine-tune |
| Usar BERTimbau-base ou large | Memória disponível (16GB pode limitar large) | Teste local |

---

## 11. Riscos e Mitigações

| Risco | Impacto | Mitigação |
|-------|---------|-----------|
| Técnicas retóricas raras no corpus parlamentar | Sem dados suficientes para algumas técnicas | Piloto identifica isso antes do estudo principal; descartar técnicas sem ocorrência |
| κ baixo entre anotadores | Ground truth não confiável | Calibração iterativa; refinamento de guidelines; técnicas com κ<0.40 descartadas |
| GPT-4o com custo alto de API | Orçamento limitado | Limitar chamadas ao n necessário; usar LLaMA como alternativa open |
| BERTimbau não converge no fine-tune | Sem modelo encoder para comparar | Testar com dados aumentados; reduzir número de técnicas; usar learning rate menor |
| Viés não detectado (OR≈1) | H2 não confirmada | Reportar honestamente como resultado negativo; contribuição válida |
