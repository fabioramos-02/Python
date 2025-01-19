import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Exemplo simples: Dados simulados de ferramentas de avaliação
# Estruturando um DataFrame com as métricas coletadas

data = {
    'Site': ['site1', 'site2', 'site3', 'site4', 'site5'],
    'Ferramenta': ['Lighthouse', 'Access Monitor', 'Wave', 'ASES', 'Lighthouse'],
    'Tempo_Analise': [45, 30, 40, 50, 35],  # em segundos
    'Pontuacao_Acessibilidade': [90, 96, 85, 92, 88],
    'Erros_Identificados': [10, 3, 8, 5, 12]
}

# Criar DataFrame
sites_df = pd.DataFrame(data)

# Normalizar os dados numéricos para comparação
scaler = StandardScaler()
sites_df[['Tempo_Analise', 'Pontuacao_Acessibilidade', 'Erros_Identificados']] = scaler.fit_transform(
    sites_df[['Tempo_Analise', 'Pontuacao_Acessibilidade', 'Erros_Identificados']]
)

# Aplicar K-Means para agrupar ferramentas com base no desempenho
kmeans = KMeans(n_clusters=2, random_state=42)  # Dois clusters como exemplo
sites_df['Cluster'] = kmeans.fit_predict(sites_df[['Tempo_Analise', 'Pontuacao_Acessibilidade', 'Erros_Identificados']])

# Visualizar os clusters com legendas indicando a ferramenta
plt.figure(figsize=(8, 6))
for i, row in sites_df.iterrows():
    plt.scatter(row['Tempo_Analise'], row['Pontuacao_Acessibilidade'], c=f'C{row.Cluster}', label=f"{row['Ferramenta']} ({row['Site']})")
plt.title('Clusters de Ferramentas com Base no Desempenho')
plt.xlabel('Tempo de Análise (normalizado)')
plt.ylabel('Pontuação de Acessibilidade (normalizada)')
plt.colorbar(label='Cluster')

# Adicionar legendas personalizadas (uma por ferramenta/site)
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), loc='best', title='Ferramentas e Sites')

plt.show()

# Exibir os dados finais
print("\nDados Estruturados com Clusters:")
print(sites_df)

# Calcular estatísticas médias por ferramenta
print("\nMédias por Ferramenta:")
medias_por_ferramenta = sites_df.groupby('Ferramenta').mean()
print(medias_por_ferramenta)
