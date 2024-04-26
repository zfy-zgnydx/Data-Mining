from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

# 加载投票网络数据
with open("Wiki-Vote.txt", "r") as f:
    lines = f.readlines()[4:]  # 跳过前四行的注释
    edges = [tuple(map(int, line.strip().split())) for line in lines]



# 创建投票网络图
vote_graph = nx.DiGraph()
vote_graph.add_edges_from(edges)



# 选择前100个节点
nodes_to_draw = list(vote_graph.nodes)[:100]

# 创建包含这些节点及其相邻边的子图
subgraph = vote_graph.subgraph(nodes_to_draw)

# 绘制子图
plt.figure(figsize=(10, 8))
pos = nx.spring_layout(subgraph, seed=42)  # 使用spring布局算法布局节点
nx.draw(subgraph, pos, with_labels=True, node_size=50, node_color="skyblue", edge_color="gray", alpha=0.5)
plt.title("Partial Wikipedia Vote Network")
plt.show()



# 将数据转换成适合 mlxtend 库的格式
te = TransactionEncoder()
te_ary = te.fit_transform(edges)
df = pd.DataFrame(te_ary, columns=te.columns_)

# 使用 Apriori 算法挖掘频繁项集
frequent_itemsets = apriori(df, min_support=0.005, use_colnames=True)

# 打印挖掘结果
print(frequent_itemsets)


# 使用关联规则分析
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=0.01)

# 打印关联规则
print("\n关联规则：")
print(rules)
