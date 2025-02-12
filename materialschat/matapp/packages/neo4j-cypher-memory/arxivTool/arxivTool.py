import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import ArxivLoader
import fitz
import requests
import json
import nltk
import time
import leidenalg
import igraph as ig
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
import matplotlib.pyplot as plt

nltk.download("punkt")

# 加载.env文件
load_dotenv(".env")

SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY")
SILICONFLOW_API_BASE = os.getenv("SILICONFLOW_API_BASE")

# 将用户提问统一翻译为英文并构建ArxivExplorer搜索URL
def translator(user_question):
    chat_model = ChatOpenAI(
        model="Qwen/Qwen2.5-72B-Instruct",
        openai_api_base=SILICONFLOW_API_BASE,
        openai_api_key=SILICONFLOW_API_KEY,
    )

    messages = [
        ("system", "Translate the following text to English. Only return the translation without any explanation:"),
        ("human", f"{user_question}"),
    ]

    ai_msg = chat_model.invoke(messages)
    return ai_msg.content

# 使用ArxivExplorer获取相关文献摘要
def fetch_arxiv_summaries_byexplorer(user_question):
    encoded_query = translator(user_question)
    url = f"https://arxivxplorer.com/?query={encoded_query}"

    chrome_options = Options()
    chrome_options.add_argument("--headless")  # 启用无头模式

    # 初始化WebDriver
    driver = None
    try:
        driver = webdriver.Chrome(options=chrome_options)

        # 访问目标网页
        driver.get(url)

        # 等待页面加载
        WebDriverWait(driver, 60).until(
            EC.presence_of_element_located(
                (By.CSS_SELECTOR, "h2.chakra-heading.css-1cy2trb")
            )
        )

        # 使用BeautifulSoup解析HTML内容
        html = driver.page_source
        soup = BeautifulSoup(html, "html.parser")

        # 找到所有包含标题和摘要的元素
        titles = soup.find_all("h2", class_="chakra-heading css-1cy2trb")[
            :12
        ]  # 根据您提供的类名来查找标题
        abstracts = soup.find_all("p", class_="chakra-text css-1s87rte")[
            :12
        ]  # 根据您提供的类名来查找摘要

        titles_ = []
        abstracts_ = []
        # 打印标题和摘要
        for title, abstract in zip(titles, abstracts):
            titles_.append(title.get_text(strip=True))
            abstracts_.append(abstract.get_text(strip=True))

        print(f"Found {len(titles_)} titles and {len(abstracts_)} abstracts.")
        return abstracts_, titles_
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if driver is not None:
            driver.quit()

# 调用硅基流动API获取文本嵌入向量（BGE Embedding）
def get_embeddings(texts, model="Pro/BAAI/bge-m3", token="your-token-here"):
    url = "https://api.siliconflow.cn/v1/embeddings"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    embeddings = []
    for text in texts:
        try:
            payload = {"model": model, "input": text, "encoding_format": "float"}
            response = requests.request("POST", url, json=payload, headers=headers)
            response_json = response.json()
            embedding = response_json["data"][0]["embedding"]
            embeddings.append(embedding)
        except Exception as e:
            print(f"Error occurred while processing text: {text}")
            print(e)

    return np.array(embeddings)


# 使用KMeans算法聚类文献摘要向量
def cluster_summaries(summaries, num_clusters=5):
    # 获取摘要的嵌入向量
    embeddings = get_embeddings(summaries, token=SILICONFLOW_API_KEY)
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    kmeans.fit(embeddings)
    return kmeans.labels_, kmeans.cluster_centers_, embeddings


# 使用Leiden算法聚类文献摘要向量（更适合大型图）
def cluster_summaries_leiden(summaries, resolution=1.0, threshold=0.6):
    # 获取摘要的嵌入向量
    embeddings = get_embeddings(summaries, token=SILICONFLOW_API_KEY)
    print("嵌入向量的形状：", embeddings.shape)
    # 计算余弦相似度矩阵
    similarity_matrix = cosine_similarity(embeddings)

    print("相似度矩阵的形状：", similarity_matrix.shape)
    # 构建图，将相似度矩阵转换为邻接矩阵(可以设置阈值)
    adjacency_matrix = (similarity_matrix > threshold).astype(int)

    # 创建图对象
    G = ig.Graph.Adjacency(adjacency_matrix.tolist(), mode="undirected")

    # 将相似度作为边的权重
    weights = []
    for i in range(len(G.es)):
        source, target = G.es[i].tuple
        weights.append(similarity_matrix[source][target])
    G.es["weight"] = weights

    # 运行Leiden算法
    partition = leidenalg.find_partition(
        G,
        leidenalg.RBConfigurationVertexPartition,
        weights="weight",
        resolution_parameter=resolution,
    )

    # 获取聚类标签
    labels = partition.membership

    # 计算聚类中心(使用每个聚类中嵌入向量的平均值)
    unique_labels = np.unique(labels)
    cluster_centers = []
    for label in unique_labels:
        mask = np.array(labels) == label
        center = np.mean(embeddings[mask], axis=0)
        cluster_centers.append(center)
    cluster_centers = np.array(cluster_centers)

    # 确定聚类的数量
    num_communities = max(labels) + 1

    # 例如，使用'nipy_spectral' colormap
    cmap = plt.get_cmap("nipy_spectral", num_communities)

    # 创建颜色列表
    colors = [cmap(i) for i in range(num_communities)]

    vertex_colors = [colors[label] for label in labels]

    # 设置节点标签
    G.vs["label"] = labels

    # 设置节点大小
    G.vs["size"] = 20

    # 设置边宽度
    G.es["width"] = [1 + 2 * weight for weight in G.es["weight"]]

    # 计算度中心性
    degree_centrality = G.degree()
    # 找到每个聚类的代表性文献（度中心性最高的节点）
    representative_documents = {}
    for label in set(labels):
        cluster_nodes = [node for node, lab in enumerate(labels) if lab == label]
        max_degree = max(degree_centrality[node] for node in cluster_nodes)
        representative_node = [
            node for node in cluster_nodes if degree_centrality[node] == max_degree
        ]
        representative_documents[label] = representative_node[
            0
        ]  # 选择第一个节点作为代表

    # 突出显示中心节点
    for node in representative_documents.values():
        # vertex_colors[node] = "red"  # 将中心节点颜色设置为红色
        G.vs[node]["size"] = 50  # 设置中心节点大小

    # 绘制图形
    visual_style = {
        "vertex_size": G.vs["size"],
        "vertex_color": vertex_colors,
        "vertex_label": G.vs["label"],
        "vertex_label_size": 20,
        "edge_width": G.es["width"],
        "layout": G.layout("fr"),  # 使用Fruchterman-Reingold布局
        "bbox": (800, 800),  # 图形大小
        "margin": 20,  # 边距
    }

    # 保存图形到文件
    ig.plot(G, **visual_style, target="image.png")

    return labels, cluster_centers, embeddings, representative_documents


# 打印社区聚类信息
def print_cluster_info(cluster_labels, summaries, titles):
    # 将摘要按聚类分组
    clustered_summaries = {}
    cluster_titles = {}
    for i, label in enumerate(cluster_labels):
        clustered_summaries.setdefault(label, []).append(summaries[i])
        cluster_titles.setdefault(label, []).append(titles[i])

    # 为每个聚类生成摘要
    community_summaries = {}
    for label, summaries in clustered_summaries.items():
        summary = generate_community_summaries(summaries)
        community_summaries[label] = summary
    community_info = ""
    # 打印每个聚类的摘要
    for label, summary in community_summaries.items():
        community_info += f"Community {label}: {summary}\n"
        for title in cluster_titles[label]:
            community_info += f"\n-Title: {title}\n"
        community_info += "\n"
    return community_info

# 打印社区聚类信息（按照既定格式）
def print_cluster_info_with_format(cluster_labels, summaries, titles):
    # 将摘要按聚类分组
    clustered_summaries = {}
    cluster_titles = {}
    for i, label in enumerate(cluster_labels):
        clustered_summaries.setdefault(label, []).append(summaries[i])
        cluster_titles.setdefault(label, []).append(titles[i])

    # 为每个聚类生成摘要
    community_summaries = {}
    for label, summaries in clustered_summaries.items():
        summary = generate_community_summaries(summaries)
        community_summaries[label] = summary

    # 按label排序并生成美化的输出
    community_info = ""
    sorted_labels = sorted(community_summaries.keys())

    for label in sorted_labels:
        # 使用HTML格式美化输出
        community_info += f"""
        <div class="community-card">
            <h4 style="color: #1E88E5; margin-bottom: 10px;">📑 社区 {label}</h4>
            <div class="community-summary">
                <p style="font-weight: 500; margin-bottom: 15px;">{community_summaries[label]}</p>
            </div>
            <div class="community-titles">
                <h5 style="color: #424242; margin-bottom: 8px;">📚 相关论文：</h5>
                <ul style="list-style-type: none; padding-left: 0;">
        """

        for idx, title in enumerate(cluster_titles[label], 1):
            community_info += f'<li style="margin-bottom: 5px; padding-left: 20px; position: relative;">'
            community_info += (
                f'<span style="position: absolute; left: 0;">{idx}.</span> {title}</li>'
            )

        community_info += """
                </ul>
            </div>
        </div>
        """

    return community_info


# 使用正则匹配的方式粗略去除参考文献和致谢部分
def remove_references(text):
    reference_keywords = [
        "REFERENCES",
        "REFERENCE",
        "References",
        "Reference",
        "ACKNOWLEDGMENTS",
        "ACKNOWLEDGMENT",
        "Acknowledgments",
        "Acknowledgement",
    ]
    lowest_index = len(text)

    for keyword in reference_keywords:
        index = text.find(keyword)
        if index != -1 and index < lowest_index:
            lowest_index = index

    if lowest_index < len(text):
        print(
            f"Found ack and reference section starting with '{text[lowest_index:lowest_index+20]}...'"
        )
        return text[:lowest_index].strip()
    else:
        print("No reference section found")
        return text

# 从文献中提取chunk段落文本
def split_into_chunks(text, chunk_size=1000):
    sentences = nltk.sent_tokenize(text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= chunk_size:
            current_chunk += sentence + " "
        else:
            if current_chunk:
                chunks.append(current_chunk.replace("\n", " ").strip())
            current_chunk = sentence + " "

    if current_chunk:
        chunks.append(current_chunk.replace("\n", " ").strip())

    return chunks


# chunk相似性检索
def get_most_relevant_chunk(user_question, chunks, n=3):
    user_question_embedding = get_embeddings(
        [user_question], token=SILICONFLOW_API_KEY
    )[0]
    chunk_embeddings = get_embeddings(chunks, token=SILICONFLOW_API_KEY)

    user_question_embedding = user_question_embedding.reshape(-1, 1)

    print("user_question_embedding shape:", user_question_embedding.shape)
    print("chunk_embeddings shape:", chunk_embeddings.shape)
    dot_products = np.dot(chunk_embeddings, user_question_embedding)
    norm_a = np.linalg.norm(chunk_embeddings, axis=1, keepdims=True)
    norm_b = np.linalg.norm(user_question_embedding, axis=0, keepdims=True)
    cosine_similarities = dot_products / (norm_a * norm_b)
    cosine_similarities_flat = cosine_similarities.flatten()

    # 找到相似度最高的n个索引
    top_n_indices = np.argsort(cosine_similarities_flat)[-n:][::-1]

    # 获取相应的chunks
    top_n_chunks = [chunks[i] for i in top_n_indices]

    return top_n_chunks


# 根据相关的chunk和user_question总结文献信息（可以改进）
def answer_question(user_question, chunks):
    chat_model = ChatOpenAI(
        model="Qwen/Qwen2.5-72B-Instruct",
        openai_api_base=SILICONFLOW_API_BASE,
        openai_api_key=SILICONFLOW_API_KEY,
    )

    messages = [
        ("system", "请根据以下内容回答用户的问题："),
        ("human", f"从文献中找到的相关内容：{chunks}\n问题：{user_question}"),
    ]

    ai_msg = chat_model.invoke(messages)
    return ai_msg.content

# 根据文献摘要总结出社区主题
def generate_community_summaries(summaries):
    chat_model = ChatOpenAI(
        model="Qwen/Qwen2.5-72B-Instruct",
        openai_api_base=SILICONFLOW_API_BASE,
        openai_api_key=SILICONFLOW_API_KEY,
    )

    messages = [
        ("system", "请根据以下文献摘要总结出该社区的主题，不超过两句话："),
        ("human", f"文献摘要：{summaries}\n 社区主题："),
    ]

    ai_msg = chat_model.invoke(messages)
    return ai_msg.content


# 测试主流程
if __name__ == "__main__":
    start_time_1 = time.time()
    user_question = "Bi2Se3的能带结构"

    summaries, titles = fetch_arxiv_summaries_byexplorer(user_question)

    cluster_labels, _, embeddings, representative_docs = cluster_summaries_leiden(
        summaries, resolution=1.0
    )

    # 打印聚类信息
    print_cluster_info(cluster_labels, summaries, titles)

    end_time_1 = time.time()
    elaspsed_time_1 = end_time_1 - start_time_1
    print("第一阶段耗时/s：", elaspsed_time_1)

    # 获取用户确认的聚类
    confirmed_cluster = 0

    start_time_2 = time.time()

    doc_index = representative_docs[confirmed_cluster]
    most_relevant_title = titles[doc_index]
    print("最相关的文献：", most_relevant_title)

    pdf_text = (
        ArxivLoader(query=most_relevant_title, load_max_docs=1).load()[0].page_content
    )

    # TODO-去除参考文献和致谢需改进
    pdf_text_without_references = remove_references(pdf_text)
    print("\nLength of original text:", len(pdf_text))
    print("Length of text without references:", len(pdf_text_without_references))

    # 输出提取的文本
    with open("extracted_text.txt", "w", encoding="utf-8") as f:
        f.write(pdf_text_without_references)

    # 将文本切分为chunks
    chunks = split_into_chunks(pdf_text_without_references, chunk_size=2000)
    top_n_chunks = get_most_relevant_chunk(user_question, chunks, n=5)

    print("最相关的文献片段：", top_n_chunks)
    response = answer_question(user_question, top_n_chunks)

    print("Response：", response)

    end_time_2 = time.time()
    elaspsed_time_2 = end_time_2 - start_time_2
    print("第二阶段耗时/s：", elaspsed_time_2)

    print("总耗时/s：", elaspsed_time_1 + elaspsed_time_2)
