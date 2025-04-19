import os
import re
import logging
import requests
import arxiv
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

web_url = "https://cvpr.thecvf.com/Conferences/2025/AcceptedPapers"

web_response = requests.get(web_url, verify=True)
if web_response.status_code != 200:
    print(f"网页获取失败，状态码：{web_response.status_code}")
    exit()

soup = BeautifulSoup(web_response.text, 'html.parser')
paper_titles = [title.get_text(strip=True) for title in soup.find_all('strong')]
#这里需要修改成你需要获得的网站资源
if not paper_titles:
    print("未找到任何论文标题")
    exit()

api_url = "https://api.siliconflow.cn/v1/chat/completions"
api_key = "输入你自己的api"

prompt = "你是一个专业的学术信息提取员,请从以下论文标题中筛选出30个与 三维生成模型 直接或者间接相关的标题，以列表形式(注意：格式要保持一致)返回:直接返回txt格式 不要markdown形式 不用附上任何理解 我只要符合三维生成模型的论文题目\n"
for title in paper_titles:
    prompt += f"- {title}\n"

payload = {
    "model": "Qwen/Qwen2.5-VL-72B-Instruct",
    "stream": False,
    "max_tokens": 512,
    "temperature": 0.7,
    "top_p": 0.7,
    "top_k": 50,
    "frequency_penalty": 0.5,
    "n": 1,
    "stop": [],
    "messages": [
        {
            "role": "user",
            "content": prompt
        }
    ]
}

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

try:
    response = requests.post(api_url, json=payload, headers=headers)
    response.raise_for_status()
    resultAi = response.json()
    relevant_titles = resultAi["choices"][0]["message"]["content"]
    QueryList = relevant_titles.split('\n')
    QueryList = [s.lstrip('- ').strip() for s in QueryList if s.strip()]
    
except requests.exceptions.HTTPError as e:
    print(f"API请求失败, 状态码: {e.response.status_code}")
    print(f"错误详情：{e.response.text}")
except Exception as e:
    print(f"发生异常：{str(e)}") 
    


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(BASE_DIR, "NewlyDownloadPapers")
os.makedirs(output_dir, exist_ok=True)

client = arxiv.Client()

for title in QueryList:  
    try:
        search = arxiv.Search( 
            query=title, 
            max_results=10, 
            sort_by=arxiv.SortCriterion.Relevance
        )
        result = next(client.results(search), None)
        if result is None:
            print(f"未在 arxiv 上找到：'{title}'")
            continue
        safe_title = re.sub(r'[\\/:\*\?"<>\|]', "_", result.title)
        file_path = os.path.join("NewlyDownloadPapers", f"{safe_title}.pdf")
        pdf_url = result.pdf_url
        pdf_resp = requests.get(pdf_url, timeout=20)
        pdf_resp.raise_for_status()
        with open(file_path, "wb") as f:
            f.write(pdf_resp.content)
        print(f"下载成功：《{result.title}》保存为 {file_path}")
    except Exception as e:  
        print(f"下载《{title}》时出错: {e}")
    except Exception as e:
        print(f"处理 '{title}' 时发生异常: {str(e)}")
