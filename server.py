from flask import Flask
from flask import render_template
from flask import request
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.http.models import PointStruct
import tqdm
import openai
import os


app = Flask(__name__)
#使用qdrant本地模式
client = QdrantClient(":memory:") # Create in-memory Qdrant instance
collection_name = "data_collection"

def to_embeddings(items):
    openai.api_base = "https://openkey.cloud/v1" # 换成代理，一定要加v1
    openai.api_key = "sk-zazShpXus8VzmGAYWudqMB9Ye1K2iz1dpp0r8EbKMR69J2ga"
    sentence_embeddings = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=items[1]
    )
    return [items[0], items[1], sentence_embeddings["data"][0]["embedding"]]

def prompt(question, answers):
    """
    生成对话的示例提示语句，格式如下：
    demo_q:
    使用以下段落来回答问题，如果段落内容不相关就返回未查到相关信息："成人头疼，流鼻涕是感冒还是过敏？"
    1. 朋友借我的身份证贷款，但是我的身份证已经失信了，可朋友又用我的身份证贷款X万至今没有还？"，你好，如果是你签名或按手印，你是债务人。或者有证据表明贷款资金实际部分或全部为你所用，你应该承担偿还义务。如果不如期偿还，银行可以通过法院强制执行。还款后向其追偿。
    demo_a:
    朋友借你的身份证贷款，你的身份证已经失信了，现在信用社让你还，如果是你签名或按手印，你是债务人。或者有证据表明贷款资金实际部分或全部为你所用，你应该承担偿还义务。如果不如期偿还，银行可以通过法院强制执行。还款后向其追偿。
    system:
    你是一个专业律师
    """
    demo_q = '使用以下段落来回答问题："朋友借我的身份证贷款，但是我的身份证已经失信了，可朋友又用我的身份证贷款X万至今没有还？"\n你好，如果是你签名或按手印，你是债务人。或者有证据表明贷款资金实际部分或全部为你所用，你应该承担偿还义务。如果不如期偿还，银行可以通过法院强制执行。还款后向其追偿。'
    demo_a = '朋友借你的身份证贷款，你的身份证已经失信了，现在信用社让你还，如果是你签名或按手印，你是债务人。或者有证据表明贷款资金实际部分或全部为你所用，你应该承担偿还义务。如果不如期偿还，银行可以通过法院强制执行。还款后向其追偿。'
    system = '你是一个专业律师'
    #q = '使用以下段落来回答问题，如果段落内容不相关就返回未查到相关信息："'
    q = '使用以下段落来回答问题："'
    q += question + '"'
    # 带有索引的格式
    for index, answer in enumerate(answers):
        q += str(index + 1) + '. ' + str(answer['title']) + ': ' + str(answer['text']) + '\n'

    """
    system:代表的是你要让GPT生成内容的方向，在这个案例中我要让GPT生成的内容是医院问诊机器人的回答，所以我把system设置为医院问诊机器人
    前面的user和assistant是我自己定义的，代表的是用户和律师机器人的示例对话，主要规范输入和输出格式
    下面的user代表的是实际的提问
    """
    # res = [
    #     {'role': 'system', 'content': system},
    #     {'role': 'user', 'content': demo_q},
    #     {'role': 'assistant', 'content': demo_a},
    #     {'role': 'user', 'content': q},
    # ]

    res = [
        {'role': 'system', 'content': system},
        {'role': 'user', 'content': q},
    ]
    print("===res:=== "+str(res))
    return res


def query(text):
    """
    执行逻辑：
    首先使用openai的Embedding API将输入的文本转换为向量
    然后使用Qdrant的search API进行搜索，搜索结果中包含了向量和payload
    payload中包含了title和text，title是疾病的标题，text是摘要
    最后使用openai的ChatCompletion API进行对话生成
    """
    #client = QdrantClient("127.0.0.1", port=6333)
    #client = QdrantClient(":memory:") # Create in-memory Qdrant instance
    #collection_name = "data_collection"
    openai.api_base = "https://openkey.cloud/v1" # 换成代理，一定要加v1
    openai.api_key = "sk-zazShpXus8VzmGAYWudqMB9Ye1K2iz1dpp0r8EbKMR69J2ga"
    sentence_embeddings = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=text
    )
    print("===query text embedding:=== "+str(sentence_embeddings["data"][0]["embedding"]))
    """
    因为提示词的长度有限，所以我只取了搜索结果的前三个，如果想要更多的搜索结果，可以把limit设置为更大的值
    """
    search_result = client.search(
        collection_name=collection_name,
        query_vector=sentence_embeddings["data"][0]["embedding"],
        limit=1,
        search_params={"exact": False, "hnsw_ef": 128}
    )
    print("===search result:=== "+str(search_result))
    answers = []
    tags = []

    """
    因为提示词的长度有限，每个匹配的相关摘要我在这里只取了前300个字符，如果想要更多的相关摘要，可以把这里的300改为更大的值
    """
    for result in search_result:
        if len(result.payload["text"]) > 300:
            summary = result.payload["text"][:300]
        else:
            summary = result.payload["text"]
        answers.append({"title": result.payload["title"], "text": summary})

    completion = openai.ChatCompletion.create(
        temperature=0.7,
        model="gpt-3.5-turbo",
        messages=prompt(text, answers),
    )
    print("===completion:=== "+str(completion))
    return {
        "answer": completion.choices[0].message.content,
        "tags": tags,
    }


@app.route('/')
def hello_world():
    return render_template('index.html')


@app.route('/search', methods=['POST'])
def search():
    data = request.get_json()
    search = data['search']
    print("====search:===="+search)
    res = query(search)

    return {
        "code": 200,
        "data": {
            "search": search,
            "answer": res["answer"],
            "tags": res["tags"],
        },
    }


if __name__ == '__main__':

    # openai.api_key = os.getenv("OPENAI_API_KEY")
    openai.api_base = "https://openkey.cloud/v1" # 换成代理，一定要加v1
    openai.api_key = "sk-zazShpXus8VzmGAYWudqMB9Ye1K2iz1dpp0r8EbKMR69J2ga"
    # 创建collection
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
    )

    count = 0
    for root, dirs, files in os.walk("./source_data"):
        for file in tqdm.tqdm(files):
            file_path = os.path.join(root, file)
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                parts = text.split('#####')
                item = to_embeddings(parts)
                print("===item==="+item[0])
                client.upsert(
                    collection_name=collection_name,
                    wait=True,
                    points=[
                        PointStruct(id=count, vector=item[2], payload={"title": item[0], "text": item[1]}),
                    ],
                )
            count += 1
    #启动flask app
    app.run(host='0.0.0.0', port=3000)
