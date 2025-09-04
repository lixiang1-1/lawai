import openai

# openai.api_base = "https://api.openai.com/v1" # 换成代理，一定要加v1
openai.api_base = "https://openkey.cloud/v1" # 换成代理，一定要加v1
# openai.api_key = "API_KEY"
openai.api_key = "sk-zazShpXus8VzmGAYWudqMB9Ye1K2iz1dpp0r8EbKMR69J2ga"

for resp in openai.ChatCompletion.create(
                                    model="gpt-3.5-turbo",
                                    messages=[
                                      {"role": "user", "content": "证明费马大定理"}
                                    ],
                                    # 流式输出
                                    stream = True):
    if 'content' in resp.choices[0].delta:
        print(resp.choices[0].delta.content, end="", flush=True)