import streamlit as st
from openai import OpenAI

# 页面配置
st.set_page_config(page_title="医疗专家诊断系统", page_icon="🩺")
st.title("🩺 医疗思维链专家系统")

# 初始化客户端 (指向本地 LLaMA Factory API)
client = OpenAI(api_key="0", base_url="http://localhost:8000/v1")

# 【专家级系统提示词】
SYSTEM_PROMPT = """你是一位中医外科权威专家。在分析病例时，请遵循以下原则：
1. 空间定位：头皮病症应优先考虑外科疮疡，而非内科疳积。
2. 特征匹配：若体征包含“皮下空洞”、“状如蝼蛄穿掘”，这是“蝼蛄疖”的唯一金标准。
3. 逻辑严密：必须在<think>标签内进行鉴别诊断，排除掉相似但错误的病名。"""

# 【逻辑纠偏：Few-shot 引导】
# 哪怕模型练得不够深，这两组对话也能强行把它的思维定死在正确逻辑上
FEW_SHOT_EXAMPLES = [
    {"role": "user", "content": "1岁患儿夏季头皮出现多处小结节，溃破流脓，有空洞，皮肤增厚。诊断是什么？"},
    {"role": "assistant", "content": "<think>症状点：1. 幼儿夏季发病；2. 位在头皮；3. 关键体征为皮下空洞。排除：疳积无穿掘性空洞。结论：蝼蛄疖。</think>最终诊断：蝼蛄疖。"}
]

# 初始化会话历史
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    st.session_state.messages.extend(FEW_SHOT_EXAMPLES)

# 显示历史（隐藏系统提示和示范）
for message in st.session_state.messages[3:]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 处理用户输入
if user_input := st.chat_input("请输入病例描述..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        
        # 核心参数设置
        responses = client.chat.completions.create(
            model="DeepSeek-R1-Distill-Qwen-7B",
            messages=st.session_state.messages,
            stream=True,
            temperature=0.0,      # 【关键】设为 0 彻底消除随机性，防止它胡思乱想
            max_tokens=600,       # 限制长度防止复读
            presence_penalty=1.2, # 惩罚重复
            stop=["<｜endoftext｜>", "###"] 
        )

        for response in responses:
            token = response.choices[0].delta.content
            if token:
                full_response += token
                response_placeholder.markdown(full_response + "▌")
        
        response_placeholder.markdown(full_response)
    
    st.session_state.messages.append({"role": "assistant", "content": full_response})