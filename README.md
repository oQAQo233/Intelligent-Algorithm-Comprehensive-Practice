# 项目使用手册
- （需要提前安装cuda环境）

## 配置环境

```cmd
git clone https://github.com/oQAQo233/Intelligent-Algorithm-Comprehensive-Practice.git
cd Intelligent-Algorithm-Comprehensive-Practice
python -m venv .venv
.venv\Scripts\activate

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
pip install -e .
```

### 此外还需要提前在根目录下创建.env文件，至少包含以下信息

- DATABASE_NAME
- GRAPH_URL
- GRAPH_USERNAME
- GRAPH_PASSWORD

- LOCAL_MODEL_NAME
- LOCAL_BASE_URL
- SILICONFLOW_API

- VITE_ALIYUN_BAILIAN_API_KEY
- VITE_ALIYUN_BAILIAN_API_URL
- VITE_ALIYUN_BAILIAN_MODEL

## 前端
- （需要提前安装bun环境）

```cmd
cd frontend
bun install
bun run dev
```

## 后端

```
cd src/FASTAPI_FrameWork
python APIRun.py
```
