# Intro

This is Repo is made for local RAG testing and various implementation

### Let's learn folder structure

**.venv** - This is name of virtaul enviroment

if you are on windows run to activate virtaul enviroment

```
.Venv\Scripts\activate
```

if you have docker & docker compose installed then run in the directory containing docker-compose.yaml folder

```
docker-compose up -d
```

This project uses uv for dependiencies manangement [Go to Docs](https://docs.astral.sh/uv/guides/projects/)

**data** - This is directory contains raw pdf, reranker on qdrant, images extracted from pdf & reranked ran on faiss

**qrant_storage** - This is directory contains all the data required to store by qdrant vector store

**utils** - This is directory contains all the _imporant_ testing done

### In utils folder

This folder contains all tests related to RAG all in one file and their function is represented by their name except `test_rag.py` & `test_rag_v2.py` so the difference is just for testing ranker

### Future Testing:

- test clip modal
- test different embedding modal other than following benchmarks
- test various reranking strategies  
# testing
