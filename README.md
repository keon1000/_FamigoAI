# Famigo AI


# Application Description
This project is a voice assistant application that identifies users through speech recognition (STT) and face recognition (planned; currently using user_id), and provides personalized responses using Retrieval-Augmented Generation (RAG) technology.
Real-time voice input via microphone is converted into text
Conversation history is summarized and embedded per user to maintain personal memory
New queries are embedded, and the top 8 from group members, top 5 from user most similar summaries from memory are used as context for the prompt

# Team Members
Kihun Lee (Team Lead), keon1000@g.skku.edu
Hansol Park, firstri@g.skku.edu
Junwon Moon, mppn98@g.skku.edu    
Chanwoo Park, cksdn1290@korea.ac.kr


## how to run:

* with uv

```shell
uv sync
uv run streamlit run app.py
```

* with pip

```shell
pip install -r requirements.txt
streamlit run app.py
```

## setup db
```shell
createdb voice_face_rag
export DATABASE_URL="postgresql+psycopg://postgres:postgres@localhost:5432/voice_face_rag"
alembic -c app/db/migrations/alembic.ini upgrade head
```

 


  
