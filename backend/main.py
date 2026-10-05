import os, re, uuid
from pathlib import Path
import fitz, requests
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from google import genai

BASE=Path(__file__).resolve().parent.parent
DATA=BASE/'data'; DATA.mkdir(exist_ok=True)
app=FastAPI(title='DocLens AI Beta',version='0.2.0')
app.add_middleware(CORSMiddleware,allow_origins=['*'],allow_methods=['*'],allow_headers=['*'])
docs={}; indexes={}

class Req(BaseModel):
    document_id:str
    question:str=''
class CompareReq(BaseModel):
    document_ids:list[str]
    question:str

def clean(s): return re.sub(r'\s+',' ',s or '').strip()

def chunks(pages,size=750,overlap=120):
    out=[]
    for page,text in pages:
        words=text.split(); start=0
        while start<len(words):
            end=min(len(words),start+size)
            out.append({'id':len(out),'page':page,'text':' '.join(words[start:end])})
            if end==len(words): break
            start=max(start+1,end-overlap)
    return out

def index(doc_id):
    cs=docs[doc_id]['chunks']; v=TfidfVectorizer(stop_words='english',ngram_range=(1,2),max_features=30000)
    m=v.fit_transform([c['text'] for c in cs]) if cs else None
    indexes[doc_id]=(v,m)

def retrieve(doc_id,q,k=6):
    if doc_id not in indexes:return []
    v,m=indexes[doc_id]
    if m is None:return []
    scores=cosine_similarity(v.transform([q]),m).ravel(); ids=scores.argsort()[::-1][:k]
    return [{**docs[doc_id]['chunks'][i],'score':float(scores[i])} for i in ids if scores[i]>0]

def gemini_generate(prompt):
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        print("GEMINI_API_KEY is not configured.")
        return None

    try:
        client = genai.Client(api_key=api_key)

        response = client.models.generate_content(
            model="gemini-3.8-flash",
            contents=prompt
        )

        for candidate in response.candidates or []:
            if not candidate.content:
                continue

            for part in candidate.content.parts or []:
                if getattr(part, "text", None):
                    return part.text.strip()

        return None

    except Exception as e:
        print("Gemini error:", e)
        return None

def summary_fallback(cs,n=8):
    text=' '.join(c['text'] for c in cs); ss=re.split(r'(?<=[.!?])\s+',text)
    freq={}
    for w in re.findall(r'[A-Za-z]{4,}',text.lower()):freq[w]=freq.get(w,0)+1
    scored=[]
    for i,s in enumerate(ss):
        ws=re.findall(r'[A-Za-z]{4,}',s.lower()); scored.append((sum(freq.get(w,0) for w in ws)/max(1,len(ws)),i,s))
    return ' '.join(x[2] for x in sorted(sorted(scored,reverse=True)[:n],key=lambda x:x[1]))

def stats(text):
    words=re.findall(r'\b\w+\b',text); return {'words':len(words),'characters':len(text),'sentences':len(re.findall(r'[.!?]+',text))}

@app.get('/')
def home():return FileResponse(BASE/'frontend/index.html')
@app.get('/app.js')
def js():return FileResponse(BASE/'frontend/app.js',media_type='application/javascript')
@app.get('/styles.css')
def css():return FileResponse(BASE/'frontend/styles.css',media_type='text/css')

@app.post('/api/upload')
async def upload(file:UploadFile=File(...)):
    if not file.filename.lower().endswith('.pdf'):raise HTTPException(400,'Only PDF files are supported.')
    did=str(uuid.uuid4()); path=DATA/f'{did}.pdf'; path.write_bytes(await file.read())
    pdf=fitz.open(path); pages=[]
    for i,p in enumerate(pdf): pages.append((i+1,clean(p.get_text('text'))))
    cs=chunks(pages); alltext=' '.join(t for _,t in pages)
    docs[did]={'id':did,'name':file.filename,'pages':len(pdf),'pages_text':{str(p):t for p,t in pages},'chunks':cs,'stats':stats(alltext)}; index(did)
    return {'id':did,'name':file.filename,'pages':len(pdf),'chunks':len(cs),'stats':docs[did]['stats']}

@app.get('/api/documents')
def documents():return [{'id':d['id'],'name':d['name'],'pages':d['pages'],'chunks':len(d['chunks']),'stats':d['stats']} for d in docs.values()]

@app.post('/api/ask')
def ask(r:Req):
    if r.document_id not in docs:raise HTTPException(404,'Document not found')
    src=retrieve(r.document_id,r.question,6); context='\n\n'.join(f'[Page {s["page"]}] {s["text"]}' for s in src)
    prompt=f'''You are DocLens, a precise PDF analysis assistant. Answer ONLY from the supplied context. If unsupported, say the document does not provide enough information. Cite every important claim with [Page N].\n\nCONTEXT:\n{context}\n\nQUESTION:\n{r.question}'''
    answer=gemini_generate(prompt)
    if not answer: answer='Relevant passages found. No local LLM is currently available, so generation is disabled.\n\n'+('\n\n'.join(f'[Page {s["page"]}] {s["text"][:900]}' for s in src) if src else 'No relevant passage found.')
    return {'answer':answer,'sources':[{'page':s['page'],'score':round(s['score'],3),'text':s['text'][:900]} for s in src]}

@app.post('/api/summarize')
def summarize(r:Req):
    if r.document_id not in docs:raise HTTPException(404,'Document not found')
    d=docs[r.document_id]; context='\n\n'.join(f'[Page {c["page"]}] {c["text"]}' for c in d['chunks'][:40])
    answer=gemini_generate(f'Summarize this PDF using only the context. Cover purpose, key points, findings and conclusions. Include page citations.\n{context}') or summary_fallback(d['chunks'])
    return {'summary':answer}

@app.post('/api/compare')
def compare(r:CompareReq):
    if len(r.document_ids)<2:raise HTTPException(400,'Select at least two documents')
    for did in r.document_ids:
        if did not in docs:raise HTTPException(404,'Document not found')
    blocks=[]
    for did in r.document_ids:
        src=retrieve(did,r.question,4); blocks.append(f'DOCUMENT: {docs[did]["name"]}\n'+'\n'.join(f'[Page {s["page"]}] {s["text"]}' for s in src))
    prompt=f'''Compare the supplied documents for the user's request. Keep each document clearly attributed. Use only supplied context and cite page numbers. Do not invent missing information.\n\n{chr(10).join(blocks)}\n\nREQUEST: {r.question}'''
    answer=gemini_generate(prompt)
    if not answer: answer='Local LLM unavailable. Retrieved passages for comparison:\n\n'+('\n\n'.join(blocks))
    return {'answer':answer}

@app.get('/api/documents/{did}/pages/{page}')
def page(did:str,page:int):
    if did not in docs:raise HTTPException(404,'Document not found')
    text=docs[did]['pages_text'].get(str(page))
    if text is None:raise HTTPException(404,'Page not found')
    return {'page':page,'text':text}
