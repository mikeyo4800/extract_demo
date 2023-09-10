from fastapi import FastAPI
import helper
from models import Email, FacebookPost
import content

app = FastAPI()


@app.post("/extract")
async def extract(email: Email):
    
    content = email.content
    
    response = helper.extract_email(content)
    
    arguments = response.choices[0]["message"]["function_call"]["arguments"]
    
    name = eval(arguments).get("name")
    category = eval(arguments).get("category")
    summary = eval(arguments).get("summary")
    priority = eval(arguments).get("priority")
    
    return {"name": name, "category": category, "summary": summary, "priority": priority}


@app.post("/create")
async def create(fb: FacebookPost):
    
    niche = fb.niche
    company = fb.company
    website = fb.website
    topic = fb.topic
    title = fb.title
    
    response = content.create_fb_content(niche, company, website, topic)
    
    return {"content": response}
    
    
    
    