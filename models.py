from pydantic import BaseModel

class Email(BaseModel):
    content: str
    
    
class FacebookPost(BaseModel):
    niche: str
    company: str
    website: str
    topic: str
    title: str


