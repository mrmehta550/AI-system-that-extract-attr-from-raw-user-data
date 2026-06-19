from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_community.document_loaders import CSVLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser,PydanticOutputParser
from pydantic import BaseModel, ValidationError, Field
from typing import List, Optional
from fastapi import FastAPI, HTTPException
import os
from pydantic.experimental.missing_sentinel import MISSING

load_dotenv()

app = FastAPI(title="AI system")

class ProductAttributes(BaseModel):
    Material :Optional[str] = None
    Style :Optional[str] = None
    Color : list[str] =Field(default_factory=list)
    Dimensions : Optional[str] =None
    Assembly_Required : Optional[str] =None
    Category :Optional[str] =None
# try:
#     ProductAttributes(f=1)
# except ValidationError as exc:
#     print(repr(exc.errors()[0]['type']))
#     #> 'missing_sentinel_error'
    
    
parse = PydanticOutputParser(
    pydantic_object = ProductAttributes
)

prompt = PromptTemplate(
    template="""
You are a furniture attribute extraction system.

Extract these attributes:

- Material
- Style
- Color
- Dimensions
- Assembly_Required
- Category

Rules:
1. If description contains "no assembly required" return Assembly_Required = "No"
2. If description contains "requires assembly" return Assembly_Required = "Yes"
3. Detect furniture category such as:
   Coffee Table
   Sofa
   Chair
   Dining Table
   Bookshelf
   Bar Stool
   TV Stand
   Bed Frame

4. Return ONLY JSON.

{format_instructions}

Description:
{description}
""",
    input_variables=["description"],
    partial_variables={
        "format_instructions": parse.get_format_instructions()
    }
)

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text_generation",
    temperature=0.0001,
    huggingfacehub_api_token=os.getenv("HF_TOKEN")
    )
loader = CSVLoader(file_path="furniture_dataset.csv")
docs= loader.load()
description = docs[0].page_content
model = ChatHuggingFace(llm=llm)
chain = prompt | model | parse
# result = chain.invoke({"description":"Mid-century walnut wood coffee table with tapered legs, 48x24 inches, no assembly required"})
# print(result)

class ExtractionRequest(BaseModel):
    description: str
    
@app.post("/extract", response_model=ProductAttributes)
async def extract_attributes(request:ExtractionRequest):
    if not request.description.strip():
        raise HTTPException(status_code=400, detail="Description text cannot be empty.")
    try:
        result = chain.invoke({"description":request.description})
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction pipeline failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app,host="127.0.0.1", port=8000)        
